import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath

from libs.backbone.pvtv2 import pvt_v2_b2


# ---------- 通用模块定义 ----------

class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super(DSConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, dilation=dilation, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class ChannelSE(nn.Module):
    def __init__(self, channels, reduction=8):
        super(ChannelSE, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x)
        return x * scale

# GBEM（Gated Bidirectional Enhancement Module）
class GBEM(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels):
        super(GBEM, self).__init__()

        self.dsconv = DSConv(in_channels_high, out_channels)
        self.dsconv_low = DSConv(in_channels_low, out_channels)

        self.se_high = ChannelSE(out_channels)
        self.se_low = ChannelSE(out_channels)

        self.gate_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.fusion = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(self, F_high, F_low):
        F_high_up = F.interpolate(F_high, size=F_low.shape[2:], mode='bilinear', align_corners=True)

        gamma_high = self.se_high(self.dsconv(F_high_up))
        beta_high = self.se_high(self.dsconv(F_high_up))

        gamma_low = self.se_low(self.dsconv_low(F_low))
        beta_low = self.se_low(self.dsconv_low(F_low))

        F_h2l = F_low * (gamma_high + 1) + beta_high
        F_l2h = F_high_up * (gamma_low + 1) + beta_low

        fusion_input = torch.cat([F_h2l, F_l2h], dim=1)
        gate = self.gate_conv(fusion_input)
        fused = gate * F_h2l + (1 - gate) * F_l2h

        out = self.fusion(fused + F_l2h)
        return self.act(out)
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, drop=0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = int(hidden_features) or in_features
        self.norm = norm_layer(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)    # LayerNorm
        x = self.fc1(x)
        x = self.act(x)     # GELU
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size**2, C)
    return windows
# SCAM（Shifted Cooperative Attention Module）
class SCAM(nn.Module):
    def __init__(self, input_channels=32, shift_size=0, window_size=8):
        super(SCAM, self).__init__()
        self.query_transform = nn.Conv2d(input_channels, input_channels, 1)
        self.key_transform = nn.Conv2d(input_channels, input_channels, 1)
        self.value_conv = nn.Conv2d(input_channels, input_channels, 1)
        self.scale = 1.0 / (input_channels ** 0.5)
        self.ffn = MLP(input_channels, hidden_features=input_channels)  # 多层感知机
        drop_rate = 0.1
        self.drop_path = DropPath(drop_prob=drop_rate)  #
        self.multi_scale_conv = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels // 2, input_channels, 1)
        )
        self.window_size = window_size
        self.channel_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channels, input_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels // 16, input_channels, 1),
            nn.Sigmoid()
        )
        self.QK = nn.Conv2d(input_channels,input_channels*2,1)
        self.out_conv = nn.Conv2d(input_channels, input_channels, 1)

        self.shift_size = shift_size
    def check_size(self, x, shift=False):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

        if shift:
            x = F.pad(x, (self.shift_size, (self.window_size - self.shift_size + mod_pad_w) % self.window_size,
                          self.shift_size, (self.window_size - self.shift_size + mod_pad_h) % self.window_size),
                      mode='reflect')
        else:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x




# ---------- 完整模型实现 ----------

class ABENet(nn.Module):
    def __init__(self, channel=32, imagenet_pretrained=True):
        super(ABENet, self).__init__()
        # 骨干网络
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = r'E:\example\FCNet_master\pth\backbone\pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        # 特征调整层 - 将骨干网络各阶段特征调整到统一维度
        self.reduce_1 = nn.Conv2d(64, channel, 1)
        self.reduce_2 = nn.Conv2d(128, channel, 1)
        self.reduce_3 = nn.Conv2d(320, channel, 1)
        self.reduce_4 = nn.Conv2d(512, channel, 1)

        # 注意力层
        self.co_att = SCAM(channel,shift_size=0)

        # 双向特征交互模块
        self.dbiem_43 = GBEM(channel, channel, channel)  # 高层到中层
        self.dbiem_32 = GBEM(channel, channel, channel)  # 中层到低层
        self.dbiem_21 = GBEM(channel, channel, channel)  # 低层到超低层

        # 自底向上的特征融合路径
        self.up_12 = nn.Sequential(
            nn.ConvTranspose2d(channel, channel, kernel_size=2, stride=2),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.up_23 = nn.Sequential(
            nn.ConvTranspose2d(channel, channel, kernel_size=2, stride=2),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.up_34 = nn.Sequential(
            nn.ConvTranspose2d(channel, channel, kernel_size=2, stride=2),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # 预测头
        self.predict_head = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, kernel_size=1)
        )

    def forward(self, x):
        # 骨干网络特征提取
        pvt = self.backbone(x)
        x1_s = pvt[0]  # 10, 64, 64, 64
        x2_s = pvt[1]  # 10, 128, 32, 32
        x3_s = pvt[2]  # 10, 320, 16, 16
        x4_s = pvt[3]  # 10, 512, 8, 8

        # 特征维度统一
        x1 = self.reduce_1(x1_s)  # 64 -> 32
        x2 = self.reduce_2(x2_s)  # 128 -> 32
        x3 = self.reduce_3(x3_s)  # 320 -> 32
        x4 = self.reduce_4(x4_s)  # 512 -> 32

        # 应用协同注意力机制
        x4_att = self.co_att(x4)

        # 双向特征交互 - 自顶向下路径
        x3_enhanced = self.dbiem_43(x4_att, x3)  # 高层增强中层
        x2_enhanced = self.dbiem_32(x3_enhanced, x2)  # 中层增强低层
        x1_enhanced = self.dbiem_21(x2_enhanced, x1)  # 低层增强超低层
        x2_enhanced = F.interpolate(x2_enhanced,self.up_12(x1_enhanced).shape[2:],mode='bilinear')
        # 自底向上融合路径
        x2_fused = x2_enhanced + self.up_12(x1_enhanced)
        x3_enhanced = F.interpolate(x3_enhanced,self.up_23(x2_fused).shape[2:],mode='bilinear')

        x3_fused = x3_enhanced + self.up_23(x2_fused)
        x4_att = F.interpolate(x4_att,self.up_34(x3_fused).shape[2:],mode='bilinear')

        x4_fused = x4_att + self.up_34(x3_fused)

        # 最终预测
        out = self.predict_head(x4_fused)
        out = F.interpolate(out, size=x.size()[2:], mode='bilinear', align_corners=True)

        return out


# 模型实例化和前向传播测试
if __name__ == "__main__":
    model = ABENet(channel=32)
    x = torch.randn(2, 3,256,256,)
    output = model(x)
    print(f"输出尺寸: {output.shape}")
