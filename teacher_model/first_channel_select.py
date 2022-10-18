# -*- coding: utf-8 -*-

"""
@Time     : "2022/9/20 18:50"
@Author   : "liao guixin"
@Location : "GDUT Lab406"
@File     : "first_channel_select.py"
@Software : "PyCharm"
"""

import torch
import torch.nn as nn
from torchsummary import summary
from torch.nn import functional as F
from thop import profile, clever_format

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# SGU模块
class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)

    def forward(self, x):
        # 将x按维度拆分为两个部分，将一部分进行线性投影与另一部分进行元素点乘，达到门控通道选择的作用
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v)
        out = u * v
        return out


# gMLP模块
class gMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        self.channel_proj2 = nn.Linear(d_ffn, d_model)
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.gelu(self.channel_proj1(x))
        x = self.sgu(x)
        x = self.channel_proj2(x)
        out = x + residual
        return out


class gMLP(nn.Module):
    def __init__(self, d_model=256, d_ffn=512, seq_len=256, num_layers=6):
        super().__init__()
        self.model = nn.Sequential(
            *[gMLPBlock(d_model, d_ffn, seq_len) for _ in range(num_layers)]
        )

    def forward(self, x):
        return self.model(x)


# convmixer层
class ConvMixerLayer(nn.Module):
    def __init__(self, dim, kernel_size=9):
        super().__init__()
        # 残差结构
        self.Resnet = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, groups=dim, padding='same'),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        # 逐点卷积
        self.Conv_1x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        x = x + self.Resnet(x)
        x = self.Conv_1x1(x)
        return x


class ConvMixer(nn.Module):
    def __init__(self, dim, depth, kernel_size=9, patch_size=7, n_classes=1000):
        super().__init__()
        self.conv2d1 = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        self.ConvMixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.ConvMixer_blocks.append(ConvMixerLayer(dim=dim, kernel_size=kernel_size))

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.fc_age_gender = nn.Linear(5, 32)

        self.fc = nn.Linear(dim + 32, n_classes)

    def forward(self, x, fr):
        x = x.unsqueeze(1)
        x = self.conv2d1(x)

        for ConvMixer_block in self.ConvMixer_blocks:
            x = ConvMixer_block(x)

        x = self.head(x)
        x_fr = self.fc_age_gender(fr)
        x_inp = torch.cat([x, x_fr], dim=1)
        y_cls = self.fc(x_inp)

        return y_cls

    def get_classifier_weight(self):
        return self.fc.weight.data, self.fc.bias.data

    def set_classifier_weight(self, weight, bias):
        self.fc.weight.data[:] = weight
        self.fc.bias.data[:] = bias

    def get_classifier_params(self):
        return self.fc.parameters()


class SGU_ConvMixer(nn.Module):
    def __init__(self, input_channel, embed_dim, seq_len, gMlp_layer, kernel_size, patch_size, ConvMixer_layer,
                 n_classes):
        super(SGU_ConvMixer, self).__init__()
        self.channel_select = gMLP(12, embed_dim, seq_len, gMlp_layer)

        self.conv2d1 = nn.Sequential(
            nn.Conv2d(input_channel, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(embed_dim)
        )
        self.ConvMixer_blocks = nn.ModuleList([])

        for _ in range(ConvMixer_layer):
            self.ConvMixer_blocks.append(ConvMixerLayer(dim=embed_dim, kernel_size=kernel_size))

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.fc_age_gender = nn.Linear(5, 32)

        self.fc = nn.Linear(embed_dim + 32, n_classes)

    def forward(self, x, fr, is_feat):
        batch_size, num_channels, _ = x.shape
        x = x.permute(0, 2, 1)
        x_s = self.channel_select(x)

        x_in = x_s.permute(0, 2, 1).unsqueeze(1)

        x_patch = self.conv2d1(x_in)

        for ConvMixer_block in self.ConvMixer_blocks:
            x_patch = ConvMixer_block(x_patch)

        x_cm = self.head(x_patch)
        x_fr = self.fc_age_gender(fr)
        x_inp = torch.cat([x_cm, x_fr], dim=1)
        y_cls = self.fc(x_inp)

        if is_feat:
            return [x_inp], y_cls
        else:
            return y_cls

    def get_classifier_weight(self):
        return self.fc.weight.data, self.fc.bias.data

    def set_classifier_weight(self, weight, bias):
        self.fc.weight.data[:] = weight
        self.fc.bias.data[:] = bias

    def get_classifier_params(self):
        return self.fc.parameters()


class SGU_ConvMixer_change(nn.Module):
    def __init__(self, input_channel, CM_embed_dim, gMlp_embed_dim, seq_len, gMlp_layer, kernel_size, patch_size,
                 ConvMixer_layer,
                 n_classes):
        super(SGU_ConvMixer_change, self).__init__()
        self.channel_select = gMLP(CM_embed_dim, gMlp_embed_dim, seq_len, gMlp_layer)

        self.conv2d1 = nn.Sequential(
            nn.Conv2d(input_channel, CM_embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(CM_embed_dim)
        )
        self.ConvMixer_blocks = nn.ModuleList([])

        for _ in range(ConvMixer_layer):
            self.ConvMixer_blocks.append(ConvMixerLayer(dim=CM_embed_dim, kernel_size=kernel_size))

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.fc_age_gender = nn.Linear(5, 32)

        self.fc = nn.Linear(CM_embed_dim + 32, n_classes)

    def forward(self, x, fr, is_feat):
        x = x.unsqueeze(2)
        x_patch = self.conv2d1(x)

        batch_size, num_channels, _, seq_len = x_patch.shape
        x_patch = x_patch.permute(0, 2, 3, 1).view(batch_size, -1, num_channels)
        x_s = self.channel_select(x_patch)
        x_s = x_s.view(batch_size, -1, seq_len, num_channels)
        x_s = x_s.permute(0, 3, 1, 2)

        for ConvMixer_block in self.ConvMixer_blocks:
            x_s = ConvMixer_block(x_s)

        x_cm = self.head(x_s)
        x_fr = self.fc_age_gender(fr)
        x_inp = torch.cat([x_cm, x_fr], dim=1)
        y_cls = self.fc(x_inp)

        if is_feat:
            return [x_inp], y_cls
        else:
            return y_cls


# -----------------------------------------------------------------------#
#    论文中给出的配置：
#    ConvMixer_h_d   h：dim 隐藏层维度  d：depth 网络深度
# -------------------------------------------------------------------------#
def ConvMixer_1536_20(n_classes=1000):
    return ConvMixer(dim=1536, depth=20, patch_size=7, kernel_size=9, n_classes=n_classes)


def ConvMixer_768_32(n_classes=1000):
    return ConvMixer(dim=768, depth=32, patch_size=12, kernel_size=12, n_classes=n_classes)


# 自定义的 ConvMixer 不传参 为 ConvMixer_768_32
def custom_ConvMixer(dim=768, depth=32, patch_size=7, kernel_size=7, n_classes=1000):
    return ConvMixer(dim=dim, depth=depth, patch_size=patch_size, kernel_size=kernel_size, n_classes=n_classes)


if __name__ == '__main__':
    # teacher_model = ConvMixer(dim=768, depth=32, patch_size=(1, 500), kernel_size=(1, 500), n_classes=20).cuda()
    # teacher_model = SGU_ConvMixer_change(12, 768, 256, 714, 6, 7, (1, 7), 32, 20).cuda()
    model = SGU_ConvMixer(1, 768, 5000, 1, (12, 7), (12, 7), 32, 20).cuda()
    flops, params = profile(model, inputs=(torch.randn(32, 12, 5000).cuda(), torch.randn(32, 5).cuda()))

    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    a = torch.randn((2, 12, 5000)).cuda()
    b = torch.randn((2, 5)).cuda()
    c = model(a, b)
    print(c.shape)
