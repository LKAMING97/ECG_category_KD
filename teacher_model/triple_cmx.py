# -*- coding: utf-8 -*-

"""
@Time     : "2022/8/6 9:25"
@Author   : "liao guixin"
@Location : "GDUT Lab406"
@File     : "triple_cmx.py"
@Software : "PyCharm"
"""
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -----------------------------------------------------------------------#
#    Conv-Mixer 网络
#    论文地址：https://openreview.net/pdf?id=TVHS5Y4dNvM
#    我的博客 ：https://blog.csdn.net/qq_38676487/article/details/120705254
# -------------------------------------------------------------------------#
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
        self.avg = nn.AvgPool1d(3)

        self.fc_age_gender = nn.Linear(5, 32)

        self.fc = nn.Linear(dim+32, n_classes)

    def select(self, data, lead_list):
        # lead_list = [0, 1, 2, 3, 4, 5]
        data = data.cpu()
        data_list = list()
        for i in lead_list:
            data_i = data[:, i, :]
            data_i = data_i[np.newaxis, :, :]
            data_list.append(data_i)
        data_list = np.vstack(data_list)
        data_list = data_list.swapaxes(0, 1)
        data_list = torch.from_numpy(data_list).cuda()
        return data_list

    def forward(self, x, fr):
        x_1 = self.select(x, [0, 1, 2, 3, 4, 5, 6])
        x_1 = x_1.unsqueeze(1)
        x_2 = self.select(x, [0, 1, 2, 3, 4, 5, 7])
        x_2 = x_2.unsqueeze(1)
        x_3 = self.select(x, [0, 1, 2, 3, 4, 5, 8])
        x_3 = x_3.unsqueeze(1)

        x_list = [x_1, x_2, x_3]
        x_head_out = list()
        for i in range(0, 3):
            x = x_list[i]
            x = self.conv2d1(x)

            for ConvMixer_block in self.ConvMixer_blocks:
                x = ConvMixer_block(x)

            x = self.head(x)
            x_head_out.append(x)
        x_fr = self.fc_age_gender(fr)
        x_head_out = torch.cat((x_head_out[0], x_head_out[1], x_head_out[2]), dim=1)
        x_head_out = self.avg(x_head_out)

        x_inp = torch.cat([x_head_out, x_fr], dim=1)
        y_cls = self.fc(x_inp)

        return y_cls


# -----------------------------------------------------------------------#
#    论文中给出的配置：
#    ConvMixer_h_d   h：dim 隐藏层维度  d：depth 网络深度
# -------------------------------------------------------------------------#
def ConvMixer_1536_20(n_classes=1000):
    return ConvMixer(dim=1536, depth=20, patch_size=7, kernel_size=9, n_classes=n_classes)


def ConvMixer_768_32(n_classes=1000):
    return ConvMixer(dim=768, depth=20, patch_size=7, kernel_size=7, n_classes=n_classes)


# 自定义的 ConvMixer 不传参 为 ConvMixer_768_32
def custom_ConvMixer(dim=768, depth=32, patch_size=7, kernel_size=7, n_classes=1000):
    return ConvMixer(dim=dim, depth=depth, patch_size=patch_size, kernel_size=kernel_size, n_classes=n_classes)


if __name__ == '__main__':
    model = ConvMixer_768_32(20).to(device)
    # summary(teacher_model, (1, 12, 5000))
    a = torch.randn((8, 12, 5000)).cuda()
    b = torch.randn((8, 5)).cuda()
    c = model(a, b)
    print(c.shape)


