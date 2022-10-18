# -*- coding: utf-8 -*-

"""
@Time     : "2022/6/28 19:42"
@Author   : "liao guixin"
@Location : "GDUT Lab406"
@File     : "HRNN.py"
@Software : "PyCharm"
"""

import torch
import torch.nn as nn
from torchsummaryX import summary


class baseblock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, block_num):
        super(baseblock, self).__init__()
        self.block_num = block_num
        self.conv = nn.Conv1d(in_channel, out_channel, kernel)

    def forward(self, x):
        for i in range(self.block_num):
            res = x
            x = self.conv(x)
            x = self.conv(x)
            x = x + res
        return x


class HRnn(nn.Module):
    def __init__(self, in_channel):
        super(HRnn, self).__init__()

        self.conv_1 = nn.Conv1d(in_channel, 64, 25, 2)
        self.pool = nn.MaxPool1d(3, 2)
        self.block_1 = baseblock(64, 64, 11, 3)

        self.conv_2 = nn.Conv1d(64, 128, 11, 2)
        self.conv_3 = nn.Conv1d(128, 128, 11)
        self.block_2 = baseblock(128, 128, 11, 3)

        self.conv_4 = nn.Conv1d(128, 256, 7, 2)
        self.conv_5 = nn.Conv1d(256, 256, 7)
        self.block_3 = baseblock(256, 256, 7, 5)

        self.conv_6 = nn.Conv1d(256, 512, 7, 2)
        self.conv_7 = nn.Conv1d(512, 512, 7)
        self.block_4 = baseblock(512, 512, 7, 2)

    def _make_layer1d(self, block, planes, blocks, stride=2, size=3, res=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, padding=0, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, size=size, res=res))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, size=size, res=res))

        return nn.Sequential(*layers)

    def forward(self, x):
        x_1 = self.conv_1(x)
        x_1 = self.pool(x_1)
        x_1 = self.block_1(x_1)

        x_2 = self.conv_2(x_1)
        x_2 = self.conv_3(x_2)
        x_2 = self.block_2(x_2)

        x_3 = self.conv_4(x_2)
        x_3 = self.conv_5(x_3)
        x_3 = self.block_3(x_3)

        x_4 = self.conv_6(x_3)
        x_4 = self.conv_7(x_4)
        x_4 = self.block_4(x_4)

        return x_4


if __name__ == '__main__':
    net = HRnn(12).cuda()
    a = torch.randn((16, 12, 5000)).cuda()
    summary(net, a)





