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
        self.fc_age_gender = nn.Linear(5, 32)

        self.fc = nn.Linear(dim + 32, n_classes)

    def forward(self, x, fr, is_feat=False):
        x = x.unsqueeze(1)
        x = self.conv2d1(x)

        for ConvMixer_block in self.ConvMixer_blocks:
            x = ConvMixer_block(x)

        x = self.head(x)
        x_fr = self.fc_age_gender(fr)
        x_inp = torch.cat([x, x_fr], dim=1)
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

    # def shuffle_last_conv_classifier(self):
    #     last_block = self.layer4[-1]
    #     if isinstance(last_block, BasicBlock):
    #         conv = last_block.conv2
    #         bn = last_block.bn2
    #     elif isinstance(last_block, Bottleneck):
    #         conv = last_block.conv3
    #         bn = last_block.bn3
    #     feat_dim = len(bn.weight)
    #     idx = torch.randperm(feat_dim)
    #     conv.weight.data = conv.weight.data[idx]
    #     # conv.bias.data = conv.bias.data[idx]
    #     bn.weight.data = bn.weight.data[idx]
    #     bn.bias.data = bn.bias.data[idx]
    #     bn.running_mean.data = bn.running_mean.data[idx]
    #     bn.running_var.data = bn.running_var.data[idx]
    #     self.fc.weight.data = self.fc.weight.data.t()[idx].t()


# -----------------------------------------------------------------------#
#    论文中给出的配置：
#    ConvMixer_h_d   h：dim 隐藏层维度  d：depth 网络深度
# -------------------------------------------------------------------------#
def ConvMixer_1536_20(n_classes=1000):
    return ConvMixer(dim=1536, depth=20, patch_size=8, kernel_size=8, n_classes=n_classes)


def ConvMixer_768_32(n_classes=1000):
    return ConvMixer(dim=768, depth=32, patch_size=7, kernel_size=7, n_classes=n_classes)


# 自定义的 ConvMixer 不传参 为 ConvMixer_768_32
def custom_ConvMixer(dim=768, depth=32, patch_size=7, kernel_size=7, n_classes=1000):
    return ConvMixer(dim=dim, depth=depth, patch_size=patch_size, kernel_size=kernel_size, n_classes=n_classes)


if __name__ == '__main__':
    model = ConvMixer_768_32(20).to(device)
    a = torch.randn((1, 12, 5000)).cuda()
    b = torch.randn((1, 5)).cuda()
    c = model(a, b)
    print(c.shape)
    # summary(teacher_model, (12, 5000))
