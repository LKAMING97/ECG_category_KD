# ECGNet_201911091150
import pickle

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.nn import Parameter
from torchsummaryX import summary
import torch


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


def conv_2d(in_planes, out_planes, stride=(1, 1), size=3):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, size), stride=stride,
                     padding=(0, (size - 1) // 2), bias=False)


def conv_1d(in_planes, out_planes, stride=1, size=3):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=size, stride=stride,
                     padding=(size - 1) // 2, bias=False)


class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, size=3, res=True):
        super(BasicBlock1d, self).__init__()
        self.conv1 = conv_1d(inplanes, planes, stride, size=size)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_1d(planes, planes, size=size)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv_1d(planes, planes, size=size)
        self.bn3 = nn.BatchNorm1d(planes)
        self.dropout = nn.Dropout(.2)
        self.downsample = downsample
        self.stride = stride
        self.res = res

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.res:
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
        # out = self.relu(out)

        return out


class BasicBlock2d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1, 1), downsample=None, size=3, res=True):
        super(BasicBlock2d, self).__init__()
        self.conv1 = conv_2d(inplanes, planes, stride, size=size)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_2d(planes, planes, size=size)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv_2d(planes, planes, size=size)
        self.bn3 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(.2)
        self.downsample = downsample
        self.stride = stride
        self.res = res

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.res:
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
        # out = self.relu(out)

        return out


class ECGNet(nn.Module):
    def __init__(self, input_channel=1, num_classes=20):  # , layers=[2, 2, 2, 2, 2, 2]
        sizes = [
            [3, 3, 3, 3, 3, 3],
            [5, 5, 5, 5, 3, 3],
            [7, 7, 7, 7, 3, 3],
        ]
        self.sizes = sizes
        layers = [
            [3, 3, 2, 2, 2, 2],
            [3, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2]
        ]

        super(ECGNet, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channel, 32, kernel_size=(1, 50), stride=(1, 2), padding=(0, 0),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 2), padding=(0, 0),
                               bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=(1,16), stride=(1,2), padding=(0,0),
        #                       bias=False)
        # print(self.conv2)
        # self.bn3 = nn.BatchNorm2d(32)
        # self.dropout = nn.Dropout(.2)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 0))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.num_classes = num_classes

        self.layers1_list = nn.ModuleList()
        self.layers2_list = nn.ModuleList()
        for i, size in enumerate(sizes):
            self.inplanes = 32
            self.layers1 = nn.Sequential()
            self.layers2 = nn.Sequential()
            self.layers1.add_module('layer{}_1_1'.format(size),
                                    self._make_layer2d(BasicBlock2d, 32, layers[i][0], stride=(1, 1), size=sizes[i][0]))
            self.layers1.add_module('layer{}_1_2'.format(size),
                                    self._make_layer2d(BasicBlock2d, 32, layers[i][1], stride=(1, 1), size=sizes[i][1]))
            self.inplanes *= 12
            self.layers2.add_module('layer{}_2_1'.format(size),
                                    self._make_layer1d(BasicBlock1d, 256, layers[i][2], stride=2, size=sizes[i][2]))
            self.layers2.add_module('layer{}_2_2'.format(size),
                                    self._make_layer1d(BasicBlock1d, 256, layers[i][3], stride=2, size=sizes[i][3]))
            self.layers2.add_module('layer{}_2_3'.format(size),
                                    self._make_layer1d(BasicBlock1d, 256, layers[i][4], stride=2, size=sizes[i][4]))
            self.layers2.add_module('layer{}_2_4'.format(size),
                                    self._make_layer1d(BasicBlock1d, 256, layers[i][5], stride=2, size=sizes[i][5]))

            self.layers1_list.append(self.layers1)
            self.layers2_list.append(self.layers2)

        adj_path = "F:/lgx/category_v7/data/adj_file.pkl"

        self.gc1 = GraphConvolution(num_classes, 1024)
        self.gc2 = GraphConvolution(1024, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)

        data_adj = pickle.load(open(adj_path, 'rb'))
        self.A_data = torch.from_numpy(data_adj).float().cuda()
        self.t = 10

        # self.drop = nn.Dropout(p=0.2)
        self.fc_age = nn.Linear(5, 64)
        self.fc = nn.Linear(256 * len(sizes)+64, num_classes)

    def gen_adj(self, A, t=0):
        # print(A[1,:])
        A[A < t] = 0
        D = torch.pow(A.sum(1).float(), -0.5)
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(A, D).t(), D)
        return adj

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

    def _make_layer2d(self, block, planes, blocks, stride=(1, 2), size=3, res=True):
        downsample = None
        if stride != (1, 1) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=(1, 1), padding=(0, 0), stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, size=size, res=res))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, size=size, res=res))

        return nn.Sequential(*layers)

    def forward(self, x0, fr):
        x0 = x0.unsqueeze(1)
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)
        x0 = self.conv2(x0)
        # x0 = self.bn2(x0)
        # x0 = self.relu(x0)
        x0 = self.maxpool(x0)
        # x0 = self.dropout(x0)

        xs = []
        for i in range(len(self.sizes)):
            # print(self.layers1_list[i])
            x = self.layers1_list[i](x0)
            x = torch.flatten(x, start_dim=1, end_dim=2)
            x = self.layers2_list[i](x)
            x = self.avgpool(x)
            xs.append(x)
        out = torch.cat(xs, dim=2)
        out = out.view(out.size(0), -1)
        fr = self.fc_age(fr)
        out = torch.cat([out, fr], dim=1)
        y_cls = self.fc(out)

        y_inp = out.unsqueeze(2)
        y_feature = y_inp.repeat((1, 1, self.num_classes)).permute(0, 2, 1)

        xita = self.fc.weight.permute(1, 0)
        y_inp = torch.matmul(y_feature, xita)

        data_adj = self.gen_adj(self.A_data, self.t)

        x_gcn = self.gc1(y_inp, data_adj)  # (34, 576)
        x_gcn = self.leakyrelu(x_gcn)
        y_gcn = self.gc2(x_gcn, data_adj)  # (34, 576)
        y_gcn = y_gcn.view(y_gcn.size(0), -1)
        y = y_cls + y_gcn

        return y_cls


if __name__ == "__main__":
    net = ECGNet(1, 20).cuda()
    a = torch.randn((2, 12, 5000)).cuda()
    b = torch.randn((2, 5)).cuda()
    c = net(a, b)
    print(c.shape)
    # summary(net, a)

