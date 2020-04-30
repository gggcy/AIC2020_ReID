import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import math

__all__ = ['cross_entropy_trihard_resnet18', 'cross_entropy_trihard_resnet34', 'cross_entropy_trihard_resnet50', 'cross_entropy_trihard_resnet101','cross_entropy_trihard_resnet152']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNetBase(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNetBase, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

class Cross_Entropy_Trihard_ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, num_features):
        super(Cross_Entropy_Trihard_ResNet, self).__init__()
        self.base = ResNetBase(block, layers, num_classes)
        self.num_classes = num_classes
        self.num_features = num_features

        self.dropout = nn.Dropout()
        self.dim_red_conv = nn.Conv2d(512 * block.expansion, self.num_features, 1, bias=False)
        nn.init.kaiming_normal_(self.dim_red_conv.weight.data, mode='fan_out')

        self.dim_red_bn = nn.BatchNorm2d(self.num_features)
        self.dim_red_bn.weight.data.fill_(1)
        self.dim_red_bn.bias.data.zero_()

        self.fc = nn.Linear(self.num_features, self.num_classes, True)
        nn.init.normal_(self.fc.weight, std=0.001)
        nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, x):
        x = self.base(x)
        x = nn.functional.avg_pool2d(x, x.size()[2:])
        x = self.dropout(x)
        if self.training:
            x = self.dim_red_conv(x)
            # x_g = x
            # x = self.dim_red_bn(x)
            x = self.dim_red_bn(x)
            x_g = x

            x = x.contiguous().view(-1, self.num_features)
            x = self.fc(x)
            return x, x_g
        else:
            x = self.dim_red_conv(x)
            x = x.div(x.norm(2, 1, keepdim=True).add(1e-8).expand_as(x))
            x = x.view(x.size(0), -1)
            return x


def cross_entropy_trihard_resnet18(pretrained=False, **kwargs):
    model = Cross_Entropy_Trihard_ResNet(BasicBlock, [2,2,2,2], **kwargs)
    return model

def cross_entropy_trihard_resnet34(pretrained=False, **kwargs):
    model = Cross_Entropy_Trihard_ResNet(BasicBlock, [3,4,6,3], **kwargs)
    return model

def cross_entropy_trihard_resnet50(pretrained=False, **kwargs):
    model = Cross_Entropy_Trihard_ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def cross_entropy_trihard_resnet101(pretrained=False, **kwargs):
    model = Cross_Entropy_Trihard_ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def cross_entropy_trihard_resnet152(pretrained=False, **kwargs):
    model = Cross_Entropy_Trihard_ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
