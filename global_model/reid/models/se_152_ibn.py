import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import math
from .ibnblock import *
from torch.nn import Parameter

__all__ = ['se_152_ibn']

class Cross_Trihard_Seresnet(nn.Module):

    def __init__(self, num_classes, num_features):
        super(Cross_Trihard_Seresnet, self).__init__()
        self.base = SENet(block=SEBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=1)	
        self.num_classes = num_classes
        self.num_features = num_features

        self.dropout = nn.Dropout()
        self.dim_red_conv = nn.Conv2d(512 * SEResNeXtBottleneck.expansion, self.num_features, 1, bias=False)
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
        x = self.dim_red_conv(x)

        if self.training:
            # x_g = x
            # x = self.dim_red_bn(x)
            x = self.dim_red_bn(x)
            x_g = x
            x = x.contiguous().view(-1, self.num_features)
            x = self.fc(x)
            return x, x_g
        else:
            x = x.div(x.norm(2, 1, keepdim=True).add(1e-8).expand_as(x))
            x = x.view(x.size(0), -1)
            return x

def se_152_ibn(*args, **kwargs):
    print('se_resnet152_ibn')
    model = Cross_Trihard_Seresnet(*args, **kwargs)
    return model


