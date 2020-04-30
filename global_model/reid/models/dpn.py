""" PyTorch implementation of DualPathNetworks
Based on original MXNet implementation https://github.com/cypw/DPNs with
many ideas from another PyTorch implementation https://github.com/oyam/pytorch-DPNs.
This implementation is compatible with the pretrained weights
from cypw's MXNet implementation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from collections import OrderedDict
import os
from .adaptive_avgmax_pool import adaptive_avgmax_pool2d

logger = logging.getLogger(__name__)
__all__ = ['DPN', 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn131', 'dpn107']


model_urls = {
    'dpn68':
        'https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn68-66bebafa7.pth',
    'dpn68b-extra':
        'https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn68b_extra-84854c156.pth',
    'dpn92-extra':
        'https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn92_extra-b040e4a9b.pth',
    'dpn98':
        'https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn98-5b90dec4d.pth',
    'dpn131':
        'https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn131-71dfe43e0.pth',
    'dpn107-extra':
        'https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn107_extra-1ac7121e2.pth'
}


def dpn68(pretrained=False, test_time_pool=False, **kwargs):
    """Constructs a DPN-68 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet-1K
        test_time_pool (bool): If True, pools features for input resolution beyond
            standard 224x224 input with avg+max at inference/validation time
        **kwargs : Keyword args passed to model __init__
            num_classes (int): Number of classes for classifier linear layer, default=1000
    """
    model = DPN(
        small=True, num_init_features=10, k_r=128, groups=32,
        k_sec=(3, 4, 12, 3), inc_sec=(16, 32, 32, 64),
        test_time_pool=test_time_pool, **kwargs)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(model_urls['dpn68']))
    return model


def dpn68b(pretrained=False, test_time_pool=False, **kwargs):
    """Constructs a DPN-68b model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet-1K
        test_time_pool (bool): If True, pools features for input resolution beyond
            standard 224x224 input with avg+max at inference/validation time
        **kwargs : Keyword args passed to model __init__
            num_classes (int): Number of classes for classifier linear layer, default=1000
    """
    model = DPN(
        small=True, num_init_features=10, k_r=128, groups=32,
        b=True, k_sec=(3, 4, 12, 3), inc_sec=(16, 32, 32, 64),
        test_time_pool=test_time_pool, **kwargs)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(model_urls['dpn68b-extra']))
    return model


def dpn92(pretrained=False, test_time_pool=False, **kwargs):
    """Constructs a DPN-92 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet-1K
        test_time_pool (bool): If True, pools features for input resolution beyond
            standard 224x224 input with avg+max at inference/validation time
        **kwargs : Keyword args passed to model __init__
            num_classes (int): Number of classes for classifier linear layer, default=1000
    """
    model = DPN(
        num_init_features=64, k_r=96, groups=32,
        k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128),
        test_time_pool=test_time_pool, **kwargs)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(model_urls['dpn92-extra']))
    return model


def dpn98(pretrained=False, test_time_pool=False, **kwargs):
    """Constructs a DPN-98 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet-1K
        test_time_pool (bool): If True, pools features for input resolution beyond
            standard 224x224 input with avg+max at inference/validation time
        **kwargs : Keyword args passed to model __init__
            num_classes (int): Number of classes for classifier linear layer, default=1000
    """
    model = DPN(
        num_init_features=96, k_r=160, groups=40,
        k_sec=(3, 6, 20, 3), inc_sec=(16, 32, 32, 128),
        test_time_pool=test_time_pool, **kwargs)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(model_urls['dpn98']))
    return model


def dpn131(pretrained=False, test_time_pool=False, **kwargs):
    """Constructs a DPN-131 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet-1K
        test_time_pool (bool): If True, pools features for input resolution beyond
            standard 224x224 input with avg+max at inference/validation time
        **kwargs : Keyword args passed to model __init__
            num_classes (int): Number of classes for classifier linear layer, default=1000
    """
    model = DPN(
        num_init_features=128, k_r=160, groups=40,
        k_sec=(4, 8, 28, 3), inc_sec=(16, 32, 32, 128),
        test_time_pool=test_time_pool, **kwargs)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(model_urls['dpn131']))
    return model


def dpn107(pretrained=True, test_time_pool=False, **kwargs):
    """Constructs a DPN-107 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet-1K
        test_time_pool (bool): If True, pools features for input resolution beyond
            standard 224x224 input with avg+max at inference/validation time
        **kwargs : Keyword args passed to model __init__
            num_classes (int): Number of classes for classifier linear layer, default=1000
    """
    model = DPN(
        num_init_features=128, k_r=200, groups=50,
        k_sec=(4, 8, 20, 3), inc_sec=(20, 64, 64, 128),
        test_time_pool=test_time_pool, **kwargs)
    if pretrained:
        # model.load_state_dict(load_state_dict_from_url(model_urls['dpn107-extra']))
        model.init_weights(pretrained = './pretrain_models/dpn107_extra-1ac7121e2.pth')

    return model


class CatBnAct(nn.Module):
    def __init__(self, in_chs, activation_fn=nn.ReLU(inplace=True)):
        super(CatBnAct, self).__init__()
        self.bn = nn.BatchNorm2d(in_chs, eps=0.001)
        self.act = activation_fn

    def forward(self, x):
        x = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        return self.act(self.bn(x))


class BnActConv2d(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride,
                 padding=0, groups=1, activation_fn=nn.ReLU(inplace=True)):
        super(BnActConv2d, self).__init__()
        self.bn = nn.BatchNorm2d(in_chs, eps=0.001)
        self.act = activation_fn
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, groups=groups, bias=False)

    def forward(self, x):
        return self.conv(self.act(self.bn(x)))


class InputBlock(nn.Module):
    def __init__(self, num_init_features, kernel_size=7,
                 padding=3, activation_fn=nn.ReLU(inplace=True)):
        super(InputBlock, self).__init__()
        self.conv = nn.Conv2d(
            3, num_init_features, kernel_size=kernel_size, stride=2, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(num_init_features, eps=0.001)
        self.act = activation_fn
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class DualPathBlock(nn.Module):
    def __init__(
            self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, groups, block_type='normal', b=False):
        super(DualPathBlock, self).__init__()
        self.num_1x1_c = num_1x1_c
        self.inc = inc
        self.b = b
        if block_type == 'proj':
            self.key_stride = 1
            self.has_proj = True
        elif block_type == 'down':
            self.key_stride = 2
            self.has_proj = True
        else:
            assert block_type == 'normal'
            self.key_stride = 1
            self.has_proj = False

        if self.has_proj:
            # Using different member names here to allow easier parameter key matching for conversion
            if self.key_stride == 2:
                self.c1x1_w_s2 = BnActConv2d(
                    in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=2)
            else:
                self.c1x1_w_s1 = BnActConv2d(
                    in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=1)
        self.c1x1_a = BnActConv2d(in_chs=in_chs, out_chs=num_1x1_a, kernel_size=1, stride=1)
        self.c3x3_b = BnActConv2d(
            in_chs=num_1x1_a, out_chs=num_3x3_b, kernel_size=3,
            stride=self.key_stride, padding=1, groups=groups)
        if b:
            self.c1x1_c = CatBnAct(in_chs=num_3x3_b)
            self.c1x1_c1 = nn.Conv2d(num_3x3_b, num_1x1_c, kernel_size=1, bias=False)
            self.c1x1_c2 = nn.Conv2d(num_3x3_b, inc, kernel_size=1, bias=False)
        else:
            self.c1x1_c = BnActConv2d(in_chs=num_3x3_b, out_chs=num_1x1_c + inc, kernel_size=1, stride=1)

    def forward(self, x):
        x_in = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        if self.has_proj:
            if self.key_stride == 2:
                x_s = self.c1x1_w_s2(x_in)
            else:
                x_s = self.c1x1_w_s1(x_in)
            x_s1 = x_s[:, :self.num_1x1_c, :, :]
            x_s2 = x_s[:, self.num_1x1_c:, :, :]
        else:
            x_s1 = x[0]
            x_s2 = x[1]
        x_in = self.c1x1_a(x_in)
        x_in = self.c3x3_b(x_in)
        if self.b:
            x_in = self.c1x1_c(x_in)
            out1 = self.c1x1_c1(x_in)
            out2 = self.c1x1_c2(x_in)
        else:
            x_in = self.c1x1_c(x_in)
            out1 = x_in[:, :self.num_1x1_c, :, :]
            out2 = x_in[:, self.num_1x1_c:, :, :]
        resid = x_s1 + out1
        dense = torch.cat([x_s2, out2], dim=1)
        return resid, dense


class DPN(nn.Module):
    def __init__(self, small=False, num_init_features=64, k_r=96, groups=32,
                 b=False, k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128),
                 num_classes=1000, num_features=2048,test_time_pool=False):
        super(DPN, self).__init__()
        self.test_time_pool = test_time_pool
        self.b = b
        bw_factor = 1 if small else 4

        blocks = OrderedDict()

        # conv1
        if small:
            blocks['conv1_1'] = InputBlock(num_init_features, kernel_size=3, padding=1)
        else:
            blocks['conv1_1'] = InputBlock(num_init_features, kernel_size=7, padding=3)

        # conv2
        bw = 64 * bw_factor
        inc = inc_sec[0]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv2_1'] = DualPathBlock(num_init_features, r, r, bw, inc, groups, 'proj', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[0] + 1):
            blocks['conv2_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc

        # conv3
        bw = 128 * bw_factor
        inc = inc_sec[1]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv3_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[1] + 1):
            blocks['conv3_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc

        # conv4
        bw = 256 * bw_factor
        inc = inc_sec[2]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv4_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[2] + 1):
            blocks['conv4_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc

        # conv5
        bw = 512 * bw_factor
        inc = inc_sec[3]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv5_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[3] + 1):
            blocks['conv5_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        blocks['conv5_bn_ac'] = CatBnAct(in_chs)

        self.features = nn.Sequential(blocks)
        self.num_classes = num_classes
        self.num_features = num_features
        # Using 1x1 conv for the FC layer to allow the extra pooling scheme
        # self.classifier = nn.Conv2d(in_chs, num_classes, kernel_size=1, bias=True)
        self.dropout = nn.Dropout()
        self.dim_red_conv = nn.Conv2d(2688, self.num_features, 1, bias=False)
        nn.init.kaiming_normal_(self.dim_red_conv.weight.data, mode='fan_out')

        self.dim_red_bn = nn.BatchNorm2d(self.num_features)
        self.dim_red_bn.weight.data.fill_(1)
        self.dim_red_bn.bias.data.zero_()

        self.fc = nn.Linear(self.num_features, self.num_classes, True)

        nn.init.normal_(self.fc.weight, std=0.001)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        x = self.features(x)
        # if not self.training and self.test_time_pool:
        #     x = F.avg_pool2d(x, kernel_size=7, stride=1)
        #     out = self.classifier(x)
        #     # The extra test time pool should be pooling an img_size//32 - 6 size patch
        #     out = adaptive_avgmax_pool2d(out, pool_type='avgmax')
        # else:
        #     x = adaptive_avgmax_pool2d(x, pool_type='avg')
        #     out = self.classifier(x)
        # return out.view(out.size(0), -1)
        if not self.training:
            x = nn.functional.avg_pool2d(x, x.size()[2:])
            x = self.dropout(x)
            x = self.dim_red_conv(x)
            x = x.div(x.norm(2, 1, keepdim=True).add(1e-8).expand_as(x))
            x = x.view(x.size(0), -1)
            return x

        else:
            x = nn.functional.avg_pool2d(x, x.size()[2:])
            x = self.dropout(x)
            x = self.dim_red_conv(x)
            x = self.dim_red_bn(x)
            x_g = x
            x = x.contiguous().view(-1, self.num_features)
            out = self.fc(x)
            return out,x_g



    def init_weights(self, pretrained='',):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            print('load pretrained_model')
            pretrained_dict = torch.load(pretrained)
            
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                # print('=> loading {} pretrained model {}'.format(k, pretrained))
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict = False)