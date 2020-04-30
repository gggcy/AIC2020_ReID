import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import math

__all__ = ['multi_attribute_8_resnet152']


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        print(classname)
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        print(classname)
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        print(classname)
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m, is_bias=True):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        print(classname)
        nn.init.normal_(m.weight, std=0.001)
        if is_bias:
            nn.init.constant_(m.bias, 0.0)

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

class Multi_Attribute_ResNet_3(nn.Module):

    def __init__(self, block, layers, num_classes, num_features, is_cls=True, test_attribute=False):
        super(Multi_Attribute_ResNet_3, self).__init__()
        self.base = ResNetBase(block, layers, num_classes)
        self.num_classes = num_classes
        self.num_features = num_features
        self.is_cls = is_cls
        self.test_attribute = test_attribute

        self.dropout = nn.Dropout()
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

        self.dim_red_conv = nn.Conv2d(512 * block.expansion, self.num_features, 1, bias=False)
        self.dim_red_conv.apply(weights_init_kaiming)
        self.dim_red_bn = nn.BatchNorm2d(self.num_features)
        self.dim_red_bn.apply(weights_init_kaiming)
        
        self.cls_task_conv = nn.Conv2d(self.num_features, 512, 1, bias=False)
        self.cls_task_conv.apply(weights_init_kaiming)
        self.cls_task_bn = nn.BatchNorm2d(512)
        self.cls_task_bn.apply(weights_init_kaiming)
        
        self.type_task_conv = nn.Conv2d(self.num_features, 512, 1, bias=False)
        self.type_task_conv.apply(weights_init_kaiming)
        self.type_task_bn = nn.BatchNorm2d(512)
        self.type_task_bn.apply(weights_init_kaiming)

        self.color_task_conv = nn.Conv2d(self.num_features, 512, 1, bias=False)
        self.color_task_conv.apply(weights_init_kaiming)
        self.color_task_bn = nn.BatchNorm2d(512)
        self.color_task_bn.apply(weights_init_kaiming)


        self.roof_task_conv = nn.Conv2d(self.num_features, 512, 1, bias=False)
        self.roof_task_conv.apply(weights_init_kaiming)
        self.roof_task_bn = nn.BatchNorm2d(512)
        self.roof_task_bn.apply(weights_init_kaiming)

        self.window_task_conv = nn.Conv2d(self.num_features, 512, 1, bias=False)
        self.window_task_conv.apply(weights_init_kaiming)
        self.window_task_bn = nn.BatchNorm2d(512)
        self.window_task_bn.apply(weights_init_kaiming)

        self.logo_task_conv = nn.Conv2d(self.num_features, 512, 1, bias=False)
        self.logo_task_conv.apply(weights_init_kaiming)
        self.logo_task_bn = nn.BatchNorm2d(512)
        self.logo_task_bn.apply(weights_init_kaiming)

        self.cat_red_conv = nn.Conv2d(3072, self.num_features, 1, bias=False)
        self.cat_red_conv.apply(weights_init_kaiming)
        self.cat_red_bn = nn.BatchNorm2d(self.num_features)
        self.cat_red_bn.apply(weights_init_kaiming)
        
        self.cls_fc = nn.Linear(self.num_features, self.num_classes, True)
        self.cls_fc.apply(weights_init_classifier)

        self.type_fc = nn.Linear(512, 8, True)
        self.type_fc.apply(weights_init_classifier)
        self.color_fc = nn.Linear(512, 10, True)
        self.color_fc.apply(weights_init_classifier)
        self.roof_fc = nn.Linear(512, 2, True)
        self.roof_fc.apply(weights_init_classifier)
        self.window_fc = nn.Linear(512, 3, True)
        self.window_fc.apply(weights_init_classifier)
        self.logo_fc = nn.Linear(512, 30, True)
        self.logo_fc.apply(weights_init_classifier)

    def forward(self, x):
        x = self.base(x)
        x = nn.functional.avg_pool2d(x, x.size()[2:])
        x = self.dropout(x)
        x = self.dim_red_conv(x)
        x = self.dim_red_bn(x)
        x = self.relu(x)        
   
        cls_x = self.cls_task_conv(x)
        cls_x = self.cls_task_bn(cls_x)
        
        type_x = self.type_task_conv(x)
        type_x = self.type_task_bn(type_x)

        color_x = self.color_task_conv(x)
        color_x = self.color_task_bn(color_x)
        
        roof_x = self.roof_task_conv(x)
        roof_x = self.roof_task_bn(roof_x)

        window_x = self.window_task_conv(x)
        window_x = self.window_task_bn(window_x)

        logo_x = self.logo_task_conv(x)
        logo_x = self.logo_task_bn(logo_x)

        cat_x = torch.cat([cls_x, type_x, color_x, roof_x, window_x, logo_x], 1)
        cat_x = self.cat_red_conv(cat_x)
        # cat_x = self.cat_red_bn(cat_x)

        x_g = cat_x

        if self.training:
            cat_x = self.cat_red_bn(cat_x)
            cat_x = cat_x.contiguous().view(-1, self.num_features)
            type_x = type_x.contiguous().view(-1, 512)
            type_x = self.type_fc(type_x)
            color_x = color_x.contiguous().view(-1, 512)
            color_x = self.color_fc(color_x)
            roof_x = roof_x.contiguous().view(-1, 512)
            roof_x = self.roof_fc(roof_x)
            window_x = window_x.contiguous().view(-1, 512)
            window_x = self.window_fc(window_x)
            logo_x = logo_x.contiguous().view(-1, 512)
            logo_x = self.logo_fc(logo_x)
            if self.is_cls:
                final_cls_x = self.cls_fc(cat_x)
                return final_cls_x, color_x, type_x, roof_x, window_x, logo_x, x_g

                #return final_cls_x, type_x, x_g
            else:
                #return type_x, x_g
                return color_x ,type_x, roof_x , window_x, logo_x, x_g
                
                
        else:
            if self.test_attribute:
                print('test attribute')
                type_x = type_x.contiguous().view(-1, 512)
                type_x = self.type_fc(type_x)
                type_x = self.softmax(type_x)
                return type_x
            else:
                x = x_g
                x = x.div(x.norm(2, 1, keepdim=True).add(1e-8).expand_as(x))
                x = x.view(x.size(0), -1)
                return x

def multi_attribute_8_resnet152(pretrained=False, **kwargs):
    print('type model')
    model = Multi_Attribute_ResNet_3(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
