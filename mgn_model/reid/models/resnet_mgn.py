import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from torch.nn import init




import pdb

from torch.nn.init import constant_ as init_constant
from torch.nn.init import kaiming_normal_ as init_kaiming_normal
from torch.nn.init import normal_ as init_normal

__all__ = [ 'ResNet50_mgn_lr','ResNet101_mgn_lr','ResNet152_mgn_lr']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    # 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
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


class ResNet_mgn_lr(nn.Module):
    __factory = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3]
    }


    def __init__(self, depth, num_classes=283, num_features=1024, norm=True):

        super(ResNet_mgn_lr, self).__init__()
        block = Bottleneck
        if depth not in ResNet_mgn_lr.__factory:
            raise KeyError("Unsupported depth:", depth)
        layers = ResNet_mgn_lr.__factory[depth]
        self.inplanes = 64


        self.num_features = num_features
        self.norm = norm
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.layer3_0 = self._make_layer(block, 256, layers[2], stride=2,is_last=False)
        self.layer3_1 = self._make_layer(block, 256, layers[2], stride=2,is_last=False)
        self.layer3_lr = self._make_layer(block, 256, layers[2], stride=2,is_last=False)
        self.layer3_2 = self._make_layer(block, 256, layers[2], stride=2,is_last=True)
        
        self.layer4_0 = self._make_layer(block, 512, layers[3], stride=2,is_last=False)
        self.layer4_1 = self._make_layer(block, 512, layers[3], stride=1,is_last=False)
        self.layer4_lr = self._make_layer(block, 512, layers[3], stride=1,is_last=False)
        self.layer4_2 = self._make_layer(block, 512, layers[3], stride=1,is_last=True)

        # self.classifier_g1 = nn.Linear(256, num_classes)
        # self.g1_reduce = self._cbr2()

        # self.classifier_g2 = nn.Linear(256, num_classes)
        # self.g2_reduce = self._cbr2()
        # self.classifier_p2_1 = nn.Linear(256, num_classes)
        # self.p2_1_reduce = self._cbr2()
        # self.classifier_p2_2 = nn.Linear(256, num_classes)
        # self.p2_2_reduce = self._cbr2()


        # self.classifier_g2_lr = nn.Linear(256, num_classes)
        # self.g2_reduce_lr = self._cbr2()
        # self.classifier_p2_1_lr = nn.Linear(256, num_classes)
        # self.p2_1_reduce_lr = self._cbr2()
        # self.classifier_p2_2_lr = nn.Linear(256, num_classes)
        # self.p2_2_reduce_lr = self._cbr2()


        # self.classifier_g3 = nn.Linear(256, num_classes)
        # self.g3_reduce = self._cbr2()
        # self.classifier_p3_1 = nn.Linear(256, num_classes)
        # self.p3_1_reduce = self._cbr2()
        # self.classifier_p3_2 = nn.Linear(256, num_classes)
        # self.p3_2_reduce = self._cbr2()
        # self.classifier_p3_3 = nn.Linear(256, num_classes)
        # self.p3_3_reduce = self._cbr2()
###----------------------------###
        self.classifier_g1 = nn.Linear(512, num_classes)
        self.g1_reduce = self._cbr2()

        self.classifier_g2 = nn.Linear(512, num_classes)
        self.g2_reduce = self._cbr2()
        self.classifier_p2_1 = nn.Linear(512, num_classes)
        self.p2_1_reduce = self._cbr2()
        self.classifier_p2_2 = nn.Linear(512, num_classes)
        self.p2_2_reduce = self._cbr2()


        self.classifier_g2_lr = nn.Linear(512, num_classes)
        self.g2_reduce_lr = self._cbr2()
        self.classifier_p2_1_lr = nn.Linear(512, num_classes)
        self.p2_1_reduce_lr = self._cbr2()
        self.classifier_p2_2_lr = nn.Linear(512, num_classes)
        self.p2_2_reduce_lr = self._cbr2()


        self.classifier_g3 = nn.Linear(512, num_classes)
        self.g3_reduce = self._cbr2()
        self.classifier_p3_1 = nn.Linear(512, num_classes)
        self.p3_1_reduce = self._cbr2()
        self.classifier_p3_2 = nn.Linear(512, num_classes)
        self.p3_2_reduce = self._cbr2()
        self.classifier_p3_3 = nn.Linear(512, num_classes)
        self.p3_3_reduce = self._cbr2()




        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init_constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init_constant(m.weight, 1)
                init_constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init_normal(m.weight, std=0.001)
                if m.bias is not None:
                    init_constant(m.bias, 0)


    def _cbr(self, in_channel=2048, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False):
        op_s = nn.Sequential(
            nn.Linear(in_channel, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
            )
        return op_s

    def _cbr2(self, in_channel=2048, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False):
        op_s = nn.Sequential(
            nn.Conv2d(in_channel, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
            )
        return op_s

    def _make_layer(self, block, planes, blocks, stride=1,is_last=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        # if is_last:
        self.inplanes = planes * block.expansion       
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        if not is_last:
            # pdb.set_trace()
            self.inplanes = self.inplanes//2


        return nn.Sequential(*layers)

    def forward(self, x):
        batch_num = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        g1 = self.layer3_0(x)
        g1 = self.layer4_0(g1)
        g1_mp = F.avg_pool2d(g1, g1.size()[2:])
        g1_rd = self.g1_reduce(g1_mp)  # 2048->256
        g1_rd = g1_rd.view(batch_num,-1)
        g1_sm = self.classifier_g1(g1_rd)


        g2 = self.layer3_1(x)
        g2 = self.layer4_1(g2)
        g2_mp = F.avg_pool2d(g2, g2.size()[2:])
        g2_rd = self.g2_reduce(g2_mp)
        g2_rd = g2_rd.view(batch_num,-1)
        g2_sm = self.classifier_g2(g2_rd)

        h2 = g2.size(2) // 2

        p2_1 = g2[:, :, 0:h2, :]
        p2_1_mp = F.avg_pool2d(p2_1, p2_1.size()[2:])
        p2_1_rd = self.p2_1_reduce(p2_1_mp)
        p2_1_rd = p2_1_rd.view(batch_num,-1)
        p2_1_sm = self.classifier_p2_1(p2_1_rd)

        p2_2 = g2[:, :, h2:, :]
        p2_2_mp = F.avg_pool2d(p2_2, p2_2.size()[2:])
        p2_2_rd = self.p2_2_reduce(p2_2_mp)
        p2_2_rd = p2_2_rd.view(batch_num,-1)
        p2_2_sm = self.classifier_p2_2(p2_2_rd)


###########################
        g2_lr = self.layer3_lr(x)
        g2_lr = self.layer4_lr(g2_lr)
        # g2_lr = g2
        g2_mp_lr = F.avg_pool2d(g2_lr, g2_lr.size()[2:])
        g2_rd_lr = self.g2_reduce_lr(g2_mp_lr)
        g2_rd_lr = g2_rd_lr.view(batch_num,-1)
        g2_sm_lr = self.classifier_g2_lr(g2_rd_lr)

        w2_lr = g2_lr.size(3) // 2

        p2_1_lr = g2_lr[:, :, :, 0:w2_lr]
        p2_1_mp_lr = F.avg_pool2d(p2_1_lr, p2_1_lr.size()[2:])
        p2_1_rd_lr = self.p2_1_reduce_lr(p2_1_mp_lr)
        p2_1_rd_lr = p2_1_rd_lr.view(batch_num,-1)
        p2_1_sm_lr = self.classifier_p2_1_lr(p2_1_rd_lr)

        p2_2_lr = g2_lr[:, :, :, w2_lr:]
        p2_2_mp_lr = F.avg_pool2d(p2_2_lr, p2_2_lr.size()[2:])
        p2_2_rd_lr = self.p2_2_reduce_lr(p2_2_mp_lr)
        p2_2_rd_lr = p2_2_rd_lr.view(batch_num,-1)
        p2_2_sm_lr = self.classifier_p2_2_lr(p2_2_rd_lr)


#######################
        g3 = self.layer3_2(x)
        g3 = self.layer4_2(g3)
        g3_mp = F.avg_pool2d(g3, g3.size()[2:])
        g3_rd = self.g3_reduce(g3_mp)
        g3_rd = g3_rd.view(batch_num,-1)
        g3_sm = self.classifier_g3(g3_rd)

        h3 = g3.size(2) // 3

        p3_1 = g3[:, :, 0:h3, :]
        p3_1_mp = F.avg_pool2d(p3_1, p3_1.size()[2:])
        p3_1_rd = self.p3_1_reduce(p3_1_mp)
        p3_1_rd = p3_1_rd.view(batch_num,-1)
        p3_1_sm = self.classifier_p3_1(p3_1_rd)

        p3_2 = g3[:, :, h3:h3 * 2, :]
        p3_2_mp = F.avg_pool2d(p3_2, p3_2.size()[2:])
        p3_2_rd = self.p3_2_reduce(p3_2_mp)
        p3_2_rd = p3_2_rd.view(batch_num,-1)
        p3_2_sm = self.classifier_p3_2(p3_2_rd)

        p3_3 = g3[:, :, h3 * 2:h3 * 3, :]
        p3_3_mp = F.avg_pool2d(p3_3, p3_3.size()[2:])
        p3_3_rd = self.p3_3_reduce(p3_3_mp)
        p3_3_rd = p3_3_rd.view(batch_num,-1)
        p3_3_sm = self.classifier_p3_3(p3_3_rd)


        fea_trip = [g1_rd, g2_rd, g2_rd_lr, g3_rd]
        scores = [g1_sm, g2_sm, p2_1_sm, p2_2_sm, g2_sm_lr, p2_1_sm_lr, p2_2_sm_lr, g3_sm, p3_1_sm, p3_2_sm, p3_3_sm]

        if len(g1_rd.shape) > 1:
            fea_for_test = torch.cat([g1_rd, g2_rd, g3_rd, p3_2_rd],1)
        else:
            fea_for_test = torch.cat([g1_rd,g2_rd,g3_rd,p3_2_rd])
        if self.norm:
            fea_for_test = F.normalize(fea_for_test)

        return scores, fea_trip, fea_for_test

def copy_weight_for_branches(model_name, new_model, branch_num=3):
    model_weight = torch.load('./weights/resnet50-19c8e357.pth')
    model_keys = model_weight.keys()

    new_model_weight = new_model.state_dict()
    new_model_keys = new_model_weight.keys()
    # new_model_weight = copy.deepcopy(model_weight)
    handled =[]
    for block in new_model_keys:
        # print(block)
        prefix = block.split('.')[0]
        ori_prefix = prefix.split('_')[0]
        suffix = block.split('.')[1:]
        ori_key_to_join = [ori_prefix] + suffix
        ori_key = '.'.join(ori_key_to_join)
        if(ori_prefix == 'layer3') or (ori_prefix =='layer4'):
            for i in range(0,branch_num):
                if ori_key in model_keys:
                    # print(new_model_weight[block].size(),model_weight[ori_key].size())
                    new_model_weight[block] = model_weight[ori_key]
                else:
                    continue
                # pdb.set_trace()
                # handled.append(ori_key)
        elif ori_key in model_keys:
            new_model_weight[block] = model_weight[ori_key]
    save_name = './weights/'+model_name + '_mgn.tar'
    torch.save(new_model_weight,save_name)

def copy_weight_for_branches101(model_name, new_model, branch_num=3):
    model_weight = torch.load('./weights/resnet101-5d3b4d8f.pth')
    model_keys = model_weight.keys()

    new_model_weight = new_model.state_dict()
    new_model_keys = new_model_weight.keys()
    # new_model_weight = copy.deepcopy(model_weight)
    handled =[]
    for block in new_model_keys:
        # print(block)
        prefix = block.split('.')[0]
        ori_prefix = prefix.split('_')[0]
        suffix = block.split('.')[1:]
        ori_key_to_join = [ori_prefix] + suffix
        ori_key = '.'.join(ori_key_to_join)
        if(ori_prefix == 'layer3') or (ori_prefix =='layer4'):
            for i in range(0,branch_num):
                if ori_key in model_keys:
                    # print(new_model_weight[block].size(),model_weight[ori_key].size())
                    new_model_weight[block] = model_weight[ori_key]
                else:
                    continue
                # pdb.set_trace()
                # handled.append(ori_key)
        elif ori_key in model_keys:
            new_model_weight[block] = model_weight[ori_key]
    save_name = './weights/'+model_name + '_mgn.tar'
    torch.save(new_model_weight,save_name)

def copy_weight_for_branches152(model_name, new_model, branch_num=3):
    model_weight = torch.load('./weights/resnet152-b121ed2d.pth')
    model_keys = model_weight.keys()

    new_model_weight = new_model.state_dict()
    new_model_keys = new_model_weight.keys()
    # new_model_weight = copy.deepcopy(model_weight)
    handled =[]
    for block in new_model_keys:
        # print(block)
        prefix = block.split('.')[0]
        ori_prefix = prefix.split('_')[0]
        suffix = block.split('.')[1:]
        ori_key_to_join = [ori_prefix] + suffix
        ori_key = '.'.join(ori_key_to_join)
        if(ori_prefix == 'layer3') or (ori_prefix =='layer4'):
            for i in range(0,branch_num):
                if ori_key in model_keys:
                    # print(new_model_weight[block].size(),model_weight[ori_key].size())
                    new_model_weight[block] = model_weight[ori_key]
                else:
                    continue
                # pdb.set_trace()
                # handled.append(ori_key)
        elif ori_key in model_keys:
            new_model_weight[block] = model_weight[ori_key]
    save_name = './weights/'+model_name + '_mgn.tar'
    torch.save(new_model_weight,save_name)


def ResNet50_mgn_lr(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_mgn_lr(50, **kwargs)
    if pretrained:
        # pdb.set_trace()
        copy_weight_for_branches('resnet50',model)
        model.load_state_dict(torch.load('./weights/resnet50_mgn.tar'))
    return model


def ResNet101_mgn_lr(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_mgn_lr(101, **kwargs)
    if pretrained:
        copy_weight_for_branches101('resnet101',model)
        model.load_state_dict(torch.load('./weights/resnet101_mgn.tar'))
    return model


def ResNet152_mgn_lr(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_mgn_lr(152, **kwargs)
    if pretrained:
        copy_weight_for_branches152('resnet152',model)
        model.load_state_dict(torch.load('./weights/resnet152_mgn.tar'))
    return model

# def ResNet_mgn_lr_model(depth, pretrained=True, **kwargs):
#     if depth==50:
#         print('resnet50-mgn_lr')
#         return ResNet50_mgn_lr(pretrained=pretrained, **kwargs)
#     elif depth==101:
#         print('resnet101-mgn_lr')
#         return ResNet101_mgn_lr(pretrained=pretrained, **kwargs)
#     else:
#         raise KeyError("Unsupported depth:", depth)
