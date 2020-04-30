from __future__ import absolute_import

import math
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

class MultiAttributeLoss(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, use_gpu=True, is_cls=True):
        super(MultiAttributeLoss, self).__init__()
        self.use_gpu = use_gpu
        self.is_cls = is_cls
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        if self.is_cls:
            pid, color, car_type, roof, window, logo = inputs
            pid_tar, color_tar, type_tar, roof_tar, window_tar, logo_tar = targets
            cls_log_probs = self.logsoftmax(pid)
            pid_tar = Variable(torch.zeros(cls_log_probs.size()).scatter_(1, pid_tar.unsqueeze(1).data.cpu(), 1))
        else:
            color, car_type, roof, window, logo = inputs
            _, color_tar, type_tar, roof_tar, window_tar, logo_tar = targets
        
        color_log_probs = self.logsoftmax(color)
        type_log_probs = self.logsoftmax(car_type)
        roof_log_probs = self.logsoftmax(roof)
        window_log_probs = self.logsoftmax(window)
        logo_log_probs = self.logsoftmax(logo)
        
        color_tar = Variable(torch.zeros(color_log_probs.size()).scatter_(1, color_tar.unsqueeze(1).data.cpu(), 1))
        type_tar = Variable(torch.zeros(type_log_probs.size()).scatter_(1, type_tar.unsqueeze(1).data.cpu(), 1))
        
        real_roof_tar = Variable(torch.zeros(roof_log_probs.size()))
        for i in range(roof_tar.size()[0]):
            if roof_tar[i] == 0:
                real_roof_tar[i][0] = 1
            elif roof_tar[i] == 1:
                real_roof_tar[i][1] =1
            else:
                real_roof_tar[i][0], real_roof_tar[i][1] = 0.5, 0.5
        
        real_window_tar = Variable(torch.zeros(window_log_probs.size()))
        for i in range(window_log_probs.size()[0]):
            if window_tar[i] == 0:
                real_window_tar[i][0] = 1
            elif window_tar[i] == 1:
                real_window_tar[i][1] = 1
            else:
                real_window_tar[i][0], real_window_tar[i][1] = 0.5, 0.5
        
        real_logo_tar = Variable(torch.zeros(logo_log_probs.size()))
        for i in range(logo_log_probs.size()[0]):
            if 'x' in logo_tar[i]:
                logo_index = int(logo_tar[i][:-1])
                real_logo_tar[i][logo_index] = 0.5
            else:
                logo_index = int(logo_tar[i])
                real_logo_tar[i][logo_index] = 1

        if self.use_gpu:
            if self.is_cls:
                pid_tar = pid_tar.cuda()
                color_tar = color_tar.cuda()
                type_tar = type_tar.cuda()
                real_roof_tar = real_roof_tar.cuda()
                real_window_tar = real_window_tar.cuda()
                real_logo_tar = real_logo_tar.cuda()
        
        if self.is_cls:
            cls_loss = (- pid_tar * cls_log_probs).mean(0).sum()
            color_loss = (- color_tar * color_log_probs).mean(0).sum()
            type_loss = (- type_tar * type_log_probs).mean(0).sum()
            roof_loss = (- real_roof_tar * roof_log_probs).mean(0).sum()
            window_loss = (- real_window_tar * window_log_probs).mean(0).sum()
            logo_loss = (- real_logo_tar * logo_log_probs).mean(0).sum()
        
        if self.is_cls:
            losses = [cls_loss, color_loss, type_loss, roof_loss, window_loss, logo_loss]
        else:
            losses = [color_loss, type_loss, roof_loss, window_loss, logo_loss]

        return losses



class TypeAttributeLoss(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, use_gpu=True, is_cls=True):
        super(TypeAttributeLoss, self).__init__()
        self.use_gpu = use_gpu
        self.is_cls = is_cls
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        if self.is_cls:
            pid, car_type = inputs
            pid_tar, type_tar = targets
            cls_log_probs = self.logsoftmax(pid)
            pid_tar = Variable(torch.zeros(cls_log_probs.size()).scatter_(1, pid_tar.unsqueeze(1).data.cpu(), 1))
        else:
            car_type = inputs
            _, type_tar = targets
        
        type_log_probs = self.logsoftmax(car_type)
        type_tar = Variable(torch.zeros(type_log_probs.size()).scatter_(1, type_tar.unsqueeze(1).data.cpu(), 1))

        if self.use_gpu:
            if self.is_cls:
                pid_tar = pid_tar.cuda()
                type_tar = type_tar.cuda()
        
        if self.is_cls:
            cls_loss = (- pid_tar * cls_log_probs).mean(0).sum()
            type_loss = (- type_tar * type_log_probs).mean(0).sum()
        
        if self.is_cls:
            losses = [cls_loss, type_loss]
        else:
            losses = [type_loss]

        return losses


class MultiAttributeLoss_s(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, use_gpu=True, is_cls=True):
        super(MultiAttributeLoss_s, self).__init__()
        self.use_gpu = use_gpu
        self.is_cls = is_cls
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        if self.is_cls:
            pid, color, car_type = inputs
            pid_tar, color_tar, type_tar = targets
            cls_log_probs = self.logsoftmax(pid)
            pid_tar = Variable(torch.zeros(cls_log_probs.size()).scatter_(1, pid_tar.unsqueeze(1).data.cpu(), 1))
        else:
            color, car_type = inputs
            _, color_tar, type_tar = targets
        
        color_log_probs = self.logsoftmax(color)
        type_log_probs = self.logsoftmax(car_type)
        # roof_log_probs = self.logsoftmax(roof)
        # window_log_probs = self.logsoftmax(window)
        # logo_log_probs = self.logsoftmax(logo)
        
        color_tar = Variable(torch.zeros(color_log_probs.size()).scatter_(1, color_tar.unsqueeze(1).data.cpu(), 1))
        type_tar = Variable(torch.zeros(type_log_probs.size()).scatter_(1, type_tar.unsqueeze(1).data.cpu(), 1))
        
        # real_roof_tar = Variable(torch.zeros(roof_log_probs.size()))
        # for i in range(roof_tar.size()[0]):
        #     if roof_tar[i] == 0:
        #         real_roof_tar[i][0] = 1
        #     elif roof_tar[i] == 1:
        #         real_roof_tar[i][1] =1
        #     else:
        #         real_roof_tar[i][0], real_roof_tar[i][1] = 0.5, 0.5
        
        # real_window_tar = Variable(torch.zeros(window_log_probs.size()))
        # for i in range(window_log_probs.size()[0]):
        #     if window_tar[i] == 0:
        #         real_window_tar[i][0] = 1
        #     elif window_tar[i] == 1:
        #         real_window_tar[i][1] = 1
        #     else:
        #         real_window_tar[i][0], real_window_tar[i][1] = 0.5, 0.5
        
        # real_logo_tar = Variable(torch.zeros(logo_log_probs.size()))
        # for i in range(logo_log_probs.size()[0]):
	    # if 'x' in logo_tar[i]:
	    #     logo_index = int(logo_tar[i][:-1])
		# real_logo_tar[i][logo_index] = 0.5
	    # else:
		# logo_index = int(logo_tar[i])
		# real_logo_tar[i][logo_index] = 1

        if self.use_gpu:
            if self.is_cls:
                pid_tar = pid_tar.cuda()
                color_tar = color_tar.cuda()
                type_tar = type_tar.cuda()
                # real_roof_tar = real_roof_tar.cuda()
                # real_window_tar = real_window_tar.cuda()
                # real_logo_tar = real_logo_tar.cuda()
        
        if self.is_cls:
            cls_loss = (- pid_tar * cls_log_probs).mean(0).sum()
            color_loss = (- color_tar * color_log_probs).mean(0).sum()
            type_loss = (- type_tar * type_log_probs).mean(0).sum()
            # roof_loss = (- real_roof_tar * roof_log_probs).mean(0).sum()
            # window_loss = (- real_window_tar * window_log_probs).mean(0).sum()
            # logo_loss = (- real_logo_tar * logo_log_probs).mean(0).sum()
        
        if self.is_cls:
            losses = [cls_loss, color_loss, type_loss]
        else:
            losses = [color_loss, type_loss]

        return losses