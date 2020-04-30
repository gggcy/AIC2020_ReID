from __future__ import print_function, absolute_import
from __future__ import division
import time
from torch.nn import DataParallel

import torch
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .loss import TripletLoss,CrossEntropyLabelSmooth, CenterLoss
from .utils.meters import AverageMeter
from .loss import AngularPenaltySMLoss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

class BaseTrainer(object):
    def __init__(self, model, criterion):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion

    def train(self, epoch, data_loader, optimizer, base_lr, print_freq=1, warm_up=False, warm_up_ep=20, accumu_step = 1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            my_losses, prec1 = self._forward(inputs, targets)
            loss = my_losses[0]
            #for loss_i in range(1, len(my_losses)):
            #    loss += my_losses[i]
            #loss /= len(my_losses)

            losses.update(loss.data.item(), targets.size(0))
            precisions.update(prec1, targets.size(0))
            
            if warm_up:
                warm_iters = float(len(data_loader) * warm_up_ep)
                if epoch < warm_up_ep:
                    lr = (base_lr / warm_iters) + (epoch*len(data_loader) + (i+1))*(base_lr / warm_iters)
                    for g in optimizer.param_groups:
                        g['lr'] = lr * g.get('lr_mult', 1)
                else:
                    lr = base_lr
                    for g in optimizer.param_groups:
                        g['lr'] = lr * g.get('lr_mult', 1)
            else:
                lr =  base_lr
   
            optimizer.zero_grad()
            #loss.backward()
            torch.autograd.backward(my_losses, [torch.ones(1).cuda() for loss_i in range(len(my_losses))])
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Base_Lr: {:0.5f} \t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              lr,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError



class Cross_Trihard_Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_loss_weight=0.02):
        super(Cross_Trihard_Trainer, self).__init__(model, criterion)
        assert len(criterion.keys()) == 2
        self.metric_loss_weight = metric_loss_weight
    
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        losses = []
        if isinstance(self.criterion['crossentropy'], torch.nn.CrossEntropyLoss) and \
           isinstance(self.criterion['trihard'], TripletLoss):
        # if isinstance(self.criterion['crossentropy'], CrossEntropyLabelSmooth) and \
        #    isinstance(self.criterion['trihard'], TripletLoss):
        # if isinstance(self.criterion['crossentropy'], CrossEntropyLabelSmooth) and \
        #    isinstance(self.criterion['trihard'], TripletLoss) and \
        #    isinstance(self.criterion['center'], CenterLoss):

            os_c = outputs[0]
            os_g = outputs[1]
            os_c = os_c.contiguous().view(-1, os_c.size(1))
            os_g = os_g.contiguous().view(-1, os_g.size(1))
            loss_c = self.criterion['crossentropy'](os_c, targets)
            loss_g = self.criterion['trihard'](os_g, targets)
            loss = loss_c + loss_g * self.metric_loss_weight 
            losses.append(loss)
            prec, = accuracy(os_c.data, targets.data)
            prec = prec[0]
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return losses, prec


class Cross_Trihard_Center_Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_loss_weight=0.02):
        super(Cross_Trihard_Center_Trainer, self).__init__(model, criterion)
        assert len(criterion.keys()) == 2
        self.metric_loss_weight = metric_loss_weight

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        losses = []
        # if isinstance(self.criterion['crossentropy'], torch.nn.CrossEntropyLoss) and \
        #    isinstance(self.criterion['trihard'], TripletLoss):
        # if isinstance(self.criterion['crossentropy'], CrossEntropyLabelSmooth) and \
        #    isinstance(self.criterion['trihard'], TripletLoss):
        if isinstance(self.criterion['crossentropy'], CrossEntropyLabelSmooth) and \
                isinstance(self.criterion['trihard'], TripletLoss) and \
                isinstance(self.criterion['center'], CenterLoss):

            os_c = outputs[0]
            os_g = outputs[1]
            os_c = os_c.contiguous().view(-1, os_c.size(1))
            os_g = os_g.contiguous().view(-1, os_g.size(1))
            loss_c = self.criterion['crossentropy'](os_c, targets)
            loss_g = self.criterion['trihard'](os_g, targets)
            loss_center  = self.criterion['center'](os_g, targets)

            loss = loss_c + loss_g * self.metric_loss_weight + loss_center * 0.02
            losses.append(loss)
            prec, = accuracy(os_c.data, targets.data)
            prec = prec[0]
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return losses, prec



class Arc_Trihard_Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_fc ,metric_loss_weight=0.02):
        super(Arc_Trihard_Trainer, self).__init__(model, criterion)
        assert len(criterion.keys()) == 2
        self.metric_loss_weight = metric_loss_weight
        self.metric_fc = metric_fc

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        losses = []
        # if isinstance(self.criterion['crossentropy'],torch.nn.CrossEntropyLoss) and \
        #    isinstance(self.criterion['trihard'], TripletLoss):
        if isinstance(self.criterion['crossentropy'], CrossEntropyLabelSmooth) and \
           isinstance(self.criterion['trihard'], TripletLoss):
            os_c = outputs[0]
            os_g = outputs[1]
            os_c = os_c.contiguous().view(-1, os_c.size(1))
            os_g = os_g.contiguous().view(-1, os_g.size(1))
            output = self.metric_fc(os_c,targets)
            loss_c = self.criterion['crossentropy'](output, targets)
            loss_g = self.criterion['trihard'](os_g, targets)
            loss = loss_c + loss_g * self.metric_loss_weight
            losses.append(loss)
            prec, = accuracy(output.data, targets.data)
            prec = prec[0]
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return losses, prec

