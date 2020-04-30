from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .loss import TripletLoss
from .loss import MultiAttributeLoss, TypeAttributeLoss,MultiAttributeLoss_s
from .utils.meters import AverageMeter


class BaseTrainer(object):
    def __init__(self, model, criterion):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion

    def train(self, epoch, data_loader, optimizer, base_lr, print_freq=1, warm_up=False, warm_up_ep=20):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        type_precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)
            # print(inputs)
            inputs, targets = self._parse_data(inputs)
            loss, multi_loss, multi_prec = self._forward(inputs, targets)

            losses.update(loss.data.item(), targets[0].size(0))
            precisions.update(multi_prec[0], targets[0].size(0))
            type_precisions.update(multi_prec[1], targets[1].size(0))

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
            loss.backward()
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
                      'TypePrec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              lr,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg,
                              type_precisions.val, type_precisions.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError



class Multi_Attribute_Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_loss_weight=0.02, sub_task_loss_weight=1.0):
        super(Multi_Attribute_Trainer, self).__init__(model, criterion)
        assert len(criterion.keys()) == 2
        self.metric_loss_weight = metric_loss_weight
        self.sub_task_loss_weight = sub_task_loss_weight
    
    def _parse_data(self, inputs):
        imgs, _, pids, _, color, car_type, roof, window, logo = inputs
        inputs = [Variable(imgs)]
        targets = (pids, color, car_type, roof, window, logo) 
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        if isinstance(self.criterion['MultiAttributeLoss'], MultiAttributeLoss) and \
           isinstance(self.criterion['trihard'], TripletLoss):
            if self.criterion['MultiAttributeLoss'].is_cls:
                os_cls = outputs[0]
                os_color = outputs[1]
                os_type = outputs[2]
                os_roof = outputs[3]
                os_window = outputs[4]
                os_logo = outputs[5]
                os_g = outputs[6]
                os_cls = os_cls.contiguous().view(-1, os_cls.size(1))
            else:
                os_color = outputs[0]
                os_type = outputs[1]
                os_roof = outputs[2]
                os_window = outputs[3]
                os_logo = outputs[4]
                os_g = outputs[5]
            os_color = os_color.contiguous().view(-1, os_color.size(1))
            os_type = os_type.contiguous().view(-1, os_type.size(1))
            os_roof = os_roof.contiguous().view(-1, os_roof.size(1))
            os_window = os_window.contiguous().view(-1, os_window.size(1))
            os_logo = os_logo.contiguous().view(-1, os_logo.size(1))
            os_g = os_g.contiguous().view(-1, os_g.size(1))
	    
            if self.criterion['MultiAttributeLoss'].is_cls:
                loss_input = (os_cls, os_color, os_type, os_roof, os_window, os_logo)
            else:
                loss_input = (os_color, os_type, os_roof, os_window, os_logo)

            attribute_losses = self.criterion['MultiAttributeLoss'](loss_input, targets)
            loss_g = self.criterion['trihard'](os_g, targets[0].cuda())
            loss = attribute_losses[0]
            for i in range(1, len(attribute_losses)):
                loss += self.sub_task_loss_weight * attribute_losses[i]
            loss += loss_g * self.metric_loss_weight

            multi_prec = []
            if self.criterion['MultiAttributeLoss'].is_cls:
                prec, = accuracy(os_cls.data, targets[0].cuda().data)
                prec = prec[0]
                multi_prec.append(prec)
            color_prec, = accuracy(os_color.data, targets[1].cuda().data)
            type_prec, = accuracy(os_type.data, targets[2].cuda().data)
            color_prec = color_prec[0]
            type_prec = type_prec[0]
            multi_prec.append(color_prec)
            multi_prec.append(type_prec)
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, attribute_losses, multi_prec


class Type_Attribute_Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_loss_weight=0.02, sub_task_loss_weight=1.0):
        super(Type_Attribute_Trainer, self).__init__(model, criterion)
        assert len(criterion.keys()) == 2
        self.metric_loss_weight = metric_loss_weight
        self.sub_task_loss_weight = sub_task_loss_weight

    def _parse_data(self, inputs):
        imgs, _, pids, _, _, car_type, _, _, _ = inputs
        inputs = [Variable(imgs)]
        targets = (pids, car_type)
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        if isinstance(self.criterion['TypeAttributeLoss'], TypeAttributeLoss) and \
           isinstance(self.criterion['trihard'], TripletLoss):
            if self.criterion['TypeAttributeLoss'].is_cls:
                os_cls = outputs[0]
                os_type = outputs[1]
                os_g = outputs[2]
                os_cls = os_cls.contiguous().view(-1, os_cls.size(1))
            else:
                os_type = outputs[0]
                os_g = outputs[1]
            os_type = os_type.contiguous().view(-1, os_type.size(1))
            os_g = os_g.contiguous().view(-1, os_g.size(1))

            if self.criterion['TypeAttributeLoss'].is_cls:
                loss_input = (os_cls, os_type)
            else:
                loss_input = (os_type)

            attribute_losses = self.criterion['TypeAttributeLoss'](loss_input, targets)
            loss_g = self.criterion['trihard'](os_g, targets[0].cuda())
            loss = attribute_losses[0]
            for i in range(1, len(attribute_losses)):
                loss += self.sub_task_loss_weight * attribute_losses[i]
            loss += loss_g * self.metric_loss_weight

            multi_prec = []
            if self.criterion['TypeAttributeLoss'].is_cls:
                prec, = accuracy(os_cls.data, targets[0].cuda().data)
                prec = prec[0]
                multi_prec.append(prec)
            type_prec, = accuracy(os_type.data, targets[1].cuda().data)
            type_prec = type_prec[0]
            multi_prec.append(type_prec)
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, attribute_losses, multi_prec


class Multi_Attribute_Trainer_s(BaseTrainer):
    def __init__(self, model, criterion, metric_loss_weight=0.02, sub_task_loss_weight=1.0):
        super(Multi_Attribute_Trainer_s, self).__init__(model, criterion)
        assert len(criterion.keys()) == 2
        self.metric_loss_weight = metric_loss_weight
        self.sub_task_loss_weight = sub_task_loss_weight
    
    def _parse_data(self, inputs):
        # print(len(inputs))
        imgs, _, pids, _, color, car_type = inputs
        inputs = [Variable(imgs)]
        targets = (pids, color, car_type) 
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        if isinstance(self.criterion['MultiAttributeLoss'], MultiAttributeLoss_s) and \
           isinstance(self.criterion['trihard'], TripletLoss):
            if self.criterion['MultiAttributeLoss'].is_cls:
                os_cls = outputs[0]
                os_color = outputs[1]
                os_type = outputs[2]
                os_g = outputs[3]
                os_cls = os_cls.contiguous().view(-1, os_cls.size(1))
            else:
                os_color = outputs[0]
                os_type = outputs[1]
                os_g = outputs[2]
            os_color = os_color.contiguous().view(-1, os_color.size(1))
            os_type = os_type.contiguous().view(-1, os_type.size(1))
            os_g = os_g.contiguous().view(-1, os_g.size(1))

            if self.criterion['MultiAttributeLoss'].is_cls:
                loss_input = (os_cls, os_color, os_type)
            else:
                loss_input = (os_color, os_type)

            attribute_losses = self.criterion['MultiAttributeLoss'](loss_input, targets)
            loss_g = self.criterion['trihard'](os_g, targets[0].cuda())
            loss = attribute_losses[0]
            for i in range(1, len(attribute_losses)):
                loss += self.sub_task_loss_weight * attribute_losses[i]
            loss += loss_g * self.metric_loss_weight
	    
            multi_prec = []
            if self.criterion['MultiAttributeLoss'].is_cls:
                prec, = accuracy(os_cls.data, targets[0].cuda().data)
                prec = prec[0]
                multi_prec.append(prec)
            color_prec, = accuracy(os_color.data, targets[1].cuda().data)
            type_prec, = accuracy(os_type.data, targets[2].cuda().data)
            color_prec = color_prec[0]
            type_prec = type_prec[0]
            multi_prec.append(color_prec)
            multi_prec.append(type_prec)
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, attribute_losses, multi_prec