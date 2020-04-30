from __future__ import print_function, absolute_import
import time
from torch import nn
import torch
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .loss import  MGN_loss, XentropyLoss_SAC, TripletLoss
from .utils.meters import AverageMeter


class BaseTrainer(object):
    def __init__(self, model, criterion):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.accumulation_steps = 2
        self.ranking_loss = nn.MarginRankingLoss(margin=0.5).cuda()

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()
        lr = optimizer.param_groups[0].get('lr')

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss, prec1 = self._forward(inputs, targets)

            losses.update(loss.item(), targets.size(0))
            precisions.update(prec1, targets.size(0))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ######### accumulation lossfor enlarge batchsize for mgn #######
            # loss = loss/ self.accumulation_steps
            # # 2.2 back propagation
            # loss.backward()
            # # 3. update parameters of net
            # if((i+1)%self.accumulation_steps)==0:
            #     # optimizer the net
            #     optimizer.step()        # update parameters of net
            #     optimizer.zero_grad()   # reset gradient
            #################################################################

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


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        softmax_out = outputs[0]
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = self.criterion(softmax_out, targets)
            prec, = accuracy(softmax_out.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, MGN_loss):
            loss, prec = self.criterion(outputs, targets)
        elif isinstance(self.criterion, XentropyLoss_SAC):
            loss, prec = self.criterion(outputs, targets)

        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec

class Trainer_SAC_Triplet(BaseTrainer):
    def __init__(self, model, criterion, metric_loss_weight=0.02):
        super(Trainer_SAC_Triplet, self).__init__(model, criterion)
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

        os_g = outputs[1]
        os_g = os_g.contiguous().view(-1, os_g.size(1))
        if isinstance(self.criterion['XentropyLoss_SAC'], XentropyLoss_SAC) and \
           isinstance(self.criterion['trihard'], TripletLoss):
            loss_c, prec = self.criterion['XentropyLoss_SAC'](outputs, targets)
            loss_g = self.criterion['trihard'](os_g, targets)
            loss = loss_c + loss_g * self.metric_loss_weight
            losses.append(loss)
            loss = loss + loss_g
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec




class BaseTrainer_1(object):
    def __init__(self, model, criterion):
        super(BaseTrainer_1, self).__init__()
        self.model = model
        self.criterion = criterion

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()
        lr = optimizer.param_groups[0].get('lr')

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
            losses.update(loss.data.item(), targets.size(0))
            precisions.update(prec1, targets.size(0))
            
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

class Cross_Trihard_Trainer(BaseTrainer_1):
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
            os_c = outputs[0]
            os_g = outputs[1]
            os_c = os_c.contiguous().view(-1, os_c.size(1))
            os_g = os_g.contiguous().view(-1, os_g.size(1))
            loss_c = self.criterion['crossentropy'](os_c, targets)
            # loss_center = self.criterion['center'](os_g, targets)
            loss_g = self.criterion['trihard'](os_g, targets)
            loss = loss_c + loss_g * self.metric_loss_weight
            losses.append(loss)
            prec, = accuracy(os_c.data, targets.data)
            prec = prec[0]
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return losses, prec



