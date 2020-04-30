from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable
from .evaluation_metrics import accuracy
from .utils.meters import AverageMeter
import numpy as np
import os



class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model
    
    def evaluate(self, data_loader, print_freq=1, metric=None):
        self.model.eval()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, (imgs, fnames, directs) in enumerate(data_loader):
            data_time.update(time.time() - end)
            inputs = [Variable(imgs, requires_grad=False)]
            targets = Variable(directs.cuda())
            outputs = self.model(*inputs)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
            precisions.update(prec, targets.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Test: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  'Prec {:.2%} ({:.2%})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg,
                          precisions.val, precisions.avg))

        return precisions.avg

