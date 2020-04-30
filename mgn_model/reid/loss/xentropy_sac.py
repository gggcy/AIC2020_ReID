from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import pdb
from ..evaluation_metrics import accuracy

class XentropyLoss_SAC(nn.Module):
    def __init__(self,alpha=1.0,gamma=1.0,theta=0.0):
        super(XentropyLoss_SAC, self).__init__()
        self.xentropy_loss = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.gamma = gamma
        self.theta = theta
        
        
    def forward(self, inputs, targets):
        inputs_fea=inputs[0]
        l2_side = inputs[2]
        l3_side = inputs[3]
        l4_side = inputs[4]

        xentropy = self.xentropy_loss(inputs_fea,targets) 
        # pdb.set_trace()
        loss42 = torch.sqrt((l4_side-l2_side).pow(2).sum())
        loss43 = torch.sqrt((l4_side-l3_side).pow(2).sum())
        loss =  self.theta*(loss42+loss43)+ self.gamma *xentropy
        accuracy_val,  = accuracy(inputs_fea.data, targets.data)
        prec = accuracy_val[0]

        return loss, prec


