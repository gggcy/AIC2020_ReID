from __future__ import absolute_import

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


class DualMatch(nn.Module):
    def __init__(self, gamma=0.1):
        super(DualMatch, self).__init__()
        self.gamma = gamma

    def _forward(self, x):
        n, c, l = x.size() 
        x_flatten = x.permute(2, 0, 1).contiguous()
        x_flatten_trans = x_flatten.transpose(1, 2).contiguous()
        sim_mat = x_flatten.bmm(x_flatten_trans)
        sim_mat = sim_mat.permute(1, 2, 0).contiguous()
        sim_mat = sim_mat.mul(self.gamma)

        return sim_mat

    def forward(self, x):
        sim_mat = self._forward(x)
        return sim_mat


class DualMatchTest(nn.Module):
    def __init__(self):
        super(DualMatchTest, self).__init__()

    def _dualmatch(self, q, g):
        qg_sim = q.sub(g).pow(2).sum(1, keepdim=True)
        qg_sim = qg_sim.view(qg_sim.size(0), -1)
        sim_output = qg_sim.mean(1, keepdim=True)
        return sim_output

    def forward(self, q, g):
        sim_output = self._dualmatch(q, g)
        return sim_output


#class DualMatchTest(nn.Module):
#    def __init__(self):
#        super(DualMatchTest, self).__init__()
#
#    def _dualmatch(self, q, g):
#        n, c, l = q.size()
#        q_expand = q.view(n, c, l, 1).expand(n, c, l, l).contiguous()
#        g_expand = g.view(n, c, 1, l).expand(n, c, l, l).contiguous()
#        qg_sim = q_expand.sub(g_expand).pow(2).sum(1, keepdim=True)
#        qg_sim = qg_sim.view(n, l, l)
#        qg_sim_weight = F.softmax(qg_sim.view(n*l, l).mul(-10))
#        qg_sim_weight = qg_sim_weight.view(n, l, l)
#        qg_sim = qg_sim.mul(qg_sim_weight).sum(2, keepdim=True)
#        qg_sim = qg_sim.view(n, -1)
#        sim_output = qg_sim.mean(1, keepdim=True)
#        return sim_output
#
#    def forward(self, q, g):
#        sim_output = self._dualmatch(q, g)
#        return sim_output


class MultiPartNPairLoss(nn.Module):
    def __init__(self):
        super(MultiPartNPairLoss, self).__init__()

    def _forward(self, Ws, targets):
        labels = targets.data.cpu().numpy()
        sim_mat, score_mat = tuple(Ws.split(1, 0))
        sim_mat = sim_mat.view(sim_mat.size(1), sim_mat.size(2), sim_mat.size(3))
        score_mat = score_mat.view(score_mat.size(1), score_mat.size(2), score_mat.size(3))
        Ws = sim_mat
#        Wg = sim_mat.mul(score_mat).sum(2, keepdim=True).view(Ws.size(0), Ws.size(1))
        Wg = Ws.sum(2, keepdim=True).view(Ws.size(0), Ws.size(1))

        part_losses = []
        Ws = Ws.split(1, 2)
        for W in Ws:
            part_loss = self._calc_loss(W.contiguous().view(W.size(0), W.size(1)), Wg, labels)
            part_losses.append(part_loss)

        return part_losses

    def forward(self, Ws, targets):
        losses = self._forward(Ws, targets)
        return losses
