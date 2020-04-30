from __future__ import absolute_import

import math
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


class NPairLoss(nn.Module):
    def __init__(self, gamma = 0.001, use_ohsm = False, ratio = 0.4):
        super(NPairLoss, self).__init__()
        self.gamma = gamma
        self.use_ohsm = use_ohsm
        self.ratio = ratio

    def _calc_single_loss(self, qry_idx, pos_idx, batch_size, Wq):
        mask_np = [ int(1) for idx in range(batch_size) ]
        mask_np[qry_idx] = int(0)
        mask_np[pos_idx] = int(0)
        mask = Variable(torch.ByteTensor(mask_np).cuda())
        Wq = Wq.view(-1)
        Wq_neg = Wq.masked_select(mask)
        Wq_pos = Wq.select(0, pos_idx)
        if self.use_ohsm:
            dis_diff = Wq_neg.sub_(Wq_pos.expand_as(Wq_neg))
            _, indices = dis_diff.sort(descending=True)
            indices = indices[0:int(len(indices)*self.ratio)].data.cpu().numpy()
            hsm_mask_np = [ int(0) for idx in range(len(dis_diff)) ] 
            for indice in indices:
                hsm_mask_np[indice] = 1
            hsm_mask = Variable(torch.ByteTensor(hsm_mask_np).cuda())
            dis_diff_hsm = dis_diff.masked_select(hsm_mask)
            exp_dis_diff = dis_diff_hsm.exp()
        else:
            exp_dis_diff = Wq_neg.sub_(Wq_pos.expand_as(Wq_neg)).exp()
        between_phi = exp_dis_diff.sum()
        single_loss = between_phi.log1p()
        return single_loss

    def forward(self, inputs, targets):
        assert inputs.dim() == 2
        assert targets.dim() == 1
        assert inputs.size(0) == targets.size(0)
        labels = targets.data.cpu().numpy()

        batch_size = inputs.size(0)
        q = inputs
        qt = inputs.transpose(0, 1).contiguous()
        W = torch.mm(q, qt).mul(self.gamma)
        Wq_arr = W.split(1, 0)

        single_losses = []
        for qry_idx in range(batch_size):
            # find positive
            pos_idx = -1 
            for ref_idx in range(batch_size):
                if labels[qry_idx] == labels[ref_idx] and qry_idx != ref_idx:
                    pos_idx = ref_idx
                    break
            assert pos_idx >= 0

            # calc single loss
            single_loss = self._calc_single_loss(qry_idx, pos_idx, batch_size, Wq_arr[qry_idx])
            single_losses.append(single_loss)

        loss = single_losses[0]
        for qry_idx in range(1, batch_size):
            loss += single_losses[qry_idx]
        loss /= batch_size

        return loss


class NPairAngularLoss(nn.Module):
    def __init__(self, gamma = 0.001, angle = 45., w = 1.):
        super(NPairAngularLoss, self).__init__()
        self.gamma = gamma
        pi = math.acos(0)*2.
        tan_angle = math.tan(angle*pi / 180.)
        self.factor_const = tan_angle * tan_angle;
        self.factor_const_div_2 = self.factor_const * 0.5;
        self.w = w

    def _calc_single_loss(self, qry_idx, pos_idx, batch_size, Wq):
        mask_np = [ int(1) for idx in range(batch_size) ]
        mask_np[qry_idx] = int(0)
        mask_np[pos_idx] = int(0)
        mask = Variable(torch.ByteTensor(mask_np).cuda())
        Wq = Wq.view(-1)
        Wq_neg = Wq.masked_select(mask)
        Wq_pos = Wq.select(0, pos_idx)
        exp_dis_diff = Wq_neg.sub_(Wq_pos.expand_as(Wq_neg)).exp()
        between_phi = exp_dis_diff.sum()
        single_loss = between_phi.log1p()
        return single_loss

    def _calc_single_angular_loss(self, qry_idx, pos_idx, batch_size, Wq, Wp):
        mask_np = [ int(1) for idx in range(batch_size) ]
        mask_np[qry_idx] = int(0)
        mask_np[pos_idx] = int(0)
        mask = Variable(torch.ByteTensor(mask_np).cuda())
        Wq = Wq.view(-1)
        Wq_neg = Wq.masked_select(mask)
        Wq_pos = Wq.select(0, pos_idx)
        Wp = Wp.view(-1)
        Wp_neg = Wp.masked_select(mask) 
        exp_dis_diff = Wq_neg.add_(Wp_neg).mul(self.factor_const).sub_(Wq_pos.mul(self.factor_const_div_2+0.5).expand_as(Wq_neg)).exp()
        between_phi = exp_dis_diff.sum()
        single_loss = between_phi.log1p()
        return single_loss

    def forward(self, inputs, targets):
        assert inputs.dim() == 2
        assert targets.dim() == 1
        assert inputs.size(0) == targets.size(0)
        labels = targets.data.cpu().numpy()

        batch_size = inputs.size(0)
        q = inputs
        qt = inputs.transpose(0, 1).contiguous()
        W = torch.mm(q, qt).mul(self.gamma)
        Wq_arr = W.split(1, 0)

        single_losses = []
        single_angular_losses = []
        for qry_idx in range(batch_size):
            # find positive
            pos_idx = -1 
            for ref_idx in range(batch_size):
                if labels[qry_idx] == labels[ref_idx] and qry_idx != ref_idx:
                    pos_idx = ref_idx
                    break
            assert pos_idx >= 0

            # calc single loss
            single_loss = self._calc_single_loss(qry_idx, pos_idx, batch_size, Wq_arr[qry_idx])
            single_angular_loss = self._calc_single_angular_loss(qry_idx, pos_idx, batch_size, Wq_arr[qry_idx], Wq_arr[pos_idx])
            single_losses.append(single_loss)
            single_angular_losses.append(single_angular_loss)

        loss = single_losses[0]
        for qry_idx in range(1, batch_size):
            loss += single_losses[qry_idx]
        loss /= batch_size

        angular_loss = single_angular_losses[0]
        for qry_idx in range(1, batch_size):
            angular_loss += single_angular_losses[qry_idx]
        angular_loss /= batch_size

        loss = loss + self.w * angular_loss
        return loss

class BatchHardLoss(nn.Module):
    def __init__(self, gamma = 0.001, m = 0.0):
        super(BatchHardLoss, self).__init__()
        self.gamma = gamma

        self.softmax = nn.Softmax()
        self.M_std = 0.1
        self.rw_flag = False

        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def set_gamma(self, gamma):
        self.gamma = gamma

    def _calc_single_loss(self, qry_idx, pos_flags, neg_flags, batch_size, Wq):
        pos_mask = Variable(torch.ByteTensor(pos_flags).cuda(), requires_grad=False)
        neg_mask = Variable(torch.ByteTensor(neg_flags).cuda(), requires_grad=False)
        Wq = Wq.view(-1).mul(self.gamma).clamp(min=-16, max=16)
        Wq_pos = Wq.masked_select(pos_mask).mul(-1).exp()
#        cosine = Wq.masked_select(pos_mask)
#        sine = cosine.pow(2).sub(1).mul(-1).clamp(min=1e-16, max=1).sqrt()
#        phi = cosine * self.cos_m - sine * self.sin_m
#        Wq_pos = cosine.clone()
#        phi_mask = cosine.sub(self.th).gt(0)
#        if phi_mask.long().sum().data.cpu().numpy()[0] > 0:
#            Wq_pos.masked_scatter_(phi_mask, phi)
#        Wq_pos = Wq_pos.mul(-1).mul(self.gamma).exp()
        Wq_neg = Wq.masked_select(neg_mask).exp()
        between_phi = Wq_pos.sum() * Wq_neg.sum()
        single_loss = between_phi.log()
        return single_loss

    def _calc_M(self, W):
        n, c = W.size()
        W = W.div(W.norm(2, 1, keepdim=True).clamp(min=1e-16).expand_as(W))
        W_trans = W.transpose(0, 1).contiguous()
        M = W.mm(W_trans)
        return M

#    def _calc_random_walk_M(self, M, alpha):
#        M = M.sub(M.mean()).div(M.std()+1e-10)
#        M = M.mul(self.M_std)
#        one_diag = Variable(torch.eye(M.size(0)), requires_grad=False).cuda()
#        inf_diag = Variable(torch.diag(torch.Tensor([-float('Inf')]).expand(M.size(0))), requires_grad=False).cuda()
#        M_pre = M + inf_diag
#        M = self.softmax(M_pre)
#        M = (1 - alpha) * torch.inverse(one_diag - alpha * M)
#        M = M.transpose(0, 1).contiguous()
#        return M

    def _calc_random_walk_M(self, M, k, alpha):
        M = M.sub(M.mean()).div(M.std()+1e-10)
        M = M.mul(self.M_std)
        sorted, indices = M.topk(k, dim=1)
        indices = indices.data.cpu().numpy()
        mask = np.zeros((M.size(0), M.size(1)), dtype='int')
        for i in range(len(indices)):
            for indice in indices[i]:
                mask[i][indice] = 1
        mask = Variable(torch.ByteTensor(mask), requires_grad=False).cuda()
        M_inf = Variable(torch.FloatTensor(M.size()), requires_grad=False).cuda()
        M_inf.data.fill_(-float('Inf'))
        M_inf.masked_scatter_(mask, M)
        M = M_inf
        one_diag = Variable(torch.eye(M.size(0)), requires_grad=False).cuda()
        inf_diag = Variable(torch.diag(torch.Tensor([-float('Inf')]).expand(M.size(0))), requires_grad=False).cuda()
        M_pre = M + inf_diag
        M = self.softmax(M_pre)
        M = (1 - alpha) * torch.inverse(one_diag - alpha * M)
        M = M.transpose(0, 1).contiguous()
        return M

    def forward(self, inputs, targets):
        assert inputs.dim() == 2
        assert targets.dim() == 1
        assert inputs.size(0) == targets.size(0)
        labels = targets.data.cpu().numpy()

        batch_size = inputs.size(0)
        q = inputs
#        q = q.div(q.norm(2, 1, keepdim=True).clamp(min=1e-16).expand_as(q))
        qt = q.transpose(0, 1).contiguous()
        W = torch.mm(q, qt)
#        W = torch.mm(q, qt).mul(self.gamma)
        if self.rw_flag:
            M = self._calc_M(q)
            M = self._calc_random_walk_M(W, 4, 0.95)
            M = M.detach()
            W = W.mm(M).mul(1)
        Wq_arr = W.split(1, 0)

        single_losses = []
        for qry_idx in range(batch_size):
            pos_flags = [ 0 for ref_idx in range(batch_size) ]
            neg_flags = [ 0 for ref_idx in range(batch_size) ]
            for ref_idx in range(batch_size):
                if labels[qry_idx] == labels[ref_idx] and qry_idx != ref_idx:
                    pos_flags[ref_idx] = 1
                if labels[qry_idx] != labels[ref_idx]:
                    neg_flags[ref_idx] = 1

            # calc single loss
            single_loss = self._calc_single_loss(qry_idx, pos_flags, neg_flags, batch_size, Wq_arr[qry_idx])
            single_losses.append(single_loss)

        loss = single_losses[0]
        for qry_idx in range(1, batch_size):
            loss += single_losses[qry_idx]
        loss /= batch_size

        return loss
