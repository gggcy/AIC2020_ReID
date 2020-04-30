from __future__ import absolute_import
from collections import defaultdict

import os
import numpy as np
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances=1):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)


class RandomIdentityAttributeSampler(Sampler):
    def __init__(self, data_source, num_instances=1):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _, _, _, _, _, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)

class RandomIdentityAttributeSampler_s(Sampler):
    def __init__(self, data_source, num_instances=1):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _, _, _,) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)



class RandomIdentityBatchSamplerNew2(object):
    def __init__(self, data_source, batch_size, num_instances=2, num_anchor_pids=4, logs_dir='.'):
        assert batch_size % num_instances == 0

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_anchor_pids = num_anchor_pids
        self.logs_dir = logs_dir

        self.index_dic = defaultdict(list)
        self.cursors = {}
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
            if pid not in self.cursors:
                self.cursors[pid] = 0
        self.pids = list(self.index_dic.keys())

        self.num_pids = len(self.pids)
        self.num_selected_pids = self.batch_size / self.num_instances
        self.num_nearest_pids = (self.num_selected_pids / self.num_anchor_pids) - 1
        self.iter_len = self.num_pids // self.num_selected_pids
        self.epoch = 0

    def _load_pids(self, pids_file):
        with open(pids_file, 'r') as fp:
            lines = fp.readlines()
        assert len(lines) % self.iter_len == 0
        epochs = len(lines) / self.iter_len

        pids = []
        for epoch in range(epochs):
            pids_epoch = []
            for line_no in range(epoch*self.iter_len, (epoch+1)*self.iter_len):
                line = lines[line_no].strip()
                tokens = line.split()
                pids_batch = []
                for i,it in enumerate(tokens):
                    if i % self.num_instances == 0:
                        pids_batch.append(int(it))
                pids_epoch.append(pids_batch)
            pids.append(pids_epoch)

        return pids

    def __len__(self):
        return self.iter_len

    def _select_pids(self, M):
        selected_pids = []
        anchor_pids = np.random.choice(self.pids, self.num_anchor_pids, replace=False)
        for anchor_pid in anchor_pids:
            nearest_pids = np.argsort(M[anchor_pid])[::-1]
            nearest_sims = np.sort(M[anchor_pid])[::-1]
            min_nearest_num = int(self.num_pids*0.05)
            nearest_idxs = []
            if len(nearest_idxs) > min_nearest_num:
                nearest_pids = nearest_pids[nearest_idxs]
                nearest_sims = nearest_sims[nearest_idxs]
            else:
                nearest_pids = nearest_pids[1:min_nearest_num] # do not contain self
                nearest_sims = nearest_sims[1:min_nearest_num] # do not contain self
            nearest_pids = np.random.choice(nearest_pids, self.num_nearest_pids, replace=False)
            for nearest_pid in nearest_pids:
                selected_pids.append(nearest_pid)
            selected_pids.append(anchor_pid)

        selected_pids = list(set(selected_pids))
        while len(selected_pids) < self.num_selected_pids:
            num_diff = self.num_selected_pids - len(selected_pids)
            remain_pids = np.random.choice(self.pids, num_diff, replace=False)
            for pid in remain_pids:
                if pid not in selected_pids:
                    selected_pids.append(pid)

        return selected_pids


    def __iter__(self):
        simmat_path = os.path.join(self.logs_dir, 'simmat.npy')
        if os.path.exists(simmat_path):
            M = np.load(simmat_path)
        else:
            M = np.zeros((self.num_pids, self.num_pids))
        indices = torch.randperm(self.num_pids)
        for iter_idx in range(self.iter_len):
            selected_pids1 = []
            selected_pids2 = self._select_pids(M)
            selected_pids = list(selected_pids1) + list(selected_pids2)
            batch = []
            for pid in selected_pids:
                if len(self.index_dic[pid]) < self.num_instances:
                    selected_indexs = np.random.choice(self.index_dic[pid], self.num_instances, replace=True)
                else:
                    selected_indexs = np.random.choice(self.index_dic[pid], self.num_instances, replace=False)
                for selected_index in selected_indexs:
                    batch.append(selected_index)
            yield batch

        self.epoch += 1
