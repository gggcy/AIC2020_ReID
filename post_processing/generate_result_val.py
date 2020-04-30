from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
from collections import defaultdict
from sklearn.metrics import average_precision_score
import numpy as np
import torch
import pickle
import os
from sys import version_info


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def extract_features(pickle_file):
    f = open(pickle_file, 'rb')
    if version_info.major == 2:
        features = pickle.load(f)
    elif version_info.major == 3:
        features = pickle.load(f, encoding='iso-8859-1')
    f.close()
    return features



def vehicle_pairwise_distance(query_features, test_features,  query, gallery):
    x = torch.cat([torch.from_numpy(query_features[f]).unsqueeze(0) for f in query], 0)
    y = torch.cat([torch.from_numpy(test_features[f]).unsqueeze(0) for f in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


if __name__ == '__main__':
    print('running...')
    import sys
    root_path = sys.argv[1].strip()
    distmat_file = sys.argv[2]
    #root_path = '/data/home/cunyuangao/Project/baidu/Track2(ReID)/part1_model/val_pkl2/'
    #distmat_pfile = open(root_path + 'rerank_after100_6_0330_off_cen.pkl', 'r')
    distmat_pfile = open(root_path + distmat_file, 'rb')
    if version_info.major == 3:
        distmat = pickle.load(distmat_pfile,encoding='iso-8859-1')
    elif version_info.major == 2:
        distmat = pickle.load(distmat_pfile)

    distmat_pfile.close()

    sort_distmat_index = np.argsort(distmat, axis=1)
    with open('track2.txt', 'w') as f:
        for item in sort_distmat_index:
            for i in range(99):
                f.write(str(item[i] + 1) + ' ')
            f.write(str(item[99] + 1) + '\n')

    print('Done')
