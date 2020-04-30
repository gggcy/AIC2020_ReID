from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
from collections import defaultdict
from sklearn.metrics import average_precision_score
import numpy as np
import torch
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

from scipy.spatial.distance import cosine
from sys import version_info


def extract_features(pickle_file):
    f = open(pickle_file, 'rb')
    if version_info.major == 2:
        features = pickle.load(f)
    elif version_info.major == 3:
        features = pickle.load(f,encoding='iso-8859-1')

    f.close()
    return features


def vehicle_pairwise_distance(query_features, test_features,  query, gallery):
    x = torch.cat([torch.from_numpy(query_features[f]).unsqueeze(0) for f in query], 0)
    y = torch.cat([torch.from_numpy(test_features[f]).unsqueeze(0) for f in gallery], 0)
    m, n = x.size(0), y.size(0)
    distance = 0
    for i in range(m):
        distance += cosine(x[i],y[i])
    distance /= m
    return distance
    #distance = 0
    #for i in range(m):
    #    distance += sum(abs(x[i]-y[i]))
    distance = cosine(x,y)
    return distance
if __name__ == '__main__':
   # import sys
   # k1 = int(sys.argv[1])
   # k2 = int(sys.argv[2])
   # lambda_value = float(sys.argv[3])
    print('running...')
    root_path = "/data/home/cunyuangao/Project/finalday/post_processing/val_pkls/"
    #root_path2 = "/data/home/cunyuangao/Project/0401/baidu/Track2(ReID)/post_processing/val_pkls/"
    print(root_path)
    query_features = extract_features(root_path + 'query_densenet_ALL520_off_cen.pkl')
    test_features = extract_features(root_path + 'query_densenet_off_cen.pkl')
    q_f = open('name_query.txt', 'r')
    t_f = open('name_query.txt', 'r')
    query = []
    gallery = []

    for line in q_f.readlines():
        query.append(line.strip())

    for line in t_f.readlines():
        gallery.append(line.strip())

    q_f.close()
    t_f.close()

    q_g_dist = vehicle_pairwise_distance(query_features, test_features, query, gallery)
    print(q_g_dist)
