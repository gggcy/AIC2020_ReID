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

def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask

def extract_features(pickle_file):
    f = open(pickle_file, 'rb')
    if version_info.major == 2:
        features = pickle.load(f)
    elif version_info.major == 3:
        features = pickle.load(f,encoding='iso-8859-1')

    f.close()
    return features

def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=50, k2=15, lambda_value=0.5):
    print('k1, k2, lambda: ', k1, k2, lambda_value)
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.

    original_dist = np.concatenate(
      [np.concatenate([q_q_dist, q_g_dist], axis=1),
       np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
      axis=0)
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,:k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2.))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2.))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)
    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)


    for i in range(query_num):
        temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    return final_dist




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
    import sys
    root_path = sys.argv[1]
    k1 = int(sys.argv[2])
    k2 = int(sys.argv[3])
    lambda_value = float(sys.argv[4])
    q_pkl = sys.argv[5]
    g_pkl = sys.argv[6]
    distmat_pkl = sys.argv[7]
    print('running...')
    # root_path = "/data/home/cunyuangao/Project/baidu/Track2(ReID)/part1_model/val_pkl2/"
    # print(root_path)
    query_features = extract_features(root_path + q_pkl)
    test_features = extract_features(root_path + g_pkl)
    #query_features = extract_features(root_path + 'query_cat_6_0330_off_cen.pkl')
    #test_features = extract_features(root_path + 'after100_gallery_cat_6_0330_off_cen.pkl')
    q_f = open('name_query.txt', 'r')
    t_f = open('name_test.txt', 'r')
    query = []
    gallery = []

    for line in q_f.readlines():
        query.append(line.strip())

    for line in t_f.readlines():
        gallery.append(line.strip())

    q_f.close()
    t_f.close()
    
    q_g_dist = vehicle_pairwise_distance(query_features, test_features, query, gallery)
    q_g_dist = q_g_dist.numpy()
    q_q_dist = vehicle_pairwise_distance(query_features, query_features, query, query)
    q_q_dist = q_q_dist.numpy()
    g_g_dist = vehicle_pairwise_distance(test_features, test_features, gallery, gallery)
    g_g_dist = g_g_dist.numpy()
    
    # distmat = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=50, k2=15, lambda_value=0.5)
    # with open(root_path + 'rerank_after100_6_0330_off_cen.pkl', 'wb') as f:

    distmat = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=k1, k2=k2, lambda_value=lambda_value)
    # with open(root_path + 'rerank_{}_{}_{}.pkl'.format(k1,k2,lambda_value), 'wb') as f:
    with open(root_path + distmat_pkl, 'wb') as f:

        pickle.dump(distmat, f)
