import numpy as np
import pickle
from collections import OrderedDict
import sys
import torch
import pickle
from sys import version_info

root_path = sys.argv[1].strip()
gallery_picklefile = sys.argv[2]
query_picklefile = sys.argv[3]
feature_dimension = int(sys.argv[4])
output_file = sys.argv[5]

# --------------------------- query expansion -----------------#
# load feature
# gf = np.load('./data/gf_multi.npy')
# qf = np.load('./data/qf_ori.npy')
# root_path = '/data/home/cunyuangao/Project/baidu/Track2(ReID)/part1_model/val_pkl/'
# picklefile = open(root_path + 'after_gallery_cat_2_last.pkl','rb')
picklefile = open(root_path + gallery_picklefile, 'rb')
if version_info.major == 3:
    gf_pkl = pickle.load(picklefile,encoding='iso-8859-1')
elif version_info.major == 2:
    gf_pkl = pickle.load(picklefile)
gf_npy_int = np.zeros([18290, feature_dimension])

# gf_npy_int = np.zeros([18290,4096])
gf = gf_npy_int.astype(np.float32)
for key, value in gf_pkl.items():
    # print(key,value)
    key_ind = int(key.split('.')[0]) - 1
    gf[key_ind] = value

picklefile = open(root_path + query_picklefile, 'rb')

# picklefile=open(root_path + 'query_cat_2_last.pkl','rb')
if version_info.major == 2:
    qf_pkl = pickle.load(picklefile,encoding='iso-8859-1')
elif version_info.major == 3:
    qf_pkl = pickle.load(picklefile)
qf_npy_int = np.zeros([1052, feature_dimension])
# qf_npy_int = np.zeros([1052,4096])
qf = qf_npy_int.astype(np.float32)
for key, value in qf_pkl.items():
    # print(key,value)
    key_ind = int(key.split('.')[0]) - 1
    qf[key_ind] = value


## load indice
# qf_new = qf
# f = open('query_new.txt', 'r')
# for k, line in enumerate(f):
#    temp = int(line[0:6]) - 1
#    qf[k] = qf_new[temp]
# f.close()

# feature norm
# q_n = np.linalg.norm(qf, axis=1, keepdims=True)
# qf = qf / q_n

#####
# dist = np.dot(qf, np.transpose(gf))
# dist = 2. - 2 * dist  # change the cosine similarity metric to euclidean similarity metric
####

def extract_features(pickle_file):
    f = open(pickle_file, 'rb')
    if version_info.major == 3:
        features = pickle.load(f,encoding='iso-8859-1')
    elif version_info.major == 3:
        features = pickle.load(f)

    f.close()
    return features


def vehicle_pairwise_distance(query_features, test_features, query, gallery):
    x = torch.cat([torch.from_numpy(query_features[f]).unsqueeze(0) for f in query], 0)
    y = torch.cat([torch.from_numpy(test_features[f]).unsqueeze(0) for f in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


q_f = open('name_query.txt', 'r')
t_f = open('name_test.txt', 'r')
query_feature = extract_features(root_path + query_picklefile)
test_feature = extract_features(root_path + gallery_picklefile)
query = []
gallery = []
for line in q_f.readlines():
    query.append(line.strip())

for line in t_f.readlines():
    gallery.append(line.strip())
dist = vehicle_pairwise_distance(query_feature, test_feature, query, gallery)
dist = dist.numpy()

qf_new = []
T = 14 
num = 2
d_max = 10.0

for t in range(num):
    qf_new = []
    for i in range(len(dist)):
        indice = np.argsort(dist[i])[:T]
        temp = np.concatenate((qf[i][np.newaxis, :], gf[indice]), axis=0)
        qf_new.append(np.mean(temp, axis=0, keepdims=True))

    qf = np.squeeze(np.array(qf_new))
    # feature norm
    # q_n = np.linalg.norm(qf, axis=1, keepdims=True)
    # qf = qf / q_n

q_file = open(root_path + output_file, 'wb')
query_features = OrderedDict()
for i in range(1, 1053):
    key = str(i).zfill(6) + '.jpg'
    query_features[key] = qf[i - 1]
pickle.dump(query_features, q_file)
q_file.close()
print('Done')
