import numpy as np
import pickle
from collections import OrderedDict
import sys


picklefile = open('gallery_cat_4_0407_off_cen.pkl', 'rb')

gf_pkl = pickle.load(picklefile,encoding='iso-8859-1')
# gf_pkl = pickle.load(picklefile)
gf_npy_int = np.zeros([18290, 8192])

# gf_npy_int = np.zeros([18290,4096])
gf = gf_npy_int.astype(np.float32)
for key, value in gf_pkl.items():
    # print(key,value)
    key_ind = int(key.split('.')[0]) - 1
    gf[key_ind] = value

picklefile = open('query_cat_4_0407_off_cen.pkl', 'rb')

# picklefile=open(root_path + 'query_cat_2_last.pkl','rb')
qf_pkl = pickle.load(picklefile,encoding='iso-8859-1')
# qf_pkl = pickle.load(picklefile)
qf_npy_int = np.zeros([1052, 8192])
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
impo
np.save('gf.npy', gf)
np.save('qf.npy', qf)

# feature norm
# q_n = np.linalg.norm(qf, axis=1, keepdims=True)
# qf = qf / q_n


    # q_n = np.linalg.norm(qf, axis=1, keepdims=True)
    # qf = qf / q_n


