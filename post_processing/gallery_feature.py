import numpy as np
import pickle
from collections import OrderedDict
from sys import version_info

import sys

root_path = sys.argv[1]
gallery_concat_feature = sys.argv[2]
gallery_concat_feature_after_merge = sys.argv[3]
feature_dimension = int(sys.argv[4])


#root_path = '/data/home/cunyuangao/Project/baidu/Track2(ReID)/part1_model/val_pkl2/'
#gallery_concat_feature_after_merge
g_file = open(root_path + gallery_concat_feature_after_merge, 'wb')
picklefile=open(root_path + gallery_concat_feature,'rb')

# g_file = open(root_path + 'after100_gallery_cat_6_0330_off_cen.pkl', 'wb')
# picklefile=open(root_path + 'gallery_cat_6_0330_off_cen.pkl','rb')
if version_info.major == 3:
    gf=pickle.load(picklefile,encoding='iso-8859-1')
elif version_info.major == 2:
    gf=pickle.load(picklefile)
gf_npy_int = np.zeros([18290,feature_dimension])
# gf_npy_int = np.zeros([18290,12288])
gf_npy = gf_npy_int.astype(np.float32)

for key,value in gf.items():
    # print(key,value)
    key_ind = int(key.split('.')[0]) - 1
    gf_npy[key_ind] = value
# print(gf_npy)
track_name = []
f = open('test_track_id.txt', 'r')
for k, line in enumerate(f):
    temp = list(map(int, line.split(' ')[:-1]))
    track_name.append(list(map(lambda x: x-1, temp)))

f.close
# print(track_name)
T=100 #6
for i in range(len(track_name)):
    indice = track_name[i]
    # print("indice:   ",indice)
    for j in range(0, len(indice), T):
        if (j+T)>len(indice):
            ind = indice[j:]
        else:
            ind = indice[j:j+T]
        # print("ind:   " ,ind)
        gf_temp = np.mean(gf_npy[ind], axis=0, keepdims=True) 
        gf_npy[ind] = gf_temp



gallery_features = OrderedDict()
for i in range(1,18291):
    key = str(i).zfill(6) + '.jpg'
    gallery_features[key] = gf_npy[i-1]
pickle.dump(gallery_features, g_file)
g_file.close()
print('Done')



