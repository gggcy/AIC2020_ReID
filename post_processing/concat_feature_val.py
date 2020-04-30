import os
import pickle
import numpy as np
from sys import version_info


def load_features(pickle_file_path):
    pickle_file = open(pickle_file_path, 'rb')
    if version_info.major == 2:
        features = pickle.load(pickle_file)
    elif version_info.major == 3:
        features = pickle.load(pickle_file, encoding='iso-8859-1')
    pickle_file.close()
    return features


if __name__ == '__main__':
    print('running...')
    import sys

    root_path = sys.argv[1].strip()
    query_file_list = sys.argv[2]
    test_file_list = sys.argv[3]

    query_res_file = sys.argv[4]
    test_res_file = sys.argv[5]
    # root_path = '/data/home/cunyuangao/Project/baidu/Track2(ReID)/part1_model/val_pkl2/'
    # query_file_list = ['query_101_1695.pkl','query_152_1695.pkl','query_attr_333.pkl','query_hrnet.pkl']
    # query_file_list = ['query_mgn_off_cen.pkl','query_se152new_off_cen.pkl','query_hrnet1to1_off_cen.pkl','query_reid152n8_off_cen.pkl','query_se152ibn_off_cen.pkl','query_densenet_off_cen.pkl']
    # test_file_list = ['gallery_101_1695.pkl','gallery_152_1695.pkl','gallery_attr_333.pkl','gallery_hrnet.pkl']
    # test_file_list = ['gallery_mgn_off_cen.pkl','gallery_se152new_off_cen.pkl','gallery_hrnet1to1_off_cen.pkl','gallery_reid152n8_off_cen.pkl','gallery_se152ibn_off_cen.pkl','gallery_densenet_off_cen.pkl']
    query_file_list = query_file_list.strip().split(',')
    test_file_list = test_file_list.strip().split(',')

    query_feature_list = []
    test_feature_list = []
    for i in range(len(test_file_list)):
        query_feature = load_features(root_path + query_file_list[i].strip())
        query_feature_list.append(query_feature)
        test_feature = load_features(root_path + test_file_list[i].strip())
        test_feature_list.append(test_feature)

    concat_query_features = {}
    for k in query_feature_list[0].keys():
        concat_query_features[k] = np.concatenate([query_feature_list[i][k] for i in range(len(query_feature_list))])

    concat_test_features = {}
    for k in test_feature_list[0].keys():
        concat_test_features[k] = np.concatenate([test_feature_list[i][k] for i in range(len(test_feature_list))])

    # query_res_file = open(root_path + 'query_cat_6_0330_off_cen.pkl', 'wb')
    query_res_file = open(root_path + query_res_file, 'wb')
    pickle.dump(concat_query_features, query_res_file)
    query_res_file.close()
    test_res_file = open(root_path + test_res_file, 'wb')
    # test_res_file = open(root_path + 'gallery_cat_6_0330_off_cen.pkl', 'wb')
    pickle.dump(concat_test_features, test_res_file)
    test_res_file.close()
    print('Done')
