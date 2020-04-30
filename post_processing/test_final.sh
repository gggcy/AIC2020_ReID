#!/bin/bash
# args for concat feature
rootpath=./val_pkl_final/
single_query_feature_pkls=query_densenet_off_cen.pkl,query_densenet_ALL_off_cen.pkl,query_mgn_off_cen.pkl,query_senet152_off_cen.pkl
single_gallery_feature_pkls=gallery_densenet_off_cen.pkl,gallery_densenet_ALL_off_cen.pkl,gallery_mgn_off_cen.pkl,gallery_senet152_off_cen.pkl
concat_query_feature=query_cat_4_0410_off_cen.pkl
concat_gallery_feature=gallery_cat_4_0410_off_cen.pkl
# args for query expansion
feature_dimension=8192 # len(pkls)*2048
concat_query_feature_after_expansion=after_expansion_query_cat_6_0330_off_cen.pkl
# agrs for gallery feature merge
concat_gallery_feature_after_merge=after_gallery_cat_4_0410_off_cen.pkl
feature_dimension=8192 # len(pkls)*2048
T_for_merge=100 # images under one track for merge eg.6 or 100 can cover all images under a track
# args for re-ranking
k1=15
k2=6
lambda=0.5
distmat=rerank_dismat_150605.pkl
python concat_feature_val.py $rootpath $single_query_feature_pkls $single_gallery_feature_pkls $concat_query_feature $concat_gallery_feature
echo 'finish concat feature'
wait
#python query_expansion.py $rootpath $concat_gallery_feature $concat_query_feature $feature_dimension $concat_query_feature_after_expansion
#wait
#echo 'finish query_expansion'
#wait
python gallery_feature.py $rootpath $concat_gallery_feature $concat_gallery_feature_after_merge $feature_dimension $T_for_merge
echo 'finish gallery feature merge'
wait
python run_rerank_val.py $rootpath $k1 $k2 $lambda $concat_query_feature $concat_gallery_feature_after_merge $distmat
echo 'finish re-ranking'
wait
python generate_result_val.py $rootpath $distmat
echo 'finish generate result'
wait
