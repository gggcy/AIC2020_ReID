#!/bin/bash
# args for datapath
query_path=/data/home/cunyuangao/AIC20_ReID/image_query
gallery_path=/data/home/cunyuangao/AIC20_ReID/image_test
python ./reid/extract_fea_from_val.py \
    --arch ResNet152_mgn_lr \
    --resume ./ckpt/mgn_hastrip/model_best.pth.tar \
    --q_file query_mgn_off_cen.pkl \
    --g_file gallery_mgn_off_cen.pkl \
    --num_classes 1645 \
    --query_dir $query_path \
    --gallery_dir $gallery_path
echo "Finish extract from mgn"

