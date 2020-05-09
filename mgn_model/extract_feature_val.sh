#!/bin/bash
# args for datapath
query_path=/raid/home/ferdinandhu/datas/AIC2020/AIC20_track2/AIC20_ReID/image_query
gallery_path=/raid/home/ferdinandhu/datas/AIC2020/AIC20_track2/AIC20_ReID/image_test
pkl_out_dir=../post_processing/val_pkl_final
if [ ! -d $pkl_out_dir ]; then
    mkdir -p $pkl_out_dir
fi

python ./reid/extract_fea_from_val.py \
    --arch ResNet152_mgn_lr \
    --resume ./ckpt/mgn_hastrip/model_best.pth.tar \
    --q_file $pkl_out_dir/query_mgn_off_cen.pkl \
    --g_file $pkl_out_dir/gallery_mgn_off_cen.pkl \
    --num_classes 1645 \
    --query_dir $query_path \
    --gallery_dir $gallery_path
echo "Finish extract from mgn"

