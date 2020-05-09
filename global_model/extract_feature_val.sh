#!/bin/bash
# args for datapath
query_path=/raid/home/ferdinandhu/datas/AIC2020/AIC20_track2/AIC20_ReID/image_query #the image query path of Track2's testset
gallery_path=/raid/home/ferdinandhu/datas/AIC2020/AIC20_track2/AIC20_ReID/image_test # the image gallery path of Track2's testset
pkl_out_dir=../post_processing/val_pkl_final

if [ ! -d $pkl_out_dir ]; then
    mkdir -p $pkl_out_dir
fi

python ./reid/extract_fea_from_val.py \
    --arch densenet161 \
    --resume ./ckpt/densenet161_fake/pass520.pth.tar \
    --q_file $pkl_out_dir/query_densenet_ALL_off_cen.pkl \
    --g_file $pkl_out_dir/gallery_densenet_ALL_off_cen.pkl \
    --num_classes 1978 \
    --query_dir $query_path \
    --gallery_dir $gallery_path
echo "Finish extract from densenet161"
wait
python ./reid/extract_fea_from_val.py \
    --arch densenet161 \
    --resume ./ckpt/densenet161/pass861.pth.tar \
    --q_file $pkl_out_dir/query_densenet_off_cen.pkl \
    --g_file $pkl_out_dir/gallery_densenet_off_cen.pkl \
    --num_classes 1645 \
    --query_dir $query_path \
    --gallery_dir $gallery_path
echo "Finish extract from densenet161"
wait
python ./reid/extract_fea_from_val.py \
    --arch cross_trihard_se_resnet152 \
    --resume ./ckpt/senet152/process_0_model_best.pth.tar\
    --q_file $pkl_out_dir/query_senet152_off_cen.pkl \
    --g_file $pkl_out_dir/gallery_senet152_off_cen.pkl \
    --num_classes 1645 \
    --query_dir $query_path \
    --gallery_dir $gallery_path
echo "Finish extract from seresnet152"
