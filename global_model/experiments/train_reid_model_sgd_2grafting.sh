out_dir=./output/1to1_se_resnet152_huyi_0326_sgd_2grafting
if [ ! -d $out_dir ]; then
    mkdir -p $out_dir
fi
nohup python -u train_with_grafting.py \
    -a cross_trihard_se_resnet152  \
    -b 64 \
    -d complete_aicity_car \
    --logs-dir $out_dir \
    --weights ./pretrain_models/se_resnet152-d17c99b7.pth\
    --optimizer 0 \
    --lr 3e-2 \
    --weight-decay 0.0005 \
    --epochs 1100  \
    --step_size 300 \
    --step_size2 600 \
    --step_size3 900 \
    --lr_mult 1.0 \
    --metric_loss_weight 1 \
    --big_height 310 \
    --big_width 414 \
    --target_height 288 \
    --target_width 384 \
    --epoch_inter 20 \
    --start_save 20 \
    --dense_evaluate 1000 \
    --warm_up_ep 20 \
    --features 2048 \
    --gpus=0,1 \
    --nums 2 \
    --i 1 \
    > $out_dir/train_1.log 2>&1 &

nohup python -u train_with_grafting.py \
    -a cross_trihard_se_resnet152  \
    -b 64 \
    -d complete_aicity_car \
    --logs-dir $out_dir \
    --weights ./pretrain_models/se_resnet152-d17c99b7.pth\
    --optimizer 0 \
    --lr 3e-2 \
    --weight-decay 0.0005 \
    --epochs 1100  \
    --step_size 300 \
    --step_size2 600 \
    --step_size3 900 \
    --lr_mult 1.0 \
    --metric_loss_weight 1 \
    --big_height 310 \
    --big_width 414 \
    --target_height 288 \
    --target_width 384 \
    --epoch_inter 20 \
    --start_save 20 \
    --dense_evaluate 1000 \
    --warm_up_ep 20 \
    --features 2048 \
    --gpus=2,3 \
    --nums 2 \
    --i 2 \
    > $out_dir/train_2.log 2>&1 &