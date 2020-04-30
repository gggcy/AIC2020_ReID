export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
python train.py \
    -a ResNet152_mgn_lr \
    -b 120\
    -d gao_crop_train \
    --frozen_sublayer True \
    --weights ./weights/resnet152-b121ed2d.pth \
    --logs-dir ./logs/mgn_2048_240/ \
    --lr 0.01 \
    --gamma 0.1 \
    --weight-decay 0.0005 \
    --warm_up_factor 0.01 \
    --warm_up_iter 100 \
    --step_epoch 600 900 1000 \
    --epochs 1100 \
    --lr_mult 1.0 \
    --metric_loss_weight 1 \
    --big_height 288 \
    --big_width 384 \
    --target_height 288 \
    --target_width 384 \
    --epoch_inter 100 \
    --start_save 50 \
    --dense_evaluate 1090 \
    --num_instances 4 \
    --features 2048 \
    --combine-trainval \


