#!/usr/bin/env bash

cd src

# train
# python main.py ctdet --val_intervals 1 --exp_id seg1_pawcs --seg_weight 1 --dataset uadetrac1on10 --arch hourglass --batch_size 1 --master_batch 4 --lr 4.0e-6 --load_model ../models/ctdet_coco_hg.pth --gpu 0
# test
# python test.py ctdet --test --exp_id seg1_pawcs --dataset uadetrac1on10 --arch hourglass --keep_res --load_model /store/datasets/UA-Detrac/exp/ctdet/seg1_pawcs/model_best.pth --gpu 1


# 1 conv + attention, learning rate / 8
# train
# python main.py ctdet --val_intervals 1 --exp_id seg1_pawcs_1conv_attention --seg_weight 1 --dataset uadetrac1on10 --arch hourglass --batch_size 4 --master_batch 4 --lr 4e-5 --load_model ../models/ctdet_coco_hg.pth --gpu 1
# test
# python test.py ctdet --test --exp_id seg1_pawcs_1conv_attention --dataset uadetrac1on10 --arch hourglass --keep_res --load_model /store/datasets/UA-Detrac/exp/ctdet/seg1_pawcs_1conv_attention/model_best.pth --gpu 1

# 1 conv + attention, learning rate / 8, seg0.1
# train
# python main.py ctdet --val_intervals 1 --exp_id seg0.1_pawcs_1conv_attention --seg_weight 0.1 --dataset uadetrac1on10 --arch hourglass --batch_size 4 --master_batch 4 --lr 4e-5 --load_model ../models/ctdet_coco_hg.pth --gpu 0
# test
# python test.py ctdet --test --exp_id seg0.1_pawcs_1conv_attention --dataset uadetrac1on10 --arch hourglass --keep_res --load_model /store/datasets/UA-Detrac/exp/ctdet/seg0.1_pawcs_1conv_attention/model_best.pth --gpu 0

# 0 conv sqEx + attention, learning rate / 8
# train
# python main.py ctdet --val_intervals 1 --exp_id seg1_pawcs_0conv_sqEx_attention --seg_weight 1 --dataset uadetrac1on10 --arch hourglass --batch_size 4 --master_batch 4 --lr 4e-5 --load_model ../models/ctdet_coco_hg.pth --gpu 1
# test
# python test.py ctdet --test --exp_id seg1_pawcs_0conv_sqEx_attention --dataset uadetrac1on10 --arch hourglass --keep_res --load_model /store/datasets/UA-Detrac/exp/ctdet/seg1_pawcs_0conv_sqEx_attention/model_best.pth --gpu 1

# 0 conv + attention, learning rate / 8
# train
# python main.py ctdet --val_intervals 1 --exp_id seg1_pawcs_0conv_attention --seg_weight 1 --dataset uadetrac1on10 --arch hourglass --batch_size 4 --master_batch 4 --lr 4e-5 --load_model ../models/ctdet_coco_hg.pth --gpu 1
# test
# python test.py ctdet --test --exp_id seg1_pawcs_0conv_attention --dataset uadetrac1on10 --arch hourglass --keep_res --load_model /store/datasets/UA-Detrac/exp/ctdet/seg1_pawcs_0conv_attention/model_best.pth --gpu 1

# 2 conv + attention, learning rate / 8 (try reducing learning rate)
# train
# python main.py ctdet --val_intervals 1 --exp_id seg1_pawcs_2conv_attention --seg_weight 1 --dataset uadetrac1on10 --arch hourglass --batch_size 2 --master_batch 4 --lr 1e-5 --load_model ../models/ctdet_coco_hg.pth --gpu 1
# test
# python test.py ctdet --test --exp_id seg1_pawcs_2conv_attention --dataset uadetrac1on10 --arch hourglass --keep_res --load_model /store/datasets/UA-Detrac/exp/ctdet/seg1_pawcs_2conv_attention/model_best.pth --gpu 1

# 2 conv + attention, learning rate / 8, BCEwithlogitsloss
# train
# python main.py ctdet --val_intervals 1 --exp_id seg1_pawcs_2conv_attention_BCE --seg_weight 1 --dataset uadetrac1on10 --arch hourglass --batch_size 2 --master_batch 4 --lr 1e-5 --load_model ../models/ctdet_coco_hg.pth --gpu 1
# python main.py ctdet --val_intervals 1 --exp_id seg1_pawcs_2conv_attention_BCE --seg_weight 1 --dataset uadetrac1on10 --arch hourglass --batch_size 2 --master_batch 4 --lr 1e-5 --resume --gpu 1
# test
# python test.py ctdet --test --exp_id seg1_pawcs_2conv_attention_BCE --dataset uadetrac1on10 --arch hourglass --keep_res --load_model /store/datasets/UA-Detrac/exp/ctdet/seg1_pawcs_2conv_attention_BCE/model_best.pth --gpu 0
# python test.py ctdet --test --exp_id seg1_pawcs_2conv_attention_BCE --dataset uadetrac1on10 --arch hourglass --keep_res --load_model /store/datasets/UA-Detrac/exp/ctdet/seg1_pawcs_2conv_attention_BCE/model_last.pth --gpu 0

# 2 conv, learning rate / 8, BCEwithlogitsloss
# train
# python main.py ctdet --val_intervals 1 --exp_id seg1_pawcs_2conv_BCE --seg_weight 1 --dataset uadetrac1on10 --arch hourglass --batch_size 2 --master_batch 4 --lr 1e-5 --load_model ../models/ctdet_coco_hg.pth --gpu 1
# test
# python test.py ctdet --test --exp_id seg1_pawcs_2conv_BCE --dataset uadetrac1on10 --arch hourglass --keep_res --load_model /store/datasets/UA-Detrac/exp/ctdet/seg1_pawcs_2conv_BCE/model_best.pth --gpu 0

# 3 conv + attention, learning rate / 8, BCEwithlogitsloss
# train
# python main.py ctdet --val_intervals 1 --exp_id seg1_pawcs_3conv_attention_BCE --seg_weight 1 --dataset uadetrac1on10 --arch hourglass --batch_size 2 --master_batch 4 --lr 1e-5 --load_model ../models/ctdet_coco_hg.pth --gpu 1
# python main.py ctdet --val_intervals 1 --exp_id seg1_pawcs_3conv_attention_BCE --seg_weight 1 --dataset uadetrac1on10 --arch hourglass --batch_size 2 --master_batch 4 --lr 1e-5 --resume --gpu 1
# test
# python test.py ctdet --test --exp_id seg1_pawcs_3conv_attention_BCE --dataset uadetrac1on10 --arch hourglass --keep_res --load_model /store/datasets/UA-Detrac/exp/ctdet/seg1_pawcs_3conv_attention_BCE/model_best.pth --gpu 0

# 2 conv + attention, learning rate / 8, BCEwithlogitsloss, DatasetFull
# train
# python main.py ctdet --val_intervals 1 --exp_id seg1_pawcs_2conv_attention_BCE_DF --seg_weight 1 --dataset uadetrac --arch hourglass --batch_size 2 --master_batch 4 --lr 1e-5 --load_model ../models/ctdet_coco_hg.pth --gpu 1
# test
# python test.py ctdet --test --exp_id seg1_pawcs_2conv_attention_BCE_DF --dataset uadetrac1on10 --arch hourglass --keep_res --load_model /store/datasets/UA-Detrac/exp/ctdet/seg1_pawcs_2conv_attention_BCE_DF/model_best.pth --gpu 0
