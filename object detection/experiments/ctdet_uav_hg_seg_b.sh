#!/usr/bin/env bash

cd src

# 2 conv + attention, learning rate / 8, BCEwithlogitsloss
# train
# python main.py ctdet --val_intervals 1 --exp_id UAV-seg1_pawcs_2conv_attention_BCE --seg_weight 1 --dataset uav --arch hourglass --batch_size 2 --master_batch 4 --lr 1e-5 --load_model ../models/ctdet_coco_hg.pth --gpu 1
# test
# python test.py ctdet --test --exp_id UAV-seg1_pawcs_2conv_attention_BCE --seg_weight 1 --dataset uav --arch hourglass --keep_res --load_model /store/datasets/UA-Detrac/exp/ctdet/UAV-seg1_pawcs_2conv_attention_BCE/model_best.pth --gpu 0


# STD
# train
# python main.py ctdet --val_intervals 1 --exp_id UAV-CTNET-STD --seg_weight 0 --dataset uav --arch hourglass --batch_size 2 --master_batch 4 --lr 1e-5 --load_model ../models/ctdet_coco_hg.pth --gpu 1
# test
# python test.py ctdet --test --exp_id UAV-CTNET-STD --dataset uav --arch hourglass --keep_res --load_model /store/datasets/UA-Detrac/exp/ctdet/UAV-CTNET-STD/model_best.pth --gpu 0

# SpotNet2 ADD U-NET
# train
# python main.py ctdet --val_intervals 1 --exp_id UAV-SpotNet2_ADD --seg_weight 1 --dataset uav --arch hourglass --batch_size 1 --master_batch 4 --lr 1e-5 --load_model ../models/ctdet_coco_hg.pth --gpu 1
# test
# python test.py ctdet --test --exp_id UAV-SpotNet2_ADD --seg_weight 1 --dataset uav --arch hourglass --keep_res --load_model /store/datasets/UA-Detrac/exp/ctdet/UAV-SpotNet2_ADD/model_best.pth --gpu 1

# SpotNet2 ADD U-NET LR /
# train
# python main.py ctdet --val_intervals 1 --exp_id UAV-SpotNet2_ADD_LR2e6 --seg_weight 1 --dataset uav --arch hourglass --batch_size 1 --master_batch 4 --lr 2e-6 --load_model ../models/ctdet_coco_hg.pth --gpu 1
# test
# python test.py ctdet --test --exp_id UAV-SpotNet2_ADD_LR2e6 --seg_weight 1 --dataset uav --arch hourglass --keep_res --load_model /store/datasets/UA-Detrac/exp/ctdet/UAV-SpotNet2_ADD_LR2e6/model_best.pth --gpu 1

# SpotNet2 ADD U-NET LR /
# train
# python main.py ctdet --val_intervals 1 --exp_id UAV-SpotNet2_ADD_LR2e6 --seg_weight 1 --dataset uav --arch hourglass --batch_size 1 --master_batch 4 --lr 2e-6 --load_model ../models/ctdet_coco_hg.pth --gpu 1
# test
# python test.py ctdet --test --exp_id UAV-SpotNet2_ADD_LR2e6 --seg_weight 1 --dataset uav --arch hourglass --keep_res --load_model /store/datasets/UA-Detrac/exp/ctdet/UAV-SpotNet2_ADD_LR2e6/model_best.pth --gpu 1

# SpotNet2 MUL U-NET LR FROM-DETRAC/
# train
# python main.py ctdet --val_intervals 1 --exp_id UAV-SpotNet2_ADD_LR2e6_DETRAC --seg_weight 1 --dataset uav --arch hourglass --batch_size 1 --master_batch 4 --lr 2e-6 --load_model /store/datasets/UA-Detrac/exp/ctdet/SpotNet2_ADD_DB_PYFLOW_LR2e6/model_best.pth --gpu 1
# test
python test.py ctdet --test --exp_id UAV-SpotNet2_ADD_LR2e6_DETRAC --seg_weight 1 --dataset uav --arch hourglass --keep_res --load_model /store/datasets/UA-Detrac/exp/ctdet/UAV-SpotNet2_ADD_LR2e6_DETRAC/model_best.pth --gpu 1
