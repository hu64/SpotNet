#!/usr/bin/env bash

cd src

# 2 conv + attention, learning rate / 8, BCEwithlogitsloss
# train
# python main.py ctdet --val_intervals 1 --exp_id seg1_pawcs_2conv_attention_BCE_DB --seg_weight 1 --dataset uadetrac1on10_b --arch hourglass --batch_size 2 --master_batch 4 --lr 1e-5 --load_model ../models/ctdet_coco_hg.pth --gpu 0
# test
# python test.py ctdet --test --exp_id seg1_pawcs_2conv_attention_BCE_DB --dataset uadetrac1on10_b --arch hourglass --keep_res --load_model /store/datasets/UA-Detrac/exp/ctdet/seg1_pawcs_2conv_attention_BCE_DB/model_best.pth --gpu 1


# STD
# train
# python main.py ctdet --val_intervals 1 --exp_id CTNET-STD --seg_weight 0 --dataset uadetrac1on10_b --arch hourglass --batch_size 2 --master_batch 4 --lr 1e-5 --load_model ../models/ctdet_coco_hg.pth --gpu 0
# python main.py ctdet --val_intervals 1 --exp_id CTNET-STD --seg_weight 0 --dataset uadetrac1on10_b --arch hourglass --batch_size 2 --master_batch 4 --lr 1e-5 --resume --gpu 0
# test
python test.py ctdet --test --exp_id CTNET-STD --dataset uadetrac1on10_b --arch hourglass --keep_res --load_model /store/datasets/UA-Detrac/exp/ctdet/CTNET-STD/model_best.pth --gpu 0

# 2 conv, learning rate / 8, BCEwithlogitsloss
# train
# python main.py ctdet --val_intervals 1 --exp_id seg1_pawcs_2conv_BCE_DB --seg_weight 1 --dataset uadetrac1on10_b --arch hourglass --batch_size 2 --master_batch 4 --lr 1e-5 --load_model ../models/ctdet_coco_hg.pth --gpu 1
# test
# python test.py ctdet --test --exp_id seg1_pawcs_2conv_BCE_DB --dataset uadetrac1on10_b --arch hourglass --keep_res --load_model /store/datasets/UA-Detrac/exp/ctdet/seg1_pawcs_2conv_BCE_DB/model_best.pth --gpu 0

