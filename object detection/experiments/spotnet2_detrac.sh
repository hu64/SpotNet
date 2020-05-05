#!/usr/bin/env bash

cd src

# train
# python main.py ctdet --val_intervals 1 --exp_id SpotNet2_V0_DB --seg_weight 1 --dataset uadetrac1on10_b --arch hourglass --batch_size 1 --master_batch 4 --lr 1e-5 --load_model ../models/ctdet_coco_hg.pth --gpu 1
# test
# python test.py ctdet --test --exp_id SpotNet2_V0_DB --dataset uadetrac1on10_b --arch hourglass --keep_res --load_model /store/datasets/UA-Detrac/exp/ctdet/SpotNet2_V0_DB/model_best.pth --gpu 1

# V add
# train
# python main.py ctdet --val_intervals 1 --exp_id SpotNet2_ADD_DB --seg_weight 1 --dataset uadetrac1on10_b --arch hourglass --batch_size 1 --master_batch 4 --lr 1e-5 --load_model ../models/ctdet_coco_hg.pth --gpu 0
# test
# python test.py ctdet --test --exp_id SpotNet2_ADD_DB --dataset uadetrac1on10_b --arch hourglass --keep_res --load_model /store/datasets/UA-Detrac/exp/ctdet/SpotNet2_ADD_DB/model_best.pth --gpu 1


# tests
# python main.py ctdet --val_intervals 1 --exp_id SpotNet2_V0_DB_tests --seg_weight 1 --dataset uadetrac1on10_b --arch hourglass --batch_size 1 --master_batch 4 --lr 1e-5 --load_model ../models/ctdet_coco_hg.pth --gpu 0

# V add flow
# train
# python main.py ctdet --val_intervals 1 --exp_id SpotNet2_ADD_DB_PYFLOW --seg_weight 1 --dataset uadetrac1on10_b --arch hourglass --batch_size 1 --master_batch 4 --lr 1e-5 --load_model ../models/ctdet_coco_hg.pth --gpu 1
# test
# python test.py ctdet --test --exp_id SpotNet2_ADD_DB_PYFLOW --dataset uadetrac1on10_b --arch hourglass --keep_res --load_model /store/datasets/UA-Detrac/exp/ctdet/SpotNet2_ADD_DB_PYFLOW/model_best.pth --gpu 1

# V add flow LR2e6
# train
# python main.py ctdet --val_intervals 1 --exp_id SpotNet2_ADD_DB_PYFLOW_LR2e6 --seg_weight 1 --dataset uadetrac1on10_b --arch hourglass --batch_size 1 --master_batch 4 --lr 2e-6 --load_model ../models/ctdet_coco_hg.pth --gpu 0
# test
# python test.py ctdet --test --exp_id SpotNet2_ADD_DB_PYFLOW_LR2e6 --dataset uadetrac1on10_b --arch hourglass --keep_res --load_model /store/datasets/UA-Detrac/exp/ctdet/SpotNet2_ADD_DB_PYFLOW_LR2e6/model_best.pth --gpu 1

# V mul flow LR2e6
# train
# python main.py ctdet --val_intervals 1 --exp_id SpotNet2_MUL_DB_PYFLOW_LR2e6 --seg_weight 1 --dataset uadetrac1on10_b --arch hourglass --batch_size 1 --master_batch 4 --lr 2e-6 --load_model ../models/ctdet_coco_hg.pth --gpu 1
# test
# python test.py ctdet --test --exp_id SpotNet2_MUL_DB_PYFLOW_LR2e6 --dataset uadetrac1on10_b --arch hourglass --keep_res --load_model /store/datasets/UA-Detrac/exp/ctdet/SpotNet2_MUL_DB_PYFLOW_LR2e6/model_best.pth --gpu 1

# V add flow LR2e6 No U-Net
# train
python main.py ctdet --val_intervals 1 --exp_id SpotNet2_ADD_DB_PYFLOW_LR2e6_NOUNET --seg_weight 1 --dataset uadetrac1on10_b --arch hourglass --batch_size 1 --master_batch 4 --lr 2e-6 --load_model ../models/ctdet_coco_hg.pth --gpu 1
# test
# python test.py ctdet --test --exp_id SpotNet2_ADD_DB_PYFLOW_LR2e6_NOUNET --dataset uadetrac1on10_b --arch hourglass --keep_res --load_model /store/datasets/UA-Detrac/exp/ctdet/SpotNet2_ADD_DB_PYFLOW_LR2e6_NOUNET/model_best.pth --gpu 1
