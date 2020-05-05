#!/usr/bin/env bash
cd src

# train seg 0.1 pawcs
# python main.py ctdet --val_intervals 1 --exp_id ua-detrac-1-on-10_512-512_hg-seg0.1-2-pawcs --seg_weight 0.1 --dataset uadetrac1on10 --arch hourglass --batch_size 1 --master_batch 4 --lr 2.5e-4 --load_model ../models/ctdet_coco_hg.pth --gpu 1

# train seg 0
# python main.py ctdet --val_intervals 1 --exp_id ua-detrac-1-on-10_512-512_hg-seg0   --seg_weight   0 --dataset uadetrac1on10 --arch hourglass --batch_size 1 --master_batch 4 --lr 2.5e-4 --load_model ../models/ctdet_coco_hg.pth --gpu 0

# test
# python test.py ctdet --test --exp_id ua-detrac-1-on-10_512-512_hg --dataset uadetrac1on10 --arch hourglass --keep_res --load_model /store/datasets/UA-Detrac/exp/ctdet/ua-detrac-1-on-10_512-512_hg/model_best.pth

# test seg 0.1
# python test.py ctdet --test --exp_id ua-detrac-1-on-10_512-512_hg-seg0.1 --dataset uadetrac1on10 --arch hourglass --keep_res --load_model /store/datasets/UA-Detrac/exp/ctdet/ua-detrac-1-on-10_512-512_hg-seg0.1/model_best.pth --gpu 1

# test seg 0
# python test.py ctdet --test --exp_id ua-detrac-1-on-10_512-512_hg-seg0   --dataset uadetrac1on10 --arch hourglass --keep_res --load_model /store/datasets/UA-Detrac/exp/ctdet/ua-detrac-1-on-10_512-512_hg-seg0/model_best.pth --gpu 0

# test seg 0.1 pawcs
python test.py ctdet --test --exp_id ua-detrac-1-on-10_512-512_hg-seg0.1-2-pawcs --dataset uadetrac1on10 --arch hourglass --keep_res --load_model /store/datasets/UA-Detrac/exp/ctdet/ua-detrac-1-on-10_512-512_hg-seg0.1-2-pawcs/model_best.pth --gpu 1
