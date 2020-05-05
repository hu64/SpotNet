#!/usr/bin/env bash
cd src
# train
# python main.py ctdet --val_intervals 1 --exp_id ua-detrac_512-512_hg --dataset ua-detrac --arch hourglass --batch_size 4 --master_batch 4 --lr 2.5e-4 --load_model ../models/ctdet_coco_hg.pth --gpus 1

python test.py ctdet --test --exp_id ua-detrac_512-512_hg --dataset uadetrac --arch hourglass --keep_res --load_model /store/datasets/UA-Detrac/exp/ctdet/ua-detrac_512-512_hg/model_best.pth
