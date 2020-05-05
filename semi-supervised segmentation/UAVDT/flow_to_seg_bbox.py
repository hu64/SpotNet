import os
import cv2
import numpy as np
from PIL import Image

DATASET = 'DETRAC'

if DATASET == 'UAV':
    file_names = ['/store/datasets/UAV/val.csv', '/store/datasets/UAV/train.csv', '/store/datasets/UAV/val-sub.csv']
    seg_dir = '/store/datasets/UAV/bgsubs'
    FLOW_DIR = '/store/datasets/UAV/pyflow'
elif DATASET == 'DETRAC':
    file_names = ['/store/datasets/UA-Detrac/train-tf.csv', '/store/datasets/UA-Detrac/val-tf-all.csv',
                 '/store/datasets/UA-Detrac/test-tf-1-on-30.csv']
    seg_dir = '/store/datasets/UA-Detrac/pyflow-bgsubs'
    FLOW_DIR = '/store/datasets/UA-Detrac/pyflow'
"""
elif DATASET == 'IMAGENET':
    file_names = ['/store/datasets/ILSVRC2015/train_1_in_10.csv, /store/datasets/ILSVRC2015/minival_1_in_10.csv']
    seg_dir = '/store/datasets/UAV/bgsubs'
    FLOW_DIR = '/store/datasets/UAV/pyflow'
"""
lines = set()
for file_name in file_names:
    for line in open(file_name, 'r').readlines():
        lines.add(line)
lines = list(lines)

seq_to_lines = {}
for line in lines:
    seq = os.path.dirname(line.split(',')[0]).split('/')[-1]
    if seq in seq_to_lines:
        seq_to_lines[seq].append(line)
    else:
        seq_to_lines[seq] = [line]

for seq in sorted(seq_to_lines):
    print(seq)
    seq_lines = seq_to_lines[seq]
    img_to_mask = {}

    for count, line in enumerate(sorted(seq_lines)):
        if count % 1000 == 0:
            print('   progress within seq: ' + str(round(100*(count/len(seq_lines)), 2)) + '%')
        items = line.split(',')
        x0, y0, x1, y1 = [int(item) for item in items[1:5]]
        # if x1-x0 <= 0 or y1-y0 <= 0:
        #     continue
        img_name = items[0]
        im = Image.open(img_name)
        width, height = im.size
        if img_name in img_to_mask:
            img_to_mask[img_name][y0:y1, x0:x1] += 1
            img_to_mask[img_name] = np.minimum(1, img_to_mask[img_name])
        else:
            img_to_mask[img_name] = np.zeros((height, width)).astype(np.uint8)
            img_to_mask[img_name][y0:y1, x0:x1] += 1

    frame_names = [item.split(',')[0] for item in seq_lines]
    img_names = set(frame_names)
    for img_name in sorted(img_names):
        seq = os.path.dirname(img_name)
        seq_name = os.path.dirname(img_name).split('/')[-1]
        file_name = os.path.basename(img_name)
        seg_path = os.path.join(FLOW_DIR, seq_name, file_name.replace('.jpg', '.png'))
        if os.path.exists(seg_path):
            seg_img = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
            new_seg_img = seg_img * img_to_mask[img_name]
            new_seg_img = cv2.GaussianBlur(new_seg_img, (17, 17), cv2.BORDER_DEFAULT)

            # new_seg_img = new_seg_img**4
            THRESHOLD = 24
            new_seg_img[new_seg_img <= THRESHOLD] = 0
            new_seg_img[new_seg_img > THRESHOLD] = 255

            new_seg_path = os.path.join(seg_dir, seq_name, file_name.replace('jpg', 'png'))
            # new_seg_path = os.path.join(seq, 'flow_bbox', file_name.replace('jpg', 'png'))
            if not os.path.exists(os.path.dirname(new_seg_path)):
                os.mkdir(os.path.dirname(new_seg_path))
            cv2.imwrite(new_seg_path, new_seg_img)
