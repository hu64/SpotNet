import os
import cv2
import numpy as np
file_names = ['/store/datasets/UA-Detrac/train-tf-all.csv',
              '/store/datasets/UA-Detrac/val-tf-all.csv',
              '/store/datasets/UA-Detrac/test-tf-all.csv']

lines = []
for file_name in file_names:
    lines += open(file_name, 'r').readlines()

seq_to_lines = {}
for line in lines:
    seq = os.path.dirname(line.split(',')[0]).split('/')[-1]
    if seq in seq_to_lines:
        seq_to_lines[seq].append(line)
    else:
        seq_to_lines[seq] = [line]

for seq in seq_to_lines:
    if '63561' in seq or '63562' in seq or '63563' in seq:

        lines = seq_to_lines[seq]

        img_to_mask = {}
        for line in lines:
            items = line.split(',')
            x0, y0, x1, y1 = [int(item) for item in items[1:5]]
            if x1-x0 <= 0 or y1-y0 <= 0:
                continue
            img_name = items[0]
            if img_name in img_to_mask:
                img_to_mask[img_name][y0:y1, x0:x1] += 1
                img_to_mask[img_name] = np.minimum(1, img_to_mask[img_name])
            else:
                img_to_mask[img_name] = np.zeros((540, 960)).astype(np.uint8)
                img_to_mask[img_name][y0:y1, x0:x1] += 1

        frame_names = [item.split(',')[0] for item in lines]
        img_names = set(frame_names)

        for img_name in img_names:
            seq = os.path.dirname(img_name).split('/')[-1]
            seg_path = os.path.join('/store/datasets/UA-Detrac/uadetrac-bgs/pawcs', seq, 'With mask/PAWCS', os.path.basename(img_name).replace('jpg', 'png'))
            if not os.path.exists(seg_path):
              seg_path = os.path.join('/store/datasets/UA-Detrac/uadetrac-bgs/pawcs', seq, 'PAWCS', os.path.basename(img_name).replace('jpg', 'png'))
              if not os.path.exists(seg_path):
                seg_path = os.path.join(os.path.dirname(img_name), 'bg-sub',  os.path.basename(img_name).replace('jpg', 'png'))  # hughes

            seg_img = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
            new_seg_img = seg_img * img_to_mask[img_name]
            new_seg_path = os.path.join('/store/datasets/UA-Detrac/uadetrac-bgs/pawcs_and_bbox', seq, os.path.basename(img_name).replace('jpg', 'png'))
            if not os.path.exists(os.path.dirname(new_seg_path)):
                os.mkdir(os.path.dirname(new_seg_path))
            cv2.imwrite(new_seg_path, new_seg_img)