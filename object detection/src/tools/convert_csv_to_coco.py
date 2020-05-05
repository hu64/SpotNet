from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import json
import numpy as np
import cv2

io_dic = {
            '/store/datasets/ILSVRC2015/train.json':
                open('/store/datasets/ILSVRC2015/train_1_in_10.csv', 'r').readlines(),
            '/store/datasets/ILSVRC2015/val.json':
                open('/store/datasets/ILSVRC2015/minival_1_in_10.csv', 'r').readlines(),
            '/store/datasets/ILSVRC2015/test.json':
                open('/store/datasets/ILSVRC2015/val.csv', 'r').readlines(),
         }
"""
io_dic = {
            '/store/datasets/UAV/train.json':
                open('/store/datasets/UAV/train.csv', 'r').readlines(),
            '/store/datasets/UAV/train-1-on-10.json':
                open('/store/datasets/UAV/train-1-on-10.csv', 'r').readlines(),
            '/store/datasets/UAV/val.json':
                open('/store/datasets/UAV/val.csv', 'r').readlines(),
            '/store/datasets/UAV/val-1-on-30.json':
                open('/store/datasets/UAV/val-1-on-30.csv', 'r').readlines(),
            '/store/datasets/UAV/val-sub.json':
                open('/store/datasets/UAV/val-sub.csv', 'r').readlines(),
         }

io_dic = {  '/store/datasets/UA-Detrac/COCO-format/train_b.json':
                open('/store/datasets/UA-Detrac/train-tf-all.csv', 'r').readlines(),
            '/store/datasets/UA-Detrac/COCO-format/val_b.json':
                open('/store/datasets/UA-Detrac/val-tf-all.csv', 'r').readlines(),
            '/store/datasets/UA-Detrac/COCO-format/test_b.json':
                open('/store/datasets/UA-Detrac/test-tf-all.csv', 'r').readlines(),
            '/store/datasets/UA-Detrac/COCO-format/test-1-on-30_b.json':
                open('/store/datasets/UA-Detrac/test-tf-1-on-30.csv', 'r').readlines(),
            '/store/datasets/UA-Detrac/COCO-format/train-1-on-10_b.json':
                open('/store/datasets/UA-Detrac/train-tf.csv', 'r').readlines(),
            }

io_dic = {  '/store/datasets/UA-Detrac/COCO-format/train.json':
                open('/store/datasets/UA-Detrac/train-tf-all.csv', 'r').readlines(),
            '/store/datasets/UA-Detrac/COCO-format/val.json':
                open('/store/datasets/UA-Detrac/val-tf-all.csv', 'r').readlines(),
            '/store/datasets/UA-Detrac/COCO-format/test.json':
                open('/store/datasets/UA-Detrac/test-tf-all.csv', 'r').readlines(),
            '/store/datasets/UA-Detrac/COCO-format/test-1-on-30.json':
                open('/store/datasets/UA-Detrac/test-tf-1-on-30.csv', 'r').readlines(),
            '/store/datasets/UA-Detrac/COCO-format/train-1-on-10.json':
                open('/store/datasets/UA-Detrac/train-tf.csv', 'r').readlines(),
            }
"""

DEBUG = False
import os
# import _init_paths
# from utils.ddd_utils import compute_box_3d, project_to_image, alpha2rot_y
# from utils.ddd_utils import draw_box_3d, unproject_2d_to_3d

'''
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
'''

def _bbox_to_coco_bbox(bbox):
  return [(bbox[0]), (bbox[1]),
          (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]

cats = ['bus', 'car', 'others', 'van']
cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}
cat_info = []
for i, cat in enumerate(cats):
  cat_info.append({'name': cat, 'id': i + 1})


for outputfile in io_dic:

    csv_lines = io_dic[outputfile]

    image_to_boxes = {}
    for line in csv_lines:
        items = line.split(',')
        if '1-on-10' in outputfile:
            image_index = int(os.path.basename(items[0].replace('.jpg', '').replace('img', '')))
            if image_index % 10 != 0:
                continue
        if items[0] in image_to_boxes:
            image_to_boxes[items[0]].append(items[1:7])
        else:
            image_to_boxes[items[0]] = [items[1:7]]
    ret = {'images': [], 'annotations': [], "categories": cat_info}
    for count, path in enumerate(sorted(image_to_boxes)):

        image_info = {'file_name': path,
                      'id': count,
                      'calib': ''}
        ret['images'].append(image_info)

        for ann_ind, box in enumerate(image_to_boxes[path]):

            x0, y0, x1, y1, label = int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[4]

            # cat_id = cat_ids[label.strip()]
            # print(label, cat_id)

            truncated = 0
            occluded = 0
            bbox = [float(x0), float(y0), float(x1), float(y1)]

            ann = {'image_id': count,
                   'id': int(len(ret['annotations']) + 1),
                   'category_id': 1,  # cat_id,
                   'bbox': _bbox_to_coco_bbox(bbox),
                   'truncated': truncated,
                   'occluded': occluded,
                   'iscrowd': 0,
                   'area': (bbox[3]-bbox[1])*(bbox[2]-bbox[0])}
            ret['annotations'].append(ann)

    print("File: ", outputfile)
    print("# images: ", len(ret['images']))
    print("# annotations: ", len(ret['annotations']))
    # import pdb; pdb.set_trace()
    json.dump(ret, open(outputfile, 'w'))
