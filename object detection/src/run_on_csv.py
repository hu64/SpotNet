import sys
CENTERNET_PATH = '/store/dev/CenterNet/src/lib/'
sys.path.insert(0, CENTERNET_PATH)

from detectors.detector_factory import detector_factory
from opts import opts
import os
import cv2
import numpy as np
from PIL import Image

# class_name = ['__background__', 'bus', 'car', 'others', 'van']
class_name = ['__background__', 'object']

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

base_dir = 'YOUR_BASE_DIR'
exp_id = 'EXP_ID'
model_name = 'model_best.pth'
MODEL_PATH = os.path.join(base_dir, exp_id, model_name)
seg_dir = 'changedetection-raw'  # only relevant if you want segmentation masks
base_seg_dir = os.path.join(base_dir, exp_id, seg_dir)
TASK = 'ctdet'
# --seg_weight 1
# opt = opts().init('{} --load_model {} --arch hourglass --seg_weight 1 --dataset uadetrac1on10_b --gpu 1 --keep_res'.format(TASK, MODEL_PATH).split(' '))
opt = opts().init('{} --load_model {} --arch hourglass --seg_weight 1 --dataset uav --gpu 1 --keep_res'.format(TASK, MODEL_PATH).split(' '))
detector = detector_factory[opt.task](opt)

SPLIT = 'uav-test'

if SPLIT == 'test':
    source_lines = open('/store/datasets/UA-Detrac/test-tf-all.csv', 'r').readlines()
    target_file = open(os.path.join(base_dir, exp_id, 'ua-test.csv'), 'w')
elif SPLIT == 'train1on10':
    source_lines = open('/store/datasets/UA-Detrac/train-tf.csv', 'r').readlines() # + open('/store/datasets/UA-Detrac/val-tf-all.csv', 'r').readlines()
    target_file = open(os.path.join(base_dir, exp_id, 'ua-train1on10.csv'), 'w')
elif SPLIT == 'trainval':
    source_lines = open('/store/datasets/UA-Detrac/train-tf-all.csv', 'r').readlines() + open('/store/datasets/UA-Detrac/val-tf-all.csv', 'r').readlines()
    target_file = open(os.path.join(base_dir, exp_id, 'ua-trainval.csv'), 'w')
elif SPLIT == 'uav-test':
    source_lines = open('/store/datasets/UAV/val.csv', 'r').readlines()
    target_file = open(os.path.join(base_dir, exp_id, 'uav-test.csv'), 'w')
elif SPLIT == 'changedetection':
    source_lines = open('/store/datasets/changedetection/changedetection.csv', 'r').readlines()
    target_file = open(os.path.join(base_dir, exp_id, 'changedetection.csv'), 'w')

if not os.path.exists(base_seg_dir):
    os.mkdir(base_seg_dir)

images = [item.split(',')[0] for item in source_lines]
images = set(images)

n_images = len(images)
for count, img in enumerate(images):
    if count % 1000 == 0:
        print("Progress: %d%%   \r" % (100*(count/n_images)))
        sys.stdout.write("Progress: %d%%   \r" % (100*(count/n_images)))
        sys.stdout.flush()

    im = Image.open(img.strip())
    width, height = im.size

    runRet = detector.run(img.strip())
    ret = runRet['results']
    # for label in [1]:  # [1, 2, 3, 4]:
    boxes = []
    for det in ret[1]:
        box = [int(item) for item in det[:4]]
        if float(det[4]) > 0.15 and (box[2] - box[0]) * (box[2] - box[0]) < (width * height) / 2:
            boxes.append(box)
        det = [img.strip()] + box + [class_name[1]] + [det[4]]
        print(str(det)[1:-1].translate(str.maketrans('', '', '\' ')), file=target_file)

    """
    map = np.zeros((height, width))
    for box in boxes:
        map[box[1]:box[3], box[0]:box[2]] = 1
        # map[box[2]:box[0], box[3]:box[1]] = 1

    seg = runRet['seg']
    seg_path = os.path.join(base_seg_dir, os.path.dirname(img).split('/')[-2], os.path.basename(img))
    if not os.path.exists(os.path.dirname(seg_path)):
        os.mkdir(os.path.dirname(seg_path))

    seg = np.squeeze(seg, [0, 1])
    sheight, swidth = seg.shape
    hoffset = int((sheight-height)/2)
    woffset = int((swidth-width)/2)
    seg = seg[hoffset:-hoffset, woffset:-woffset]

    seg -= np.min(seg)
    seg /= np.max(seg)
    # seg = cv2.GaussianBlur(seg, (5, 5), 0)
    # seg = seg**8
    # seg = cv2.GaussianBlur(seg, (5, 5), 0)
    # seg -= np.min(seg)
    # seg /= np.max(seg)
    # seg[seg <= 0.67] = 0
    # seg[seg > 0.67] = 1
    # seg *= map
    seg = (seg*255).astype(np.uint8)
    cv2.imwrite(os.path.dirname(seg_path) + '/map_' + os.path.basename(seg_path).replace('.jpg', '.png').strip(), (map*255).astype(np.uint8))
    cv2.imwrite(seg_path.replace('.jpg', '.png').strip(), seg)
    # seg[seg < 240] = 0
    """






