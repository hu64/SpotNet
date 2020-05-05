# from: https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html
import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import os


def get_pairs_from_list(list_names=['/store/datasets/UAV/csv.csv']):

    images = set()
    for list_name in list_names:
        with open(list_name, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')

            for row in reader:
                was_first = False
                img1_path = row[0]
                index = (int(img1_path[-10]
                                 + img1_path[-9]
                                 + img1_path[-8]
                                 + img1_path[-7]
                                 + img1_path[-6]
                                 + img1_path[-5])
                             -1) % 1000000
                if index == 0:
                    index = 2
                    was_first = True
                index = str(index).zfill(5)

                img0_path = img1_path[:-9] + index + '.jpg'
                if was_first:
                    images.add((img1_path, img0_path, True))
                else:
                    images.add((img0_path, img1_path, False))
    return images


def normalize_2d_between_range(vector, min_range=0, max_range=255):

    max_vec = np.max(vector)
    min_vec = np.min(vector)
    vector = ((((vector - min_vec) / (max_vec - min_vec) - 0.5) * max_range) + (max_range / 2.0)) + min_range
    return vector


def dense_optical_flow(path0, path1):

    img0 = cv2.imread(path0)
    img1 = cv2.imread(path1)

    hsv = np.zeros_like(img1)
    hsv[..., 1] = 255

    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(img0, img1, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    
    return bgr


# from: https://stackoverflow.com/questions/42798659/how-to-remove-small-connected-objects-using-opencv
def remove_small_blobs(im, min_size=1000):

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(im, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # your answer image
    im_bw = np.zeros(output.shape)
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            im_bw[output == i + 1] = 255
    return im_bw


# from: https://stackoverflow.com/questions/10316057/filling-holes-inside-a-binary-object
def contour_filling(im, kernel_size=(7, 7)):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    return cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)


def binarize_flow(img):

    # Otsu's thresholding
    # ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    (thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    img = remove_small_blobs(img)
    img = contour_filling(img, kernel_size=(9, 9))

    return img


pairs = get_pairs_from_list()
pairs = sorted(pairs)
total = len(pairs)
current = 0
for pair in pairs:
    current += 1
    if current % 100 == 0:
        print(str(current) + ' / ' + str(total))

    # flow_name = os.path.join(os.path.dirname(pair[1]), 'flow_b', os.path.split(pair[1])[1])
    if pair[2]:
        flow_name = os.path.join(os.path.dirname(pair[0]), 'flow', os.path.split(pair[0])[1])
    else:
        flow_name = os.path.join(os.path.dirname(pair[1]), 'flow', os.path.split(pair[1])[1])

    if not os.path.exists(flow_name):
        # flow = cv2.imread(flow0_name)
        # flow = cv2.cvtColor(flow, cv2.COLOR_BGR2GRAY)
        flow = dense_optical_flow(pair[0], pair[1])
        # flow = binarize_flow(flow)

        if not os.path.exists(os.path.dirname(flow_name)):
            os.mkdir(os.path.dirname(flow_name))

        cv2.imwrite(flow_name, flow)


