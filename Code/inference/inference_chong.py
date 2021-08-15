import sys
import torch
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import time
import os
import numpy as np
import json
import cv2
from PIL import Image, ImageOps
import random
from tqdm import tqdm
import operator
import itertools
from scipy.io import  loadmat
import logging
from scipy import signal
from utils import data_transforms
from tqdm import tqdm

from utils import preprocess_image
import argparse

from models.gazenet import GazeNet
from models.__init__ import save_checkpoint, resume_checkpoint
from dataloader.chong import GooDataset, GazeDataset
from torch.utils.data import Dataset, DataLoader

from models.chong import ModelSpatial
# from scipy.misc import imresize

def parse_inputs():

    # Command line arguments
    p = argparse.ArgumentParser()

    p.add_argument('--test_dir', type=str,
                        help='path to test set',
                        default=None)
    p.add_argument('--test_annotation', type=str,
                        help='test annotations (pickle/mat)',
                        default=None)
    p.add_argument('--resume_path', type=str,
                        help='load model file',
                        default=None)

    args = p.parse_args()

    return args


def boxes2centers(normalized_boxes):
    center_x = (normalized_boxes[:, 0] + normalized_boxes[:, 2]) / 2
    center_y = (normalized_boxes[:, 1] + normalized_boxes[:, 3]) / 2
    center_x = np.expand_dims(center_x, axis=1)
    center_y = np.expand_dims(center_y, axis=1)
    normalized_centers = np.hstack((center_x, center_y))

    return normalized_centers

def select_nearest_bbox(gazepoint, gt_bboxes, gt_labels=None):
    '''
    In: Accepts gazepoint (2,) and bboxes (n_boxes, 4), normalized from [0,1]
    Out: Returns the bbox nearest to gazepoint.
    '''

    centers = boxes2centers(gt_bboxes)

    diff = centers - gazepoint
    l2dist = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
    min_idx = l2dist.argmin()

    if gt_labels is not None:
        predicted_class = gt_labels[min_idx]
    else:
        predicted_class = -1

    nearest_box = {
        'box': gt_bboxes[min_idx],
        'label': predicted_class,
        'index': min_idx
    }
    return nearest_box

def draw_results(image_path, eyepoint, gazepoint, idx, bboxes=None):
    # Convert to numpy arrays jic users passed lists
    eyepoint, gazepoint, gazebox = map(np.array, [eyepoint, gazepoint, bboxes])

    # Draw gazepoints and gt
    im = cv2.imread(image_path)
    image_height, image_width = im.shape[:2]
    x1, y1 = eyepoint
    x2, y2 = gazepoint
    x1, y1 = image_width * x1, y1 * image_height
    x2, y2 = image_width * x2, y2 * image_height
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    cv2.circle(im, (x1, y1), 5, [255, 255, 255], -1)
    cv2.circle(im, (x2, y2), 5, [255, 255, 255], -1)
    cv2.line(im, (x1, y1), (x2, y2), [255, 0, 0], 2)

    if bboxes is not None:
        assert (len(bboxes.shape) == 2), 'gazebox must be numpy array of shape (N,4)'
        assert (bboxes.shape[1] == 4), 'gazebox must be numpy array of shape (N,4)'

        # Select nearest bbox given the gazepoint
        bbox_data = select_nearest_bbox(gazepoint, bboxes, gt_labels=None)
        nearest_bbox = bbox_data['box']

        # Scale to image size
        nearest_bbox = nearest_bbox * [image_width, image_height, image_width, image_height]
        nearest_bbox = nearest_bbox.astype(int)

        # Draw bbox of prediction
        cv2.rectangle(im, (nearest_bbox[0], nearest_bbox[1]), (nearest_bbox[2], nearest_bbox[3]), (0, 165, 255), 2)

    img = im
    save_dir = './temp/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = 'inference_%s.png' % str(idx)
    save_path = save_dir + filename
    cv2.imwrite(save_path, img)

    return None



def test_on_image(net, image, head_channel, face):
    net.eval()
    heatmaps = []

    # data = preprocess_image(test_image_path, eye)
    # image, face_image, gaze_field, eye_position = data['image'], data['face_image'], data['gaze_field'], data['eye_position']
    # image, face_image, gaze_field, eye_position = map(lambda x: Variable(x.unsqueeze(0).cuda()), [image, face_image, gaze_field, eye_position])

    raw_hm, _, inout  = net(image, head_channel, face)
    # x=val_gaze_heatmap_pred.squeeze(0).cpu().detach().numpy().squeeze(0)
    # print (output)

    raw_hm = raw_hm.cpu().detach().numpy() * 255
    raw_hm = raw_hm.squeeze()
    inout = inout.cpu().detach().numpy()
    inout = 1 / (1 + np.exp(-inout))
    inout = (1 - inout) * 255
    width, height = (224,224)
    norm_map = cv2.resize(raw_hm, (height, width)) - inout
    # return norm_map
    # final_output = predict_heatmap.cpu().data.numpy()
    #
    # heatmap = norm_map.reshape([224 // 4, 224 // 4]
    heatmap = cv2.resize(norm_map, (224//4, 224//4))

    h_index, w_index = np.unravel_index(heatmap.argmax(), heatmap.shape)
    f_point = np.array([w_index / 56., h_index / 56.])
    #
    #
    return heatmap, f_point[0], f_point[1]


def main():
    args = parse_inputs()

    # Load Model
    # net = GazeNet()
    net = ModelSpatial()
    net.cuda()

    resume_path = args.resume_path
    net, optimizer, start_epoch = resume_checkpoint(net, None, resume_path)

    # Prepare dataloaders
    test_images_dir = args.test_dir
    test_pickle_path = args.test_annotation

    # For GOO
    val_set = GooDataset(test_images_dir, test_pickle_path, 'test', use_gtbox=True)
    # test_data_loader = DataLoader(val_set, batch_size=2,
    #                               shuffle=False, num_workers=8)
    test_only = True
    if not test_only:
        test_and_save(net, test_data_loader)

    # Get a random sample image from the dataset
    for i in range(10):
        idx = np.random.randint(100)

        val_item = val_set.__getitem__(idx)
        image_path = val_item[7]
        # image_path = '/media/primesh/F4D0EA80D0EA4906/PROJECTS/FYP/Gaze detection/Datasets/gooreal/general images from internet/3.jpg'
        print(image_path)
        # import sys
        # sys.exit()
        img = val_item[0].unsqueeze(0).cuda()
        head_channel = val_item[2].unsqueeze(0).cuda()
        face = val_item[1].unsqueeze(0).cuda()
        bboxes = val_item[8]
        eyes = val_item[3]

        heatmap, x, y = test_on_image(net, img, head_channel, face)
        draw_results(image_path, eyes, (x,y), idx)
        print ("DONE")

if __name__ == "__main__":
    main()