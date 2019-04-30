"""Example
python script/experiment/infer_images_example.py \
--model_weight_file YOUR_MODEL_WEIGHT_FILE
"""
from __future__ import print_function

import sys
sys.path.insert(0, '.')

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import numpy as np
import argparse
import cv2
from PIL import Image
import os.path as osp
import cv2
import math
from tqdm import tqdm
import time
import scipy.io as sci

from tri_loss.model.Model import Model
from tri_loss.utils.utils import load_state_dict
from tri_loss.utils.utils import set_devices
from tri_loss.utils.dataset_utils import get_im_names
from tri_loss.utils.distance import normalize


class Config(object):
    def __init__(self):

        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--sys_device_ids', type=eval, default=(0,))
        parser.add_argument('--resize_h_w', type=eval, default=(256, 128))
        parser.add_argument('--ckpt_file', type=str, default='')
        parser.add_argument('--model_weight_file', type=str, default='')
        parser.add_argument('--saved_feature_mat_path', type=str, default='')
        parser.add_argument('--image_dir', type=str, default='')
        parser.add_argument('--net', type=str, default='mobilenetV2',
                            choices=['resnet50', 'shuffelnetV2', 'mobilenetV2'])
        parser.add_argument('--bbox_mat', type=str, default='')
        parser.add_argument('--batch_size', type=eval, default=(1))

        args = parser.parse_args()

        # gpu ids
        self.sys_device_ids = args.sys_device_ids
        # Image Processing
        self.resize_h_w = args.resize_h_w
        self.im_mean = [0.486, 0.459, 0.408]
        self.im_std = [0.229, 0.224, 0.225]

        # This contains both model weight and optimizer state
        self.ckpt_file = args.ckpt_file
        # This only contains model weight
        self.model_weight_file = args.model_weight_file

        self.saved_feature_mat_path = args.saved_feature_mat_path

        self.image_dir = args.image_dir

        self.net = args.net

        self.bbox_mat = args.bbox_mat

        self.batch_size = args.batch_size

class BoundingBoxCrops(Dataset):
    def pre_process_im(self, im, cfg):
        """Pre-process image.
        `im` is a numpy array with shape [H, W, 3], e.g. the result of
        matplotlib.pyplot.imread(some_im_path), or
        numpy.asarray(PIL.Image.open(some_im_path))."""

        # Resize.
        im = cv2.resize(im, cfg.resize_h_w[::-1], interpolation=cv2.INTER_LINEAR)
        # scaled by 1/255.
        im = im / 255.

        # Subtract mean and scaled by std
        im = im - np.array(cfg.im_mean)
        im = im / np.array(cfg.im_std).astype(float)

        # shape [H, W, 3] -> [3, H, W]
        im = im.transpose(2, 0, 1)
        return im
    def __init__(self, cfg):

        self.cfg = cfg

        self.im_dir = osp.expanduser(cfg.image_dir)

        print('Reading bounding box detections from ' + self.cfg.bbox_mat)
        print('Feature maps will be saved to ' + self.cfg.saved_feature_mat_path)

        self.bbox_dict = sci.loadmat(self.cfg.bbox_mat)
        self.bboxes = self.bbox_dict['bboxes']

        self.num_detections = self.bboxes.shape[0]

    def __getitem__(self, det):
        # Get the bounding box info for the detection
        frame = self.bboxes[det,0]
        x1 = self.bboxes[det,1]
        y1 = self.bboxes[det,2]
        x2 = self.bboxes[det,3]
        y2 = self.bboxes[det,4]

        # Find the corresponding image
        img_name = "{:06d}".format(int(frame)) + '.jpg'
        im_path = self.im_dir + "/" + img_name

        # Read image and convert to RGB
        bgr_img = cv2.imread(im_path)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        # Crop the original image to detection bbox
        height, width = rgb_img.shape[:2]
        scaled_x1 = int(math.floor(x1*width))
        scaled_x2 = int(math.floor(x2*width))
        scaled_y1 = int(math.floor(y1*height))
        scaled_y2 = int(math.floor(y2*height))

        scaled_x1 = min(width, max(scaled_x1,0))
        scaled_x2 = min(width, max(scaled_x2,0))
        scaled_y1 = min(height, max(scaled_y1,0))
        scaled_y2 = min(height, max(scaled_y2,0))

        # Only run the ReID when the bounding box has a nonzero area
        area = (scaled_x2-scaled_x1)*(scaled_y2-scaled_y1)
        if area > 0:
            crop_img = rgb_img[scaled_y1:scaled_y2, scaled_x1:scaled_x2]
            im = self.pre_process_im(crop_img, self.cfg)
            self.bad_idx = False
        else:
            crop_img = np.zeros((256,128,3))
            im = self.pre_process_im(crop_img, self.cfg)
            self.bad_idx = True
        self.img = torch.from_numpy(im).float()

        return self.img, self.bad_idx

    def __len__(self):
        return self.num_detections

def main():
    cfg = Config()

    TVT, TMO = set_devices(cfg.sys_device_ids)

    #########
    # Model #
    #########

    model = Model(net=cfg.net, pretrained=False)

    #####################
    # Load Model Weight #
    #####################

    used_file = cfg.model_weight_file or cfg.ckpt_file
    loaded = torch.load(used_file, map_location=(lambda storage, loc: storage))
    if cfg.model_weight_file == '':
        loaded = loaded['state_dicts'][0]
    load_state_dict(model, loaded)
    print('Loaded model weights from {}'.format(used_file))

    model = model.base
    # Set eval mode. Force all BN layers to use global mean and variance, also disable dropout.
    model.eval()
    # Transfer Model to Specified Device.
    TMO([model])

    #######################
    # Extract Image Crops #
    #######################

    all_feat = []

    dataset = BoundingBoxCrops(cfg)
    loader = DataLoader(dataset=dataset,
                        batch_size=cfg.batch_size,
                        shuffle=False,
                        num_workers=min(cfg.batch_size,16))

    print('Processing detections')
    for i, data in tqdm(enumerate(loader, 0)):
        imgs, bad_boxes = data

        imgs = Variable(TVT(imgs))
        with torch.no_grad():
            feats = model(imgs)
        feats = feats.data.cpu().numpy()
        for j in range(len(bad_boxes)):
            if bad_boxes[j] == True:
                feats[j,:] = np.zeros(1280)
        sci.savemat(cfg.saved_feature_mat_path+'/'+"{:05d}".format(i)+'.mat', {'features':feats})

if __name__ == '__main__':
    main()
