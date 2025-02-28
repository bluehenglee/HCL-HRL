import torch
import os
import numpy as np
from torch import nn
import torchvision
import torch.utils.data
from torchvision import datasets, transforms, models
import cv2
import json
from utils import *
from data_pre import one_json_to_numpy,json_to_numpy
from similarity import save_dismap
from concurrent.futures import ThreadPoolExecutor
import time

class Dataset_all(torch.utils.data.Dataset):
    # 初始化
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.img_name_list = sorted(os.listdir(os.path.join(dataset_path, 'imgs')))
        print(self.img_name_list)
        self.to_tensor = transforms.ToTensor()
        # Precompute paths to avoid repeated os.path.join calls
        self.img_paths = [os.path.join(dataset_path, 'imgs', img_name) for img_name in self.img_name_list]
        self.label_paths = [os.path.join(dataset_path, 'labels', img_name.split('.')[0]+'.json') for img_name in self.img_name_list]
        self.fore_paths = [os.path.join(dataset_path, 'SAM_fore', img_name) for img_name in self.img_name_list]
        self.back_paths = [os.path.join(dataset_path, 'SAM_back', img_name) for img_name in self.img_name_list]
        self.box_paths = [os.path.join(dataset_path, 'SAM_box', img_name) for img_name in self.img_name_list]
        self.point_paths = [os.path.join(dataset_path, 'point', img_name) for img_name in self.img_name_list]
        self.gt_paths = [os.path.join(dataset_path, 'gt', img_name) for img_name in self.img_name_list]

    # 根据 index 返回位置的图像和label
    def __getitem__(self, index):
        # Process image
        img_path = self.img_paths[index]
        img = cv2.imread(img_path)
        # 调整形状为 (3, 256, 256)
        img = cv2.resize(img, (256, 256))/255.0
        img = img.transpose((2, 0, 1))
        # img = self.to_tensor(img)

        box_path = self.box_paths[index]
        box = cv2.imread(box_path,0)/255.0
        box = np.expand_dims(box, axis=0)
        # box = self.to_tensor(box)

        fore_path = self.fore_paths[index]
        fore = cv2.imread(fore_path,0)/255.0
        fore = np.expand_dims(fore, axis=0)

        # fore = self.to_tensor(fore)

        back_path = self.back_paths[index]
        back = cv2.imread(back_path,0)/255.0
        back = np.expand_dims(back, axis=0)

        # back = self.to_tensor(back)

        gt_path = self.gt_paths[index]
        gt = cv2.imread(gt_path,0)/255.0
        gt = np.expand_dims(gt, axis=0)

        return img, box, fore, back, gt, self.gt_paths[index]


    def __len__(self):
        return len(self.img_name_list)
 