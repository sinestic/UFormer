import torch
import torch.nn as nn
import tensorboard
from timm.utils import NativeScaler
from torch.utils.tensorboard import SummaryWriter
import time

import torchmetrics as tm
import math
import numpy as np
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from glob import glob
import cv2


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img/255.
    return img

##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform

        gt_dir = 'groundtruth'
        input_dir = 'input'

        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))

        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_image_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x)       for x in noisy_files if is_image_file(x)]

        self.img_options=img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        clean = load_img(self.clean_filenames[tar_index])
        noisy = load_img(self.noisy_filenames[tar_index])



        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        #Crop Input and Target
        ps = 256
        H = clean.shape[0]
        W = clean.shape[1]
        # r = np.random.randint(0, H - ps) if not H-ps else 0
        # c = np.random.randint(0, W - ps) if not H-ps else 0
        if H-ps==0:
            r=0
            c=0
        else:
            r = np.random.randint(0, H - ps)
            c = np.random.randint(0, W - ps)

        transformed = self.target_transform(image=clean,mask=noisy)
        clean = transformed["image"]
        noisy = transformed['mask']
        # clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]

        return clean, noisy


##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform

        gt_dir = 'groundtruth'
        input_dir = 'input'

        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))


        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_image_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files if is_image_file(x)]


        self.tar_size = len(self.clean_filenames)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size


        clean = load_img(self.clean_filenames[tar_index])
        noisy = load_img(self.noisy_filenames[tar_index])

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]



        transformed = self.target_transform(image=clean,mask=noisy)
        clean = transformed["image"]
        noisy = transformed['mask']
        # clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)
        return clean, noisy




def get_training_data(rgb_dir):
    # assert os.path.exists(rgb_dir)
    transforms = A.Compose([A.Rotate(),A.Flip(),ToTensorV2()])

    return DataLoaderTrain(rgb_dir, target_transform=transforms)


def get_validation_data(rgb_dir):
    transforms = A.Compose([A.CenterCrop(256,256),ToTensorV2()])
    # assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, target_transform=transforms)