import os
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import csv
import pandas as pd
from torch.utils.data import Dataset, DataLoader


DIEZ = "##########"
EXTENTION_PNG = ".png"
EXTENTION_JPG = ".jpg"

class ImageDATA(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file_path, image_directory, mask_directory, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file_path).head(10)
        self.transform = transform
        self.image_directory = image_directory
        self.mask_directory = mask_directory
        self.IMG_SIZE = 64

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        # print(self.data_frame.head())

        img_name = os.path.join(self.image_directory, self.data_frame["img"].iloc[idx] + EXTENTION_JPG)
        print("image name: ", img_name)
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

        if(image is None):
            print("This image is None: image name: ",img_name)
            assert (not image is None)
        # cv2.imshow("Image", image)
        # cv2.waitKey(0)
        image = cv2.resize(image, (self.IMG_SIZE, self.IMG_SIZE))

        mask_name = os.path.join(self.mask_directory, self.data_frame["img"].iloc[idx] + EXTENTION_PNG)
        print("mask name: ", mask_name)
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
        # cv2.imshow("Mask", mask)
        # cv2.waitKey(0)

        if(mask is None):
            print("This image is None: image name: ",mask_name)
            assert (not mask is None)
        mask = cv2.resize(mask, (self.IMG_SIZE, self.IMG_SIZE))

        sample = {'image': np.array(image), 'mask': np.array(mask)}

        if self.transform:
            sample = self.transform(sample)

        # return sample
        return (sample['image'], sample['mask'])


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        if len(image.shape) == 3:
            print(image.shape)
            image = image.transpose((2, 0, 1))
            mask = mask.transpose((2, 0, 1))
        elif len(image.shape) == 2:
            mask = np.reshape(mask, (1, mask.shape[0], mask.shape[1]))
            image = np.reshape(image, (1, image.shape[0], image.shape[1]))

        return {'image': torch.from_numpy(image).float(),
                'mask': torch.from_numpy(mask).float()}


class Normalize(object):

    def __call__(self, sample):
        image = sample['image']

        # landmarks = landmarks.transpose((2, 0, 1))
        return {'image': (image/255) -0.5,
                'mask': sample['mask']}