import os
import cv2
import numpy as np
# from tqdm import tqdm

import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import sys
# import csv
import pandas as pd
from torch.utils.data import Dataset  # , DataLoader


DIEZ = "##########"
EXTENTION_PNG = ".png"
EXTENTION_JPG = ".jpg"


class ImageLoader(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file_path, image_directory, mask_directory, dataset_size, image_size=512, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file_path).head(dataset_size)
        self.transform = transform
        self.image_directory = image_directory
        self.mask_directory = mask_directory
        self.IMG_SIZE = image_size

    def __len__(self):
        return len(self.data_frame)

    def random_transform(self, image, mask):
        rows, cols, _ = image.shape
        p = 0.5
        max_angle = 45
        random_angle = np.random.random() * max_angle
        RotationMatrix = cv2.getRotationMatrix2D(
            ((cols-1)/2.0, (rows-1)/2.0), random_angle, 1)
        if np.random.random() < p:
            image, mask = image[::-1, :, :], mask[::-1, :]
        if np.random.random() < p:
            image, mask = image[:, ::-1, :], mask[:, ::-1]
        if np.random.random() < p:
            image, mask = cv2.warpAffine(image, RotationMatrix, (cols, rows)), cv2.warpAffine(
                mask, RotationMatrix, (cols, rows))
        return image, mask

    def __getitem__(self, idx):
        img_name = os.path.join(
            self.image_directory,
            self.data_frame["img"].iloc[idx] + EXTENTION_JPG)
        image = cv2.imread(img_name, cv2.IMREAD_COLOR)
        assert (image is not None), "This image is None: image name: {}".format(img_name)
        # image = cv2.resize(image, (self.IMG_SIZE, self.IMG_SIZE))

        mask_name = os.path.join(
            self.mask_directory,
            self.data_frame["img"].iloc[idx] + EXTENTION_PNG)
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
        assert (mask is not None), "This image is None: image name: {}".format(mask_name)
        # mask = cv2.resize(mask, (self.IMG_SIZE, self.IMG_SIZE))

        image, mask = self.random_transform(image, mask)

        sample = {'image': np.array(image), 'mask': np.array(mask)}

        if self.transform:
            sample = self.transform(sample)

        return (sample['image'], sample['mask'])


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        """
        swap color axis because
        numpy image: H x W x C
        torch image: C X H X W
        """
        image, mask = sample['image'], sample['mask']

        if len(image.shape) == 3:
            image = image.transpose((2, 0, 1))
        elif len(image.shape) == 2:
            image = np.reshape(image, (1, image.shape[0], image.shape[1]))

        if len(mask.shape) == 3:
            mask = mask.transpose((2, 0, 1))
        elif len(mask.shape) == 2:
            mask = np.reshape(mask, (1, mask.shape[0], mask.shape[1]))

        return {'image': torch.from_numpy(image).float(),
                'mask': torch.from_numpy(mask).float()}


class Normalize(object):
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        # landmarks = landmarks.transpose((2, 0, 1))
        return {'image': (image/255),
                'mask': (mask/255)}
