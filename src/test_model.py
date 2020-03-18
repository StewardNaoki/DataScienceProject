import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import time
import sys
import pandas as pd
import cv2
import numpy as np

import autoencoder as nw
import utils

width = 1280
height = 720

DATA_PATH = "./../DATA/"
MODEL_DIR = "model/"
BEST_MODELE = "best_model.pt"
run_dir = "run-1584468877/"
LOG_DIR = "./../log/"
IMAGE_FOLDER_PATH = DATA_PATH + "Images/train/images/"
IMG_SIZE = 512
EXTENTION_JPG = ".jpg"


def generate(num_run)
    run_dir = "run-"+str(num_run)+"/"
    model = nw.AutoEncoder(num_block=3, depth = 6)
    data_frame = pd.read_csv(DATA_PATH + "train_label.csv")
    print(data_frame.shape)
    model.load_state_dict(torch.load(LOG_DIR +run_dir + MODEL_DIR + BEST_MODELE))
    data_submission = pd.DataFrame(columns=['img', 'rle_mask'])
    for idx in range(data_frame.shape[0]):
        img_name = os.path.join(
                IMAGE_FOLDER_PATH, data_frame["img"].iloc[idx] + EXTENTION_JPG)
        image = cv2.imread(img_name, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        assert (image is not None), "This image is None: image name: {}".format(img_name)
        image = image.transpose((2, 0, 1))
        input = torch.from_numpy(image).float()
        input = input/255
        input = input.view(1,image.shape[0],image.shape[1],image.shape[2])
        output = (model(input)).view(image.shape[1],image.shape[2])
        # print(output.shape)
        output = (output >  0.5).float()
        output = output.detach().numpy() #.transpose((1, 2, 0))
        output = cv2.resize(output,(height, width))
        # cv2.imshow('image', output)
        # cv2.waitKey(0)
        # print(utils.rle_encode(output).size)
        code = ' '.join(str(e) for e in utils.rle_encode(output))
        data_submission.loc[idx] = [data_frame["img"].iloc[idx]] + [code]
        # assert(False)
    data_submission.to_csv(DATA_PATH + 'submission{}.csv'.format(num_run), index=False)
    # a = torch.tensor([[1,2],[1,4]])
    # a = a.numpy()
    # print(type(a))
    # a = np.array(a, dtype='uint8')
    # print(cv2.resize(a,(5,5)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_run", type=int, default=0,
                        help="Number of the run")
    args = parser.parse_args()
    generate(args.num_run)


if __name__ == "__main__":
    main()




