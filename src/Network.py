import os
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import pandas as pd
import time
import sys
import csv

import log_writer as lw

# set to true to one once, then back to false unless you want to change something in your training data.
CREATE_CSV = True
DATA_PATH = "./../DATA/"
CSV_NAME = "train_label.csv"
LOG_DIR = "./../log/"
FC1 = "fc1/"
BEST_MODELE = "best_model.pt"
MODEL_PATH = LOG_DIR + FC1 + BEST_MODELE
LABEL_FILE_PATH = DATA_PATH + CSV_NAME
IMAGE_FOLDER_PATH = DATA_PATH + "Images/train/masks/"
MASK_FOLDER_PATH = DATA_PATH + "Images/train/images/"

# MODELE_LOG_FILE = LOG_DIR + "modele.log"
# MODELE_TIME = f"model-{int(time.time())}"
METRICS = "metrics/"
TENSORBOARD = "tensorboard/"
DIEZ = "##########"
EXTENTION_PNG = ".png"
EXTENTION_JPG = ".jpg"
# tensorboard_writer   = SummaryWriter(log_dir = LOG_DIR+TENSORBOARD)




# class CrossEntropyOneHot(object):


#     def __call__(self, sample):
#         _, labels = sample['Y'].max(dim=0)
#         # landmarks = landmarks.transpose((2, 0, 1))
#         return {'X_image': sample['X_image'],
#                 'Y': labels}


class CNN(nn.Module):
    def __init__(self, l2_reg):
        super(CNN, self).__init__()

        self.l2_reg = l2_reg

        self.conv1 = nn.Conv2d(
                in_channels=1,              # input height
                out_channels=32,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            )

        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)

        self.layer1 = nn.Sequential(        # input shape (1, 28, 28)
            self.conv1,                     # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.layer2 = nn.Sequential(        # input shape (16, 14, 14)
            self.conv2,                     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.layer3 = nn.Sequential(        # input shape (16, 14, 14)
            self.conv3,                     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )

        
        self.fc1 = nn.Linear(128*8*8, 512)  # fully connected layer, output 10 classes
        self.fc2 = nn.Linear(512, 2)        # fully connected layer, output 10 classes

    def penalty(self):
        return self.l2_reg * (self.conv1.weight.norm(2) + self.conv2.weight.norm(2) + self.conv3.weight.norm(2) + self.fc1.weight.norm(2) + self.fc2.weight.norm(2))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = self.fc1(x)
        output = self.fc2(x)
        # return output, x    # return x for visualization
        return F.softmax(output, dim = 1)



def train(model, loader, f_loss, optimizer, device):
    """
    Train a model for one epoch, iterating over the loader
    using the f_loss to compute the loss and the optimizer
    to update the parameters of the model.

    Arguments :

        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        optimizer -- A torch.optim.Optimzer object
        device    -- a torch.device class specifying the device
                     used for computation

    Returns :
    """

    # We enter train mode. This is useless for the linear model
    # but is important for layers such as dropout, batchnorm, ...
    model.train()

    N = 0
    tot_loss, correct = 0.0, 0.0
    # with tqdm(total=len(loader)) as pbar:
    for i, (inputs, targets) in enumerate(loader):
        # pbar.update(1)
        # pbar.set_description("Training step {}".format(i))
        # print("****", inputs.shape)
        inputs, targets = inputs.to(device), targets.to(device)
        # print("***",inputs.shape)

        # Compute the forward pass through the network up to the loss
        outputs = model(inputs)

        loss = f_loss(outputs, targets)
        # print("Loss: ", loss)
        N += inputs.shape[0]
        tot_loss += inputs.shape[0] * f_loss(outputs, targets).item()

        # print("Output: ", outputs)
        predicted_targets = outputs

        correct += (predicted_targets == targets).sum().item()

        optimizer.zero_grad()
        # model.zero_grad()
        loss.backward()
        # model.penalty().backward()
        optimizer.step()
    return tot_loss/N, correct/N


def test(model, loader, f_loss, device, final_test=False):
    """
    Test a model by iterating over the loader

    Arguments :

        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        device    -- The device to use for computation

    Returns :

        A tuple with the mean loss and mean accuracy

    """
    # We disable gradient computation which speeds up the computation
    # and reduces the memory usage
    with torch.no_grad():
        # We enter evaluation mode. This is useless for the linear model
        # but is important with layers such as dropout, batchnorm, ..
        model.eval()
        N = 0
        tot_loss, correct = 0.0, 0.0
        # with open(MODELE_LOG_FILE, "a") as f:
        #with tqdm(total=len(loader)) as pbar:
        for i, (inputs, targets) in enumerate(loader):
            # pbar.update(1)
            # pbar.set_description("Testing step {}".format(i))
            # We got a minibatch from the loader within inputs and targets
            # With a mini batch size of 128, we have the following shapes
            #    inputs is of shape (128, 1, 28, 28)
            #    targets is of shape (128)

            # We need to copy the data on the GPU if we use one
            inputs, targets = inputs.to(device), targets.to(device)

            # Compute the forward pass, i.e. the scores for each input image
            outputs = model(inputs)

            # We accumulate the exact number of processed samples
            N += inputs.shape[0]

            # We accumulate the loss considering
            # The multipliation by inputs.shape[0] is due to the fact
            # that our loss criterion is averaging over its samples
            tot_loss += inputs.shape[0] * f_loss(outputs, targets).item()

            # For the accuracy, we compute the labels for each input image
            # Be carefull, the model is outputing scores and not the probabilities
            # But given the softmax is not altering the rank of its input scores
            # we can compute the label by argmaxing directly the scores
            predicted_targets = outputs
            correct += (predicted_targets == targets).sum().item()

            if final_test:
                print("targets:\n", targets[0])
                print("predicted targets:\n", outputs[0])

    return tot_loss/N, correct/N


class ModelCheckpoint:

    def __init__(self, filepath, model):
        self.min_loss = None
        self.filepath = filepath
        self.model = model

    def update(self, loss):
        if (self.min_loss is None) or (loss < self.min_loss):
            print("Saving a better model to ", self.filepath)
            torch.save(self.model.state_dict(), self.filepath)
            #torch.save(self.model, self.filepath)
            self.min_loss = loss


def progress(loss, acc):
    print(' Training   : Loss : {:2.4f}, Acc : {:2.4f}\r'.format(loss, acc))
    sys.stdout.flush()



