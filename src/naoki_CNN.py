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
CSV_NAME = "train_labels.csv"
LOG_DIR = "./../log/"
FC1 = "fc1/"
BEST_MODELE = "best_model.pt"
MODEL_PATH = LOG_DIR + FC1 + BEST_MODELE
LABEL_FILE_PATH = DATA_PATH + "train_label.csv"
IMAGE_FOLDER_PATH = DATA_PATH + "Images/train/masks/"
MASK_FOLDER_PATH = DATA_PATH + "Images/train/images/"

# MODELE_LOG_FILE = LOG_DIR + "modele.log"
# MODELE_TIME = f"model-{int(time.time())}"
METRICS = "metrics/"
TENSORBOARD = "tensorboard/"
DIEZ = "##########"
EXTENTION = ".jpg"
# tensorboard_writer   = SummaryWriter(log_dir = LOG_DIR+TENSORBOARD)


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
        self.data_frame = pd.read_csv(csv_file_path)
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

        img_name = os.path.join(self.image_directory, self.data_frame["img"].iloc[idx] +EXTENTION)
        print("image name: ", img_name)
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

        if(image is None):
            print("This image is None: image name: ",img_name)
            assert (not image is None)
        image = cv2.resize(image, (self.IMG_SIZE, self.IMG_SIZE))


        mask_name = os.path.join(self.mask_directory, self.data_frame["img"].iloc[idx] +EXTENTION)
        print("mask name: ", mask_name)
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)

        if(X is None):
            print("This image is None: image name: ",mask_name)
            assert (not mask_name is None)
        mask = cv2.resize(mask, (self.IMG_SIZE, self.IMG_SIZE))


        sample = {'image': np.array(image), 'mask': np.array(mask)}

        if self.transform:
            sample = self.transform(sample)
        
        # return sample
        return (sample['image'], sample['mask'])


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, Y = sample['image'], sample['mask']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # print(type(image.shape))
        # print(len(image.shape))
        # print(Y)

        # print("image1: ",image)
        # print(image.shape)

        
        if len(image.shape) == 3:
            print(image.shape)
            image = image.transpose((2, 0, 1))
        elif len(image.shape) == 2:
            image = np.array([image])


        return {'image': torch.from_numpy(image).float(),
                'mask': torch.from_numpy(Y).float()}


class Normalize(object):

    def __call__(self, sample):
        image = sample['image']

        # landmarks = landmarks.transpose((2, 0, 1))
        return {'image': (image/255) -0.5,
                'mask': sample['mask']}

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

        self.layer1 = nn.Sequential(         # input shape (1, 28, 28)
            self.conv1,                     # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.layer2 = nn.Sequential(         # input shape (16, 14, 14)
            self.conv2,                     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.layer3 = nn.Sequential(         # input shape (16, 14, 14)
            self.conv3,                     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )

        
        self.fc1 = nn.Linear(128*8*8, 512)   # fully connected layer, output 10 classes
        self.fc2 = nn.Linear(512, 2)   # fully connected layer, output 10 classes

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
    with tqdm(total=len(loader)) as pbar:
        for i, (inputs, targets) in enumerate(loader):
            pbar.update(1)
            pbar.set_description("Training step {}".format(i))
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
        with tqdm(total=len(loader)) as pbar:
            for i, (inputs, targets) in enumerate(loader):
                pbar.update(1)
                pbar.set_description("Testing step {}".format(i))
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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=1,
                        help="number of epoch (default: 1)")
    parser.add_argument("--batch", type=int, default=100,
                        help="number of batch (default: 100)")
    parser.add_argument("--valpct", type=float, default=0.2,
                        help="proportion of test data (default: 0.2)")
    parser.add_argument("--num_threads", type=int, default=1,
                        help="number of thread used (default: 1)")
    parser.add_argument("--create_csv", type=bool, default=False,
                        help="create or not csv file (default: False)")
    parser.add_argument("--l2_reg", type=int, default=0.001,
                        help="L2 regularisation (default: 0.001)")

    # parser.add_argument("--num_var", type=int, default=5,
    #                     help="Number of variables (default: 5)")
    # parser.add_argument("--num_const", type=int, default=4,
    #                     help="number of constrains (default: 4)")
    # parser.add_argument("--num_prob", type=int, default=10,
    #                     help="number of problems to generate (default: 10)")

    args = parser.parse_args()

    # if args.create_csv:
    #     g_csv.generate_csv(DATA_PATH + CSV_NAME, args.num_var,
    #                        args.num_const, args.num_prob)

    valid_ratio = args.valpct  # Going to use 80%/20% split for train/valid

    data_transforms = transforms.Compose([
        ToTensor()
    ])

    #TODO
    full_dataset = ImageDATA(csv_file_path = LABEL_FILE_PATH,
                            image_directory = IMAGE_FOLDER_PATH,
                            mask_directory = MASK_FOLDER_PATH ,
                            transform=data_transforms)

    nb_train = int((1.0 - valid_ratio) * len(full_dataset))
    # nb_test = int(valid_ratio * len(full_dataset))
    nb_test = len(full_dataset) - nb_train
    print("Size of full data set: ", len(full_dataset))
    print("Size of training data: ", nb_train)
    print("Size of testing data: ", nb_test)
    train_dataset, test_dataset = torch.utils.data.dataset.random_split(
        full_dataset, [nb_train, nb_test])

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch,
                              shuffle=True,
                              num_workers=args.num_threads)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch,
                             shuffle=True,
                             num_workers=args.num_threads)

    for (inputs, targets) in train_loader:
        print("input:\n",inputs)
        print("target:\n", targets)

    # #TODO params
    # num_param = args.num_var + args.num_const + (args.num_var*args.num_const)
    # model = FullyConnectedRegularized(
    #     l2_reg=args.l2_reg, num_param=num_param, num_var=args.num_var)
    # print("Network architechture:\n", model)

    # use_gpu = torch.cuda.is_available()
    # if use_gpu:
    #     device = torch.device('cuda')
    # else:
    #     device = torch.device('cpu')

    # model.to(device)

    # # f_loss = torch.nn.CrossEntropyLoss() #TODO
    # f_loss = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters())

    # top_logdir = LOG_DIR + FC1
    # if not os.path.exists(top_logdir):
    #     os.mkdir(top_logdir)
    # model_checkpoint = ModelCheckpoint(top_logdir + BEST_MODELE, model)

    # log_file_path = lw.generate_unique_logpath(LOG_DIR, "Linear")

    # for t in tqdm(range(args.epoch)):
    #         # pbar.set_description("Epoch Number{}".format(t))
    #         print(DIEZ + "Epoch Number: {}".format(t) + DIEZ)
    #         train_loss, train_acc = train(
    #             model, train_loader, f_loss, optimizer, device)

    #         progress(train_loss, train_acc)
    #         time.sleep(0.5)

    #         val_loss, val_acc = test(model, test_loader, f_loss, device)
    #         print(" Validation : Loss : {:.4f}, Acc : {:.4f}".format(
    #             val_loss, val_acc))

    #         model_checkpoint.update(val_loss)

    #         lw.write_log(log_file_path, val_acc, val_loss, train_acc, train_loss)

    #         tensorboard_writer.add_scalar(METRICS + 'train_loss', train_loss, t)
    #         tensorboard_writer.add_scalar(METRICS + 'train_acc',  train_acc, t)
    #         tensorboard_writer.add_scalar(METRICS + 'val_loss', val_loss, t)
    #         tensorboard_writer.add_scalar(METRICS + 'val_acc',  val_acc, t)

    # model.load_state_dict(torch.load(MODEL_PATH))
    # print(DIEZ+" Final Test "+DIEZ)
    # test_loss, test_acc = test(
    #     model, test_loader, f_loss, device, final_test=True)
    # print(" Test       : Loss : {:.4f}, Acc : {:.4f}".format(
    #     test_loss, test_acc))

    # lw.create_acc_loss_graph(log_file_path)


if __name__ == "__main__":
    main()
