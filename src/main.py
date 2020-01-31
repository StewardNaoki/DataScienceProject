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
import time

import log_writer as lw
import Network as nw
import dataloader as dl

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
    parser.add_argument("--log", default=False, action='store_true',
                        help="Write log or not (default: False)")
    parser.add_argument("--l2_reg", type=int, default=0.001,
                        help="L2 regularisation (default: 0.001)")

    args = parser.parse_args()

    # if args.create_csv:
    #     g_csv.generate_csv(DATA_PATH + CSV_NAME, args.num_var,
    #                        args.num_const, args.num_prob)

    valid_ratio = args.valpct  # Going to use 80%/20% split for train/valid

    data_transforms = transforms.Compose([
        dl.ToTensor(), dl.Normalize()
    ])

    # TODO
    full_dataset = dl.ImageDATA(csv_file_path = LABEL_FILE_PATH,
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

    i = 0
    for (inputs, targets) in train_loader:
        if i > 10:
            break
        i += 1
        print("input:\n", inputs)
        print("target:\n", targets)


    

    # #TODO params
    # num_param = args.num_var + args.num_const + (args.num_var*args.num_const)
    # model = FullyConnectedRegularized(
    #     l2_reg=args.l2_reg, num_param=num_param, num_var=args.num_var)
    # print("Network architechture:\n", model)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.to(device)

    # # f_loss = torch.nn.CrossEntropyLoss() #TODO
    # f_loss = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters())

    # top_logdir = LOG_DIR + FC1
    # if not os.path.exists(top_logdir):
    #     os.mkdir(top_logdir)
    # model_checkpoint = ModelCheckpoint(top_logdir + BEST_MODELE, model)

    # if args.log:
    #     print("Writing log")
    #     #generate unique folder for new run
    #     run_dir_path, num_run = lw.generate_unique_run_dir(LOG_DIR,"run")

    #     tensorboard_writer = SummaryWriter(
    #         log_dir=run_dir_path, filename_suffix=".log")

    #     #write short description of the run
    #     run_desc = "Epoch{}Reg{}Var{}Const{}CLoss{}Dlayer{}Alpha{}".format(
    #         args.num_epoch, args.l2_reg, args.num_var, args.num_const, args.custom_loss, args.num_deep_layer, args.alpha)
    #     log_file_path =  LOG_DIR + "Run{}".format(num_run) + run_desc + ".log"

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