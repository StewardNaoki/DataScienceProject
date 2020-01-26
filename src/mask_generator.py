import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import argparse

import utils

DATA_PATH = "./../DATA/"

LABEL_FILE_PATH = DATA_PATH +  "train_label.csv"
TRAIN_ID_FILE_NAME = DATA_PATH + "train_ids.csv"
MASK_FOLDER_PATH = DATA_PATH + "Images/train/masks/"
HEIGHT = 720
LENGTH = 1280


def generate_masks(num_mask = -1):
    df = pd.read_csv(LABEL_FILE_PATH)
    print(df.head())

    if num_mask == -1:
        N = len(df)
    else:
        N = num_mask
    for i in range(N):
        binary_mask =utils.rle_decode(df["rle_mask"].iloc[i], (HEIGHT, LENGTH))
        result = Image.fromarray((binary_mask * 255).astype(np.uint8))
        image_path = MASK_FOLDER_PATH +df["img"].iloc[i] +".jpg"
        print(image_path)
        result.save(image_path)

        if i >= 5:
            break



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_mask", type=int, default=-1,
                        help="Number of mask (default: -1)")


    args = parser.parse_args()

    generate_masks(args.num_mask)




if __name__ == "__main__":
    main()


