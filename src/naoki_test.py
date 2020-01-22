import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

import utils

LABEL_FILE_NAME = "../DATA/train_label.csv"
TRAIN_ID_FILE_NAME = "../DATA/train_ids.csv"
MASK_FOLDER_PATH = "../DATA/Images/train/masks/"
HEIGHT = 720
LENGTH = 1280

# HEIGHT = 2
# LENGTH = 2

df = pd.read_csv(LABEL_FILE_NAME)
# df2 = pd.read_csv(TRAIN_ID_FILE_NAME)

print(df.head())
# print(df2.head)

# print(df["rle_mask"].iloc[0])

# binary_image =utils.rle_decode(df["rle_mask"].iloc[0], (HEIGHT, LENGTH) )





# print(len(df))
for i in range(len(df)):
    binary_mask =utils.rle_decode(df["rle_mask"].iloc[i], (HEIGHT, LENGTH))

    # plt.imshow(binary_mask)
    # plt.show()

    result = Image.fromarray((binary_mask * 255).astype(np.uint8))
    image_path = MASK_FOLDER_PATH +df["img"].iloc[i] +".jpg"
    print(image_path)
    result.save(image_path)

    if i >= 5:
        break


