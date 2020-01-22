import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import utils

LABEL_FILE_NAME = "../DATA/train_label.csv"
TRAIN_ID_FILE_NAME = "../DATA/train_ids.csv"
HEIGHT = 720
LENGTH = 1280

# HEIGHT = 2
# LENGTH = 2

df = pd.read_csv(LABEL_FILE_NAME)
df2 = pd.read_csv(TRAIN_ID_FILE_NAME)

print(df.head)
print(df2.head)

# print(df["rle_mask"].iloc[0])

binary_image =utils.rle_decode(df["rle_mask"].iloc[0], (HEIGHT, LENGTH) )

# plt.imshow(binary_image)

# plt.show()

result = Image.fromarray((binary_image * 255).astype(np.uint8))
result.save('out.jpg')

# results = list(map(int, df["rle_mask"].iloc[0].split(" ")))
# # results = [1, 0, 0,1]
# # print(results)

# bool_black = True
# # binary_image = [0]*(LENGTH*HEIGHT)
# k = 0
# # binary_image = np.array([])
# binary_image = []
# for x in results:
#     print(k)
#     k+=1
#     if bool_black:
#         bool_black = False
#         binary_image += [0]*x
#         # for i in range(x):
#         #     binary_image.append()
#     else:
#         bool_black = True
#         binary_image += [1]*x

# binary_image = np.asarray(binary_image)
# print("Resize")
# binary_image.resize(HEIGHT, LENGTH)
# print("SHOW")
# plt.imshow(binary_image)

# plt.show()
# print(binary_image)
