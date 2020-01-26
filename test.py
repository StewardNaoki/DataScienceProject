import matplotlib.pyplot as plt 
import pandas as pd
import glob

source_folder = "Data/"
files = glob.glob(source_folder + *.csv")
for f in files:
    print("\n" + f)
    tests = pd.read_csv(f)
    print(tests.dtypes)

img_folder = "supplementary/test/images/"
img_files = glob.glob(source_folder + img_folder + "*.jpg")
nb_img = len(images)
images = np.zeros((nb_img, 720, 1280))
for f in img_files:
    img = plt.imread(f)
    images[]
    print(img)

