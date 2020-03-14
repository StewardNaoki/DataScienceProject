import random as rd
import dataloader as dl
import torchvision.transforms as tf

full_dataset = dl.ImageDATA(csv_file_path=LABEL_FILE_PATH,
                                image_directory=IMAGE_FOLDER_PATH,
                                mask_directory=MASK_FOLDER_PATH,
                                transform=data_transforms)

nb_items = len(full_dataset)
ratio = 0.1

for i in range(int(ratio * nb_items)):
    index = rd.randint(0, nb_items)
    item, mask = get_item(full_dataset, index)
