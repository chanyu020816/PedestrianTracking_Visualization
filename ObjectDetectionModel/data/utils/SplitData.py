import os
import random
import shutil

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

val_size = 10000
test_size = 5000
postfix = "jpg"
imgpath_COCO = "./COCO 2017/COCO Only Person/images"
labelpath_COCO = "./COCO 2017/COCO Only Person/labels"
imgpath_WilderPerson = "./WilderPerson/images"
labelpath_WilderPerson = "./WilderPerson/labels"


all_images = []
all_labels = []

for filename in tqdm(os.listdir(imgpath_COCO)):
    if filename.endswith(postfix):
        image_path = os.path.join(imgpath_COCO, filename)
        label_path = os.path.join(labelpath_COCO, filename[: -len(postfix)] + "txt")
        all_images.append(image_path)
        all_labels.append(label_path)

for filename in tqdm(os.listdir(imgpath_WilderPerson)):
    if filename.endswith(postfix):
        image_path = os.path.join(imgpath_WilderPerson, filename)
        label_path = os.path.join(
            labelpath_WilderPerson, filename[: -len(postfix)] + "txt"
        )
        all_images.append(image_path)
        all_labels.append(label_path)

combined_data = list(zip(all_images, all_labels))
random.shuffle(combined_data)
all_images, all_labels = zip(*combined_data)

train_images, val_test_images, train_labels, val_test_labels = train_test_split(
    all_images, all_labels, test_size=val_size + test_size, random_state=0
)
val_images, test_images, val_labels, test_labels = train_test_split(
    val_test_images, val_test_labels, test_size=test_size, random_state=0
)
print(f"Number of training data {len(train_images)}")
print(f"Number of validating data {len(val_images)}")
print(f"Number of testing data {len(test_images)}")


def copy_data(images, labels, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)
    dest_img = os.path.join(destination_folder, "images")
    dest_label = os.path.join(destination_folder, "labels")
    os.makedirs(dest_img, exist_ok=True)
    os.makedirs(dest_label, exist_ok=True)
    for img_path, label_path in tqdm(zip(images, labels)):
        shutil.copy(img_path, os.path.join(dest_img, os.path.basename(img_path)))
        shutil.copy(label_path, os.path.join(dest_label, os.path.basename(label_path)))


copy_data(train_images, train_labels, "./train")
copy_data(val_images, val_labels, "./val")
copy_data(test_images, test_labels, "./test")
