import os
import shutil

import cv2
from tqdm import tqdm

ORI_IMAGE_PATH = "./ori_images"
ANNO_PATH = "./annotations"
IMAGE_PATH = "./images"
LABEL_PATH = "./labels"


def format_convert(file_name):
    ori_img_path = os.path.join(ORI_IMAGE_PATH, file_name)
    anno_path = os.path.join(ANNO_PATH, f"{file_name}.txt")
    img_path = os.path.join(IMAGE_PATH, file_name)
    label_path = os.path.join(LABEL_PATH, f'{file_name.split(".")[0]}.txt')
    img = cv2.imread(ori_img_path)
    h, w, _ = img.shape
    try:
        with open(anno_path, "r") as fl:
            lines = fl.readlines()
    except FileNotFoundError:
        return
    shutil.copy(ori_img_path, img_path)
    with open(label_path, "w") as f:
        for line in lines[1:]:
            class_idx, x1, y1, x2, y2 = map(float, line.split())
            if class_idx != 1:
                continue
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1

            normalized_x_center = x_center / w
            normalized_y_center = y_center / h
            normalized_width = width / w
            normalized_height = height / h

            f.write(
                f"0 {normalized_x_center} {normalized_y_center} {normalized_width} {normalized_height}\n"
            )


def main():
    files = os.listdir(ORI_IMAGE_PATH)
    for f in tqdm(files):
        if f.endswith(".jpg"):
            format_convert(f)


if __name__ == "__main__":
    main()
