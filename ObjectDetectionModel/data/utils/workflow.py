from controller.apply_album_aug import apply_aug
from controller.get_album_bb import get_bboxes_list
import cv2
import os
import yaml
import time
from tqdm import tqdm


def run_pipeline(yaml_path):
    with open(yaml_path, 'r') as stream:
        CONSTANTS = yaml.safe_load(stream)

    imgs = os.listdir(CONSTANTS["inp_img_pth"])   
    aug_size = int(CONSTANTS['aug_size'])
    count = 0
    failed_images = []
    for img_file in tqdm(imgs[0:20]):  
        file_name = img_file.split('.jpg')[0]
        image = cv2.imread(os.path.join(CONSTANTS["inp_img_pth"], img_file))           
        lab_pth = os.path.join(CONSTANTS["inp_lab_pth"], file_name + '.txt')                                
        album_bboxes = get_bboxes_list(lab_pth, CONSTANTS['CLASSES'])
        if not (check_boxes(album_bboxes)):
            failed_images.append(file_name)
            print(f'{file_name} failed')
            continue
        count += 1
        for x in range(aug_size):
            aug_file_name = file_name + "_" + CONSTANTS["transformed_file_name"] + str(x)
            apply_aug(image, album_bboxes, CONSTANTS["out_lab_pth"],  CONSTANTS["out_img_pth"], aug_file_name, CONSTANTS['CLASSES'])
        time.sleep(2)
    print(count)
    
def check_boxes(bboxes):
    for x, y, w, h, _ in bboxes:
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        if x1 > 1 or x1 < 0 or x2 > 1 or x2 < 0 or y1 > 1 or y1 < 0 or y2 > 1 or y2 < 0:
            return False
    return True


if __name__ == "__main__":
    run_pipeline("./contants.yaml")