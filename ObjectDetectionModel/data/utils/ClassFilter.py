import os
from pycocotools.coco import COCO
from tqdm import tqdm
from shutil import copyfile
import time

classes = ['person']
coco = COCO('../COCO 2017/person_keypoints_val2017.json')
classes_ids = coco.getCatIds(catNms=classes)
classes_ids = list(map(lambda x: x - 1, classes_ids))
del coco
train_image_path = "../COCO 2017/images/train2017"
train_label_path = "../COCO 2017/labels/train2017"
target_image_path = "../COCO 2017/COCO Only Person/images"
target_label_path = "../COCO 2017/COCO Only Person/labels"

def classfilter():
    
    assert os.path.exists(train_image_path)
    assert os.path.exists(train_label_path)
    
    indexes = os.listdir(train_image_path)
    rest = 500
    file_count = 0
    for k, index in enumerate(tqdm(indexes)):
        txtFile = f'{index[:index.rfind(".")]}.txt'
        
        label_file_path = os.path.join(train_label_path, txtFile)
        if not os.path.exists(label_file_path):
            continue
        
        target_labels = []
        with open(label_file_path, 'r') as fr:
            label_content = fr.readlines()
            target_labels = [label for label in label_content if int(label.split()[0]) in classes_ids]
        if not target_labels:
            continue
        
        with open(os.path.join(target_label_path, txtFile), 'w') as fw:
            for label in target_labels:
                fw.write(label)
                
        src_image_path = os.path.join(train_image_path, index)
        dest_image_path = os.path.join(target_image_path, index)
        copyfile(src_image_path, dest_image_path)
        file_count += 1
        k += 1
        if k % rest == 0: time.sleep(5)
    print(f'Total {file_count} files')
    
if __name__ == "__main__":
    classfilter()
    