import os
import time
from typing import Optional

import requests
from pycocotools.coco import COCO


class COCOData:
    def __init__(self, anno_path: str, save_dir: str, categories: list[str]):
        self.coco = COCO(anno_path)
        self.image_save_dir = os.path.join(f"{save_dir}", "images")
        self.label_save_dir = os.path.join(f"{save_dir}", "labels")
        self.categories = categories

    def _fetch_cat_data(self) -> list[int]:
        catIds = self.coco.getCatIds(catNms=self.categories)
        # Get the corresponding image ids and images using loadImgs
        return self.coco.getImgIds(catIds=catIds)
        # self.images = self.coco.loadImgs(self.imgIds)

    def _anno_exists(self, img_id: int) -> bool:
        try:
            anno = self.coco.loadAnns(img_id)
            print(f"{img_id} loaded")
            return True
        except KeyError:
            print(f"{img_id} annotation not existed")
            return False

    def _save_image_data(self, img) -> None:
        img_data = requests.get(img[0]["coco_url"]).content
        filename = os.path.join(self.image_save_dir, f"{img[0]['id']}.jpg")
        with open(filename, "wb") as handler:
            handler.write(img_data)

    def _save_yolo_format(self, img) -> None:
        img_id = img[0]["id"]
        img_w = img[0]["width"]
        img_h = img[0]["height"]
        img_ann = self.coco.loadAnns(img_id)
        with open(f"{self.label_save_dir}/{img_id}.txt", "a") as handler:
            for ann in img_ann:
                current_category = 0
                current_bbox = ann["bbox"]
                x = current_bbox[0]
                y = current_bbox[1]
                w = current_bbox[2]
                h = current_bbox[3]

                x_centre = x + w / 2
                y_centre = y + h / 2

                # Normalize coordinates
                x_centre /= img_w
                y_centre /= img_h
                w /= img_w
                h /= img_h

                x_centre = format(x_centre, ".6f")
                y_centre = format(y_centre, ".6f")
                w = format(w, ".6f")
                h = format(h, ".6f")
                handler.write(f"{current_category} {x_centre} {y_centre} {w} {h}\n")
        return None

    def generate_img_anno(self, img_id) -> None:
        img = self.coco.loadImgs(img_id)

        self._save_image_data(img)
        self._save_yolo_format(img)
        return None

    def generate_data(self) -> None:
        img_ids = self._fetch_cat_data()
        img_ids = img_ids[0:20]
        count = 0
        for img_id in img_ids:
            if self._anno_exists(img_id):
                self.generate_img_anno(img_id)
                count += 1
        print(f"Full data: {len(img_ids)} images, Fetched {count} images")
        return None


def main():
    anno_path = "./annotations/instances_train2017.json"
    data_fetcher = COCOData(anno_path, "FullData", ["person"])
    data_fetcher.generate_data()


if __name__ == "__main__":
    main()
