import cv2
import numpy as np


def draw_bounding_boxes(image_path, label_path, output_path=None):
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image from {image_path}")
        return

    # Read YOLO format label
    with open(label_path, "r") as label_file:
        lines = label_file.readlines()

    # Draw bounding boxes on the image
    for line in lines:
        elements = line.strip().split()
        class_id = int(elements[0])
        x_center, y_center, width, height = map(float, elements[1:])

        # Convert YOLO format to OpenCV format
        h, w, _ = image.shape
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int(x1 + width * width)
        y2 = int(y1 + height * height)

        # Draw bounding box
        color = (0, 255, 0)  # Green color
        thickness = 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Add class label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        label = f"Class {class_id}"
        cv2.putText(image, label, (x1, y1 - 5), font, font_scale, color, font_thickness)

    # Save or display the result
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Result saved to {output_path}")
    else:
        cv2.imshow("Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "./FullData/images/000000000086.jpg"
    label_path = "./FullData/labels/000000000086.txt"
    output_path = "./output_image.jpg"  # Optional

    draw_bounding_boxes(image_path, label_path, output_path)
