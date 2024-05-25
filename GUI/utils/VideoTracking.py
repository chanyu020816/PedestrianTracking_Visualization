import cv2

def get_video_cfg(path):
    video = cv2.VideoCapture(path)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    return cv2.VideoWriter_fourcc(*'XVID'), size, fps

def counting(image_plot, result):
    box_count = result.boxes.shape[0]
    cv2.putText(
        image_plot, f'Object Counts:{box_count}', (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4
    )
    return image_plot