from ultralytics import YOLO

results = model(img, verbose=False, stream=True, device="mps", classes=[0])