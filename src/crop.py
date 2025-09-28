import os
import cv2
import numpy as np
import torch
from torchvision.ops import nms
from doclayout_yolo import YOLOv10

# Load model
model = YOLOv10(r"C:\Users\b2324\Desktop\KhmerOCR\model\yolo.pt")

# Predict
image_path = r"C:\Users\b2324\Desktop\KhmerOCR\assets\test1.jpg"
det_res = model.predict(image_path, imgsz=1024, conf=0.25, iou=0.55, device="cuda:0")

# Load original image (for cropping)
img = cv2.imread(image_path)

# Output directory
out_dir = r"C:\Users\b2324\Desktop\KhmerOCR\cropped"
os.makedirs(out_dir, exist_ok=True)

# Get detections
boxes = det_res[0].boxes.xyxy.cpu()
confs = det_res[0].boxes.conf.cpu()
classes = det_res[0].boxes.cls.cpu()

# NMS to remove duplicates (per class)
keep_idx = []
for c in classes.unique():
    idx = (classes == c).nonzero(as_tuple=True)[0]
    kept = nms(boxes[idx], confs[idx], iou_threshold=0.75)
    keep_idx.append(idx[kept])
keep_idx = torch.cat(keep_idx)

boxes = boxes[keep_idx].numpy()
confs = confs[keep_idx].numpy()
classes = classes[keep_idx].numpy()

# Crop loop
for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
    x1, y1, x2, y2 = map(int, box)
    crop = img[y1:y2, x1:x2]
    out_path = os.path.join(out_dir, f"crop_{i:03d}_c{int(cls)}_{conf:.2f}.png")
    cv2.imwrite(out_path, crop)
