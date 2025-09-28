import os
import cv2
import numpy as np
import torch
from torchvision.ops import nms
from doclayout_yolo import YOLOv10


def run_cropper(
    image_path: str = "assets/test1.jpg",
    model_path: str = "model/yolo.pt",
    out_dir: str = "cropped",
    imgsz: int = 1024,
    conf: float = 0.25,
    iou: float = 0.55,
    nms_iou: float = 0.75,
    device: str = "cuda:0",
) -> None:
    """
    Run YOLOv10 detection on an image and save cropped regions.

    Args:
        image_path: Path to input image.
        model_path: Path to YOLOv10 weights file.
        out_dir: Directory to save cropped images.
        imgsz: Inference image size.
        conf: Confidence threshold.
        iou: IoU threshold for detection.
        nms_iou: IoU threshold for NMS deduplication.
        device: Device identifier ("cuda:0" or "cpu").
    """
    # Load model
    model = YOLOv10(model_path)

    # Predict
    det_res = model.predict(
        image_path, imgsz=imgsz, conf=conf, iou=iou, device=device
    )

    # Load original image
    img = cv2.imread(image_path)

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Get detections
    boxes = det_res[0].boxes.xyxy.cpu()
    confs = det_res[0].boxes.conf.cpu()
    classes = det_res[0].boxes.cls.cpu()

    # Apply NMS to reduce duplicates (per class)
    keep_idx = []
    for c in classes.unique():
        idx = (classes == c).nonzero(as_tuple=True)[0]
        kept = nms(boxes[idx], confs[idx], iou_threshold=nms_iou)
        keep_idx.append(idx[kept])
    keep_idx = torch.cat(keep_idx)

    # Filter boxes
    boxes = boxes[keep_idx].numpy()
    confs = confs[keep_idx].numpy()
    classes = classes[keep_idx].numpy()

    # Save crops
    for i, (box, conf_val, cls_val) in enumerate(zip(boxes, confs, classes)):
        x1, y1, x2, y2 = map(int, box)
        crop = img[y1:y2, x1:x2]
        out_path = os.path.join(out_dir, f"crop_{i:03d}_c{int(cls_val)}_{conf_val:.2f}.png")
        cv2.imwrite(out_path, crop)


if __name__ == "__main__":
    run_cropper()
