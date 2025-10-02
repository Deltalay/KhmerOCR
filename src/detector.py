from doclayout_yolo import YOLOv10
from config import YOLO_PATH, LABEL


def run_detector(
    image_path: str = "assets/test1.jpg",
    model_path: str = YOLO_PATH,
    imgsz: int = 1120,
    conf: float = 0.25,
    iou: float = 0.55,
    device: str = "cuda:0",
    max_det: int = 300,
) -> list[dict]:
    """
    Run YOLOv10 detector on a single image and save the annotated result.

    Args:
        image_path: Path to the input image.
        model_path: Path to the YOLOv10 model weights (.pt file).
        out_path: Path to save the annotated result image.
        imgsz: Inference image size.
        conf: Confidence threshold for predictions.
        iou: IoU threshold for non-max suppression.
        device: Device identifier (e.g., "cuda:0" or "cpu").
        max_det: Maximum number of detections per image.
    """
    # Load model
    model = YOLOv10(model_path)

    # Perform prediction
    results = model.predict(
        image_path,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        max_det=max_det,
    )
    detection = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            label = int(box.cls[0])
            print(label)
            if label == 6:
                continue
            eachBox = {
                "label": LABEL.get(label),
                "xyxyn": box.xyxyn[0].tolist(),
                "xyxy": box.xyxy[0].tolist(),
            }
            detection.append(eachBox)
    return detection
