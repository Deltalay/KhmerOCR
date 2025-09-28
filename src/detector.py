import cv2
import numpy as np
from doclayout_yolo import YOLOv10


def run_detector(
    image_path: str = "assets/test1.jpg",
    model_path: str = "model/yolo.pt",
    out_path: str = "result.jpg",
    imgsz: int = 1024,
    conf: float = 0.25,
    iou: float = 0.55,
    device: str = "cuda:0",
    max_det: int = 300,
) -> None:
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
    det_res = model.predict(
        image_path,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        max_det=max_det,
    )

    # Annotate and save the result
    annotated_frame = det_res[0].plot(pil=True, line_width=5, font_size=20)
    annotated_frame = cv2.cvtColor(np.array(annotated_frame), cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, annotated_frame)


if __name__ == "__main__":
    run_detector()
