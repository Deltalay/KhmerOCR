import cv2
import numpy as np
from pathlib import Path
from doclayout_yolo import YOLOv10

ROOT = Path(__file__).resolve().parent 

# ---- Class map ----
CLASS_NAMES = {
    0: "Caption",
    1: "Footnote",
    2: "Formula",
    3: "List-item",
    4: "Page-footer",
    5: "Page-header",
    6: "Picture",
    7: "Section-header",
    8: "Table",
    9: "Text",
    10: "Title",
}

# Stable color per class
_rng = np.random.default_rng(42)
_COLORS = {k: tuple(int(c) for c in _rng.integers(50, 205, 3)) for k in CLASS_NAMES}


def _draw_detections(img: np.ndarray, result) -> np.ndarray:
    """
    result: ultralytics-style single Result
            expects result.boxes.xyxy, result.boxes.cls, result.boxes.conf
    """
    if not hasattr(result, "boxes") or result.boxes is None:
        return img

    xyxy = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes.xyxy, "cpu") else result.boxes.xyxy
    cls  = result.boxes.cls.cpu().numpy().astype(int) if hasattr(result.boxes.cls, "cpu") else result.boxes.cls.astype(int)
    conf = result.boxes.conf.cpu().numpy() if hasattr(result.boxes.conf, "cpu") else result.boxes.conf

    for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
        label = CLASS_NAMES.get(int(c), f"id{int(c)}")
        color = _COLORS.get(int(c), (160, 160, 160))
        x1, y1, x2, y2 = map(lambda v: int(round(float(v))), (x1, y1, x2, y2))

        # box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)

        # label
        txt = f"{label} {p:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        th = int(th * 1.4)
        cv2.rectangle(img, (x1, max(0, y1 - th - 4)), (x1 + tw + 6, y1), color, thickness=-1)
        cv2.putText(img, txt, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return img

def run_detector(
    image_path: Path = ROOT.parent / "assets" / "test1.jpg",  # up one level
    model_path: Path = ROOT / "model" / "yolo.pt",            # inside src/model
    out_path: Path = ROOT / "result.jpg",
    imgsz: int = 1024,
    conf: float = 0.25,
    iou: float = 0.55,
    device: str = "cuda:0",
    max_det: int = 300,
) -> None:
    image_path = Path(image_path)
    model_path = Path(model_path)
    out_path = Path(out_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    model = YOLOv10(model_path)

    # Inference
    det_res = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        max_det=max_det,
        verbose=False,
    )

    # Read original for clean drawing
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    # Draw decoded labels
    if len(det_res) == 0:
        annotated = img_bgr
    else:
        annotated = _draw_detections(img_bgr.copy(), det_res[0])

    # Save
    ok = cv2.imwrite(str(out_path), annotated)
    if not ok:
        raise RuntimeError(f"Failed to write output: {out_path}")


if __name__ == "__main__":
    run_detector()
