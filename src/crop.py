from pathlib import Path
from typing import List
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms
import imagehash
from doclayout_yolo import YOLOv10

ROOT: Path = Path(__file__).resolve().parent

# Class map
CLASS_NAMES: dict[int, str] = {
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


def _phash_dedup(
    img_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
    confs: np.ndarray,
    ham_thr: int = 6,
    min_h: int = 16,
    min_area: int = 64,
) -> np.ndarray:
    """
    Perceptual-hash deduplication on cropped regions.
    Keeps one crop per pHash cluster using highest confidence, then larger area.
    """
    keep: List[int] = []
    clusters: List[tuple[imagehash.ImageHash, int]] = []  # (hash, position in keep)

    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy.astype(int)):
        w, h = max(0, x2 - x1), max(0, y2 - y1)
        if h < min_h or w * h < min_area:
            continue

        crop = img_bgr[y1:y2, x1:x2]
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).convert("L").resize((128, 128))
        hsh = imagehash.phash(pil)  # 64-bit pHash

        match = None
        for k, (prev_hash, keep_pos) in enumerate(clusters):
            if hsh - prev_hash <= ham_thr:
                match = (k, keep_pos)
                break

        if match is None:
            keep.append(i)
            clusters.append((hsh, len(keep) - 1))
        else:
            k, keep_pos = match
            j = keep[keep_pos]
            area_i = w * h
            area_j = (boxes_xyxy[j, 2] - boxes_xyxy[j, 0]) * (boxes_xyxy[j, 3] - boxes_xyxy[j, 1])
            better = (confs[i] > confs[j]) or (confs[i] == confs[j] and area_i > area_j)
            if better:
                keep[keep_pos] = i
                clusters[k] = (hsh, keep_pos)

    return np.array(keep, dtype=int)


def run_cropper(
    image_path: Path = ROOT.parent / "assets" / "test1.jpg", 
    model_path: Path = ROOT.parent / "models" / "yolo.pt",             # src/model
    out_dir: Path = ROOT / "cropped",                          # src/cropped
    imgsz: int = 1024,
    conf: float = 0.25,
    iou: float = 0.55,
    nms_iou: float = 0.75,
    device: str = "cuda:0",
    max_det: int | None = 300,
    phash_ham_thr: int = 6,
    phash_min_h: int = 16,
    phash_min_area: int = 64,
) -> None:
    """
    Detect text blocks with YOLO, apply NMS and pHash dedup, then save crops.
    """
    image_path = Path(image_path)
    model_path = Path(model_path)
    out_dir = Path(out_dir)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load model
    model = YOLOv10(str(model_path))

    # Predict (YOLO includes its own NMS)
    predict_kwargs = dict(imgsz=imgsz, conf=conf, iou=iou, device=device)
    if max_det is not None:
        predict_kwargs["max_det"] = max_det
    det_res = model.predict(str(image_path), **predict_kwargs)

    # Load image
    img = cv2.imread(str(image_path))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Raw detections (post-YOLO NMS)
    boxes = det_res[0].boxes.xyxy.cpu()
    confs = det_res[0].boxes.conf.cpu()
    classes = det_res[0].boxes.cls.cpu()

    # Extra NMS per class (torchvision)
    keep_idx = []
    for c in classes.unique():
        idx = (classes == c).nonzero(as_tuple=True)[0]
        kept = nms(boxes[idx], confs[idx], iou_threshold=nms_iou)
        keep_idx.append(idx[kept])
    keep_idx = torch.cat(keep_idx)

    boxes = boxes[keep_idx].numpy()
    confs = confs[keep_idx].numpy()
    classes = classes[keep_idx].numpy()

    # pHash dedup
    ph_keep = _phash_dedup(
        img_bgr=img,
        boxes_xyxy=boxes,
        confs=confs,
        ham_thr=phash_ham_thr,
        min_h=phash_min_h,
        min_area=phash_min_area,
    )
    boxes, confs, classes = boxes[ph_keep], confs[ph_keep], classes[ph_keep]

    # Save crops with class names
    for i, (box, conf_val, cls_val) in enumerate(zip(boxes, confs, classes)):
        x1, y1, x2, y2 = map(int, box)
        crop = img[y1:y2, x1:x2]

        cls_name = CLASS_NAMES.get(int(cls_val), f"cls{int(cls_val)}")
        # safe filenames: replace spaces with underscore
        cls_name = cls_name.replace(" ", "_")

        out_path = out_dir / f"crop_{i:03d}_{cls_name}_{conf_val:.2f}.png"
        cv2.imwrite(str(out_path), crop)


if __name__ == "__main__":
    run_cropper()
