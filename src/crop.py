from pathlib import Path
from typing import List
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms
import imagehash
from config import LABEL
from detector import run_detector




# Class map



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



def prepare_crops_for_cnn(
    image_path: Path = "doc.jpg",
    target_width: int = 256,
    target_height: int = 64,
    phash_ham_thr: int = 6,
    phash_min_h: int = 16,
    phash_min_area: int = 64,
) -> torch.Tensor:
    """
    Runs YOLO detection, applies pHash dedup, and returns CNN-ready batch tensor.
    All crops resized to fixed [C, target_height, target_width].
    """
    img = cv2.imread(str(image_path))
    det_res = run_detector(image_path=image_path)

    if not det_res:
        return torch.empty(0)

    boxes = np.array([det["xyxy"] for det in det_res])
    confs = np.ones(len(det_res))  # dummy confidences

    keep_idx = _phash_dedup(img, boxes, confs, phash_ham_thr, phash_min_h, phash_min_area)
    boxes = boxes[keep_idx]
    det_res = [det_res[i] for i in keep_idx]

    crops: List[torch.Tensor] = []
    for det, box in zip(det_res, boxes):
        x1, y1, x2, y2 = map(int, box)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_resized = cv2.resize(crop, (target_width, target_height))
        tensor = torch.from_numpy(crop_resized[:, :, ::-1].copy())  # BGR -> RGB
        tensor = tensor.permute(2, 0, 1).float() / 255.0
        crops.append(tensor)

    if crops:
        batch = torch.stack(crops)  # [num_crops, 3, H, W]
    else:
        batch = torch.empty(0)

    return batch