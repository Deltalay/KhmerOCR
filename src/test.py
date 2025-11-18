import cv2
from lxml import etree
import os

output_dir = "crops"
os.makedirs(output_dir, exist_ok=True)

amount = 0

def crop_image(label):
    global amount
    xml_file = f"fix_labels/{label}"
    no_ext = os.path.splitext(label)[0]
    image_file = f"image/{no_ext}.png"

    if not os.path.exists(image_file):
        print(f"Image not found: {image_file}")
        return

    img = cv2.imread(image_file)
    if img is None:
        print(f"Failed to load image: {image_file}")
        return

    h, w = img.shape[:2]

    tree = etree.parse(xml_file)
    root = tree.getroot()

    for line in root.xpath("//line"):
        line_id = line.get("id")
        text = line.findtext("text")
        bbox = line.find("bbox")
        if bbox is None:
            continue

        x1 = max(0, int(bbox.get("x1")))
        y1 = max(0, int(bbox.get("y1")))
        x2 = min(w, int(bbox.get("x2")))
        y2 = min(h, int(bbox.get("y2")))

        if x2 <= x1 or y2 <= y1:
            print(f"Invalid crop coordinates in {image_file}, line {line_id}")
            continue

        crop = img[y1:y2, x1:x2]
        out_path = os.path.join(output_dir, f"{no_ext}_{line_id}.png")
        cv2.imwrite(out_path, crop)
        amount += 1

xml_dir = os.listdir("fix_labels")
index = 0
for file in xml_dir:
    if index == 50000:
        break
    if not file.endswith(".xml"):
        continue
    crop_image(file)
    index = index + 1

print(f"Processed: {amount} crops")
