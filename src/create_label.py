import os
from lxml import etree
import unicodedata

ANNOT_DIR = "fix_labels"
LINE_IMG_DIR = "crops"
OUTPUT_LABELS = "labels.txt"

def normalize_khmer(text):
    return unicodedata.normalize("NFC", text.strip())

with open(OUTPUT_LABELS, "w", encoding="utf-8") as out_file:
    for fname in sorted(os.listdir(ANNOT_DIR)):
        if not fname.lower().endswith(".xml"):
            continue
        xml_path = os.path.join(ANNOT_DIR, fname)
        tree = etree.parse(xml_path)
        root = tree.getroot()
        image_name = root.findtext("image")
        if image_name is None:
            continue
        prefix = os.path.splitext(image_name)[0]
        lines = root.xpath(".//paragraph/line")
        for line in lines:
            line_id = line.get("id")
            text_el = line.find("text")
            if text_el is None:
                continue
            text = normalize_khmer(text_el.text or "")
            img_filename = f"{prefix}_{line_id}.png"
            img_path = os.path.join(LINE_IMG_DIR, img_filename)
            if os.path.exists(img_path):
                out_file.write(f"{img_path}\t{text}\n")
