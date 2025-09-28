import cv2
import numpy as np
from doclayout_yolo import YOLOv10

# Load the pre-trained model
model = YOLOv10(r"C:\Users\b2324\Desktop\KhmerOCR\model\yolo.pt")

# Perform prediction
det_res = model.predict(
    r"C:\Users\b2324\Desktop\KhmerOCR\assets\test1.jpg",
    imgsz=1024,
    conf=0.25,     # not too high, keeps weak but valid blocks
    iou=0.55,      # moderate NMS, less aggressive
    device="cuda:0",
    max_det=300    # avoid truncation 
)

# Annotate and save the result
annotated_frame = det_res[0].plot(pil=True, line_width=5, font_size=20)
annotated_frame = cv2.cvtColor(np.array(annotated_frame), cv2.COLOR_RGB2BGR)
cv2.imwrite("result.jpg", annotated_frame)
