import os

path = r"C:\Users\b2324\Desktop\KhmerOCR\crops"
path1 = r"C:\Users\b2324\Desktop\KhmerOCR\fix_labels"
count = len([f for f in os.listdir(path) if f.lower().endswith(('.xml','.png','.jpeg'))])
count1 = len([f for f in os.listdir(path1) if f.lower().endswith(('.xml','.png','.jpeg'))])
print(count, count1)
