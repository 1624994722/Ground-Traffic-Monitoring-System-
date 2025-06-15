import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import random

def visdrone2yolo(dir):
    """Convert VisDrone annotations to YOLO format, creating label files with normalized bounding box coordinates."""

    def convert_box(size, box):
        # Convert VisDrone box to YOLO xywh box
        dw = 1.0 / size[0]
        dh = 1.0 / size[1]
        return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

    def mosaic_augmentation(images, labels, img_size):
        # Mosaic 数据增强
        s = img_size
        mosaic_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        yc, xc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
        mosaic_labels = []

        for i in range(4):
            img = images[i]
            h, w = img.shape[:2]
            label = labels[i]

            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            # Update labels
            for l in label:
                cls = l[0]
                x, y, w, h = l[1:]
                x = (x * w + padw) / (s * 2)
                y = (y * h + padh) / (s * 2)
                w = w / (s * 2)
                h = h / (s * 2)
                mosaic_labels.append([cls, x, y, w, h])

        return mosaic_img, np.array(mosaic_labels)

    def clahe_enhancement(img):
        # CLAHE 图像增强
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced_img = cv2.merge((l, a, b))
        enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2BGR)
        return enhanced_img

    (dir / "labels").mkdir(parents=True, exist_ok=True)  # make labels directory
    pbar = tqdm((dir / "annotations").glob("*.txt"), desc=f"Converting {dir}")
    for f in pbar:
        img_path = (dir / "images" / f.name).with_suffix(".jpg")
        img = np.array(Image.open(img_path))
        
        # CLAHE 图像增强
        enhanced_img = clahe_enhancement(img)
        Image.fromarray(enhanced_img).save(img_path)

        img_size = Image.open(img_path).size
        lines = []
        with open(f, encoding="utf-8") as file:  # read annotation.txt
            for row in [x.split(",") for x in file.read().strip().splitlines()]:
                if row[4] == "0":  # VisDrone 'ignored regions' class 0
                    continue
                cls = int(row[5]) - 1
                box = convert_box(img_size, tuple(map(int, row[:4])))
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
        
        with open(str(f).replace(f"{os.sep}annotations{os.sep}", f"{os.sep}labels{os.sep}"), "w", encoding="utf-8") as fl:
            fl.writelines(lines)  # write label.txt

    # Mosaic 数据增强
    image_files = list((dir / "images").glob("*.jpg"))
    random.shuffle(image_files)
    for i in range(0, len(image_files) - 3, 4):
        images = []
        labels = []
        for j in range(4):
            img_path = image_files[i + j]
            img = np.array(Image.open(img_path))
            label_path = str(img_path).replace("images", "labels").replace(".jpg", ".txt")
            with open(label_path, "r", encoding="utf-8") as label_file:
                label_lines = label_file.readlines()
                label = []
                for line in label_lines:
                    parts = line.strip().split()
                    cls = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                    label.append([cls, x, y, w, h])
            images.append(img)
            labels.append(np.array(label))

        mosaic_img, mosaic_label = mosaic_augmentation(images, labels, img_size[0])
        mosaic_img_path = dir / "images" / f"mosaic_{i // 4}.jpg"
        Image.fromarray(mosaic_img).save(mosaic_img_path)
        mosaic_label_path = str(mosaic_img_path).replace("images", "labels").replace(".jpg", ".txt")
        with open(mosaic_label_path, "w", encoding="utf-8") as mosaic_label_file:
            for l in mosaic_label:
                cls = l[0]
                x, y, w, h = l[1:]
                mosaic_label_file.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

# Download
dir = Path("./VisDrone")  # dataset root dir

# Convert
for d in "VisDrone2019-DET-train", "VisDrone2019-DET-val", "VisDrone2019-DET-test-dev":
    visdrone2yolo(dir / d)  # convert VisDrone annotations to YOLO labels