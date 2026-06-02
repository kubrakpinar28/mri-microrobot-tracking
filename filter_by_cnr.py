"""
CNR Filtresi — YOLO Veri Seti Kalite Kontrolü
Yeni CSV sütun adları (r{n}_true_cx) ve train/val/test split yapısı ile uyumlu.
"""

import os
import csv
import shutil
import cv2
import numpy as np

CSV_PATH = "synthetic_dataset_all/labels.csv"
YOLO_SRC = "synthetic_dataset_yolo"
YOLO_DST = "synthetic_dataset_yolo_filtered"

CNR_THRESHOLDS = {
    "breast": 3.0,
    "heart":  1.5,
    "brain":  0.8,   # beyin doğası gereği düşük kontrast
    "knee":   1.5,
}

SHARPNESS_THRESHOLDS = {
    "breast": 50.0,
    "heart":  30.0,
    "brain":  20.0,
    "knee":   25.0,
}

MAX_ROBOTS   = 25
MAX_PER_ORGAN = 500

# Eski filtrelenmiş dataset varsa sil
if os.path.exists(YOLO_DST):
    shutil.rmtree(YOLO_DST)

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(YOLO_DST, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(YOLO_DST, split, "labels"), exist_ok=True)
os.makedirs(os.path.join(YOLO_DST, "images_annotated"), exist_ok=True)

kept   = 0
skipped = 0
skip_reasons = {
    "cnr_low":       0,
    "no_visible":    0,
    "file_missing":  0,
    "max_reached":   0,
    "sharpness_low": 0,
}
organ_kept = {"brain": 0, "heart": 0, "breast": 0, "knee": 0}

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"Toplam goruntu: {len(rows)}")
print(f"CNR eslikleri : brain={CNR_THRESHOLDS['brain']}, "
      f"heart={CNR_THRESHOLDS['heart']}, "
      f"breast={CNR_THRESHOLDS['breast']}, "
      f"knee={CNR_THRESHOLDS['knee']}")
print(f"Max per organ : {MAX_PER_ORGAN}")
print()

for row in rows:
    filename = row["filename"]

    # Negatif ornekleri atla — asagida ayrica kopyalanacak
    if filename.startswith("negative_"):
        continue

    split  = row.get("split", "train")
    organ  = row.get("organ", "brain")

    # Organ limiti
    if organ_kept.get(organ, 0) >= MAX_PER_ORGAN:
        skipped += 1
        skip_reasons["max_reached"] += 1
        continue

    # Robot CNR ve gorunurluk kontrolu
    # Yeni CSV: r{n}_true_cx, r{n}_visible, r{n}_cnr
    robot_cnrs    = []
    visible_count = 0

    for r in range(1, MAX_ROBOTS + 1):
        cx_key = f"r{r}_true_cx"
        if cx_key not in row:
            break
        if str(row[cx_key]).strip() == "-1":
            continue
        try:
            visible = int(float(row.get(f"r{r}_visible", 0)))
            cnr     = float(row.get(f"r{r}_cnr", -1))
        except (ValueError, TypeError):
            visible = 0
            cnr     = -1
        if visible == 1:
            visible_count += 1
            robot_cnrs.append(cnr)

    if visible_count == 0:
        skipped += 1
        skip_reasons["no_visible"] += 1
        continue

    # CNR esigi
    cnr_thr = CNR_THRESHOLDS.get(organ, 1.5)
    if organ == "brain":
        passing = sum(1 for c in robot_cnrs if c >= cnr_thr)
        if passing / len(robot_cnrs) < 0.80:
            skipped += 1
            skip_reasons["cnr_low"] += 1
            continue
    else:
        if not all(c >= cnr_thr for c in robot_cnrs):
            skipped += 1
            skip_reasons["cnr_low"] += 1
            continue

    # Sharpness kontrolu
    src_img = os.path.join(YOLO_SRC, split, "images", filename)
    if not os.path.exists(src_img):
        skipped += 1
        skip_reasons["file_missing"] += 1
        continue

    img = cv2.imread(src_img, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        sharpness = cv2.Laplacian(img, cv2.CV_64F).var()
        shp_thr   = SHARPNESS_THRESHOLDS.get(organ, 30.0)
        if sharpness < shp_thr:
            skipped += 1
            skip_reasons["sharpness_low"] += 1
            continue

    # Dosyalari kopyala
    dst_img = os.path.join(YOLO_DST, split, "images", filename)
    shutil.copy2(src_img, dst_img)

    src_lbl = os.path.join(YOLO_SRC, split, "labels",
                           filename.replace(".png", ".txt"))
    if os.path.exists(src_lbl):
        shutil.copy2(src_lbl,
                     os.path.join(YOLO_DST, split, "labels",
                                  filename.replace(".png", ".txt")))

    src_ann = os.path.join(YOLO_SRC, "images_annotated", filename)
    if os.path.exists(src_ann):
        shutil.copy2(src_ann,
                     os.path.join(YOLO_DST, "images_annotated", filename))

    kept += 1
    organ_kept[organ] = organ_kept.get(organ, 0) + 1

# Negatif ornekleri split'e gore kopyala
print("Negatif ornekler kopyalaniyor...")
neg_count = 0
for split in ["train", "val", "test"]:
    src_dir = os.path.join(YOLO_SRC, split, "images")
    if not os.path.exists(src_dir):
        continue
    for fname in os.listdir(src_dir):
        if not fname.startswith("negative_"):
            continue
        shutil.copy2(
            os.path.join(src_dir, fname),
            os.path.join(YOLO_DST, split, "images", fname))
        src_lbl = os.path.join(YOLO_SRC, split, "labels",
                               fname.replace(".png", ".txt"))
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl,
                         os.path.join(YOLO_DST, split, "labels",
                                      fname.replace(".png", ".txt")))
        neg_count += 1

print(f"  {neg_count} negatif goruntu eklendi")
print()

total = kept + neg_count
print(f"Kabul edilen (pozitif) : {kept}")
print(f"Reddedilen             : {skipped}")
print(f"  - CNR esik alti      : {skip_reasons['cnr_low']}")
print(f"  - Gorunur robot yok  : {skip_reasons['no_visible']}")
print(f"  - Dosya eksik        : {skip_reasons['file_missing']}")
print(f"  - Max organ limiti   : {skip_reasons['max_reached']}")
print(f"  - Sharpness dusuk    : {skip_reasons['sharpness_low']}")
print()
print("Organ dagilimi (filtrelenmis):")
for organ, count in organ_kept.items():
    print(f"  {organ}: {count}")
print()
print(f"TOPLAM: {kept} pozitif + {neg_count} negatif = {total}")
if total > 0:
    print(f"Negatif oran: %{neg_count / total * 100:.1f}")
print()

# Split dagilimi
print("Split dagilimi (filtrelenmis pozitif):")
for split in ["train", "val", "test"]:
    split_dir = os.path.join(YOLO_DST, split, "images")
    if os.path.exists(split_dir):
        n = len([f for f in os.listdir(split_dir)
                 if f.startswith("synthetic_")])
        print(f"  {split}: {n} pozitif goruntu")
print()
print(f"Filtrelenmis veri seti: {YOLO_DST}/")