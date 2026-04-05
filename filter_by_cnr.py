"""
CNR Filtresi — YOLO Veri Seti Kalite Kontrolü
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
    "heart": 1.5,
    "brain": 1.5,
    "knee": 1.5,
}

SHARPNESS_THRESHOLDS = {
    "breast": 50.0,
    "heart": 30.0,
    "brain": 20.0,
    "knee": 25.0,
}

MAX_ROBOTS = 25
MAX_PER_ORGAN = 500

# Eski filtrelenmiş dataset varsa sil
if os.path.exists(YOLO_DST):
    shutil.rmtree(YOLO_DST)

for folder in ["images", "labels", "images_annotated"]:
    os.makedirs(os.path.join(YOLO_DST, folder), exist_ok=True)

kept = 0
skipped = 0

skip_reasons = {
    "cnr_low": 0,
    "no_visible": 0,
    "file_missing": 0,
    "max_reached": 0,
    "sharpness_low": 0,
}

organ_kept = {
    "brain": 0,
    "heart": 0,
    "breast": 0,
    "knee": 0,
}

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"Toplam görüntü: {len(rows)}")
print(
    f"CNR eşikleri: beyin={CNR_THRESHOLDS['brain']}, "
    f"kalp={CNR_THRESHOLDS['heart']}, "
    f"göğüs={CNR_THRESHOLDS['breast']}"
)
print()

for row in rows:
    filename = row["filename"]
    robot_cnrs = []
    visible_count = 0

    for r in range(1, MAX_ROBOTS + 1):
        x_key = f"r{r}_x"
        if x_key not in row:
            break
        if row[x_key] == "-1":
            continue

        try:
            vis_val = float(row.get(f"r{r}_visible", 0))
            cnr_val = float(row.get(f"r{r}_cnr", -1))

            # Eski CSV düzenleri için güvenlik
            if vis_val > 1:
                vis_val, cnr_val = cnr_val, vis_val

            visible = int(vis_val)
            cnr = cnr_val
        except (ValueError, TypeError):
            visible = 0
            cnr = -1

        if visible == 1:
            visible_count += 1
            robot_cnrs.append(cnr)

    if visible_count == 0:
        skipped += 1
        skip_reasons["no_visible"] += 1
        continue

    organ = row.get("organ", "breast")
    cnr_thr = CNR_THRESHOLDS.get(organ, 3.0)
    shp_thr = SHARPNESS_THRESHOLDS.get(organ, 30.0)

    # Beyin: %80 geçmeli — diğerleri: tümü geçmeli
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

    src_img_check = os.path.join(YOLO_SRC, "images", filename)
    if os.path.exists(src_img_check):
        img = cv2.imread(src_img_check, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            sharpness = cv2.Laplacian(img, cv2.CV_64F).var()
            if sharpness < shp_thr:
                skipped += 1
                skip_reasons["sharpness_low"] += 1
                continue

    if organ_kept.get(organ, 0) >= MAX_PER_ORGAN:
        skipped += 1
        skip_reasons["max_reached"] += 1
        continue

    src_img = os.path.join(YOLO_SRC, "images", filename)
    src_lbl = os.path.join(YOLO_SRC, "labels", filename.replace(".png", ".txt"))
    src_ann = os.path.join(YOLO_SRC, "images_annotated", filename)

    if not os.path.exists(src_img):
        skipped += 1
        skip_reasons["file_missing"] += 1
        continue

    shutil.copy2(src_img, os.path.join(YOLO_DST, "images", filename))

    if os.path.exists(src_lbl):
        shutil.copy2(
            src_lbl,
            os.path.join(YOLO_DST, "labels", filename.replace(".png", ".txt"))
        )

    if os.path.exists(src_ann):
        shutil.copy2(src_ann, os.path.join(YOLO_DST, "images_annotated", filename))

    kept += 1
    organ_kept[organ] = organ_kept.get(organ, 0) + 1

print(f"Kabul edilen : {kept} görüntü")
print(f"Reddedilen  : {skipped} görüntü")
print(f"  - CNR eşik altı    : {skip_reasons['cnr_low']}")
print(f"  - Görünür robot yok: {skip_reasons['no_visible']}")
print(f"  - Dosya eksik      : {skip_reasons['file_missing']}")
print(f"  - Max organ limiti : {skip_reasons['max_reached']}")
print(f"  - Sharpness düşük  : {skip_reasons['sharpness_low']}")
print()

print(f"Filtrelenmiş veri seti: {YOLO_DST}/")
print(f"  images/            → {kept} görüntü")
print(f"  labels/            → YOLO etiketleri")
print(f"  images_annotated/  → görsel kontrol")

print()
print("Organ dağılımı (filtrelenmiş):")
for organ, count in organ_kept.items():
    print(f"  {organ}: {count}")

print()
print("Negatif örnekler kopyalanıyor...")

neg_count = 0
src_neg_dir = os.path.join(YOLO_SRC, "images")

for fname in os.listdir(src_neg_dir):
    if not fname.startswith("negative_"):
        continue

    shutil.copy2(
        os.path.join(YOLO_SRC, "images", fname),
        os.path.join(YOLO_DST, "images", fname)
    )

    lbl = fname.replace(".png", ".txt")
    src_lbl = os.path.join(YOLO_SRC, "labels", lbl)
    if os.path.exists(src_lbl):
        shutil.copy2(src_lbl, os.path.join(YOLO_DST, "labels", lbl))

    src_ann = os.path.join(YOLO_SRC, "images_annotated", fname)
    if os.path.exists(src_ann):
        shutil.copy2(src_ann, os.path.join(YOLO_DST, "images_annotated", fname))

    neg_count += 1

print(f"  Negatif: {neg_count} görüntü eklendi")
print()
print(f"TOPLAM: {kept} pozitif + {neg_count} negatif = {kept + neg_count} görüntü")
print(f"Negatif oran: %{neg_count / (kept + neg_count) * 100:.1f}")