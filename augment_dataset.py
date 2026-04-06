"""
Veri Artirma (Data Augmentation) — Sadece Train
================================================
synthetic_dataset_yolo_split/train/ klasorundeki goruntuleri
augmentation ile cogaltir.

ONEMLI: Augmentation SADECE train'e uygulanir.
Val seti temiz kalir — veri sizintisi olmaz.

Her goruntu icin 2 augmented versiyon uretir.

Kullanim:
  py augment_dataset.py
"""

import os
import cv2
import numpy as np
import shutil
import random

SRC_DIR = "synthetic_dataset_yolo_split/train"
DST_DIR = "synthetic_dataset_yolo_augmented"
N_AUG   = 2
SEED    = 42

random.seed(SEED)
np.random.seed(SEED)

# Eski augmented dataset varsa sil
if os.path.exists(DST_DIR):
    shutil.rmtree(DST_DIR)

# Klasorler
for folder in ["train/images", "train/labels", "val/images", "val/labels"]:
    os.makedirs(os.path.join(DST_DIR, folder), exist_ok=True)

# Val klasorunu oldugu gibi kopyala — augmentation yok
print("Val seti kopyalaniyor (augmentation yok)...")
val_src = "synthetic_dataset_yolo_split/val"
val_count = 0
for fname in os.listdir(os.path.join(val_src, "images")):
    if not (fname.endswith(".png") or fname.endswith(".jpg")):
        continue
    shutil.copy2(
        os.path.join(val_src, "images", fname),
        os.path.join(DST_DIR, "val", "images", fname)
    )
    lbl = fname.replace(".png", ".txt").replace(".jpg", ".txt")
    src_lbl = os.path.join(val_src, "labels", lbl)
    if os.path.exists(src_lbl):
        shutil.copy2(src_lbl, os.path.join(DST_DIR, "val", "labels", lbl))
    else:
        open(os.path.join(DST_DIR, "val", "labels", lbl), "w").close()
    val_count += 1
print(f"  {val_count} val goruntu kopyalandi")

# Train orijinallerini kopyala
print("\nTrain orijinalleri kopyalaniyor...")
orig_count = 0
for fname in os.listdir(os.path.join(SRC_DIR, "images")):
    if not (fname.endswith(".png") or fname.endswith(".jpg")):
        continue
    shutil.copy2(
        os.path.join(SRC_DIR, "images", fname),
        os.path.join(DST_DIR, "train", "images", fname)
    )
    lbl = fname.replace(".png", ".txt").replace(".jpg", ".txt")
    src_lbl = os.path.join(SRC_DIR, "labels", lbl)
    if os.path.exists(src_lbl):
        shutil.copy2(src_lbl, os.path.join(DST_DIR, "train", "labels", lbl))
    else:
        open(os.path.join(DST_DIR, "train", "labels", lbl), "w").close()
    orig_count += 1
print(f"  {orig_count} train goruntu kopyalandi")


def read_yolo_labels(label_path):
    if not os.path.exists(label_path):
        return []
    lines = []
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                lines.append((int(parts[0]),
                               float(parts[1]), float(parts[2]),
                               float(parts[3]), float(parts[4])))
    return lines


def write_yolo_labels(label_path, labels):
    with open(label_path, "w") as f:
        for cls, xc, yc, w, h in labels:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def flip_labels_h(labels):
    return [(cls, 1.0 - xc, yc, w, h) for cls, xc, yc, w, h in labels]

def flip_labels_v(labels):
    return [(cls, xc, 1.0 - yc, w, h) for cls, xc, yc, w, h in labels]

def rotate_labels(labels, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    cx, cy = 0.5, 0.5
    new_labels = []
    for cls, xc, yc, w, h in labels:
        dx, dy = xc - cx, yc - cy
        new_x = cx + dx * np.cos(angle_rad) - dy * np.sin(angle_rad)
        new_y = cy + dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
        half_w, half_h = w / 2, h / 2
        corners = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
        rot = [(cx2 * np.cos(angle_rad) - cy2 * np.sin(angle_rad),
                cx2 * np.sin(angle_rad) + cy2 * np.cos(angle_rad)) for cx2, cy2 in corners]
        new_w = max(abs(c[0]) for c in rot) * 2
        new_h = max(abs(c[1]) for c in rot) * 2
        new_x = float(np.clip(new_x, new_w / 2, 1 - new_w / 2))
        new_y = float(np.clip(new_y, new_h / 2, 1 - new_h / 2))
        new_labels.append((cls, new_x, new_y, min(new_w, 0.5), min(new_h, 0.5)))
    return new_labels

def aug_flip_h(img, labels):
    return cv2.flip(img, 1), flip_labels_h(labels)

def aug_flip_v(img, labels):
    return cv2.flip(img, 0), flip_labels_v(labels)

def aug_brightness(img, labels):
    f = np.random.uniform(0.7, 1.3)
    return np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8), labels

def aug_contrast(img, labels):
    mean = img.mean()
    f = np.random.uniform(0.8, 1.2)
    return np.clip((img.astype(np.float32) - mean) * f + mean, 0, 255).astype(np.uint8), labels

def aug_noise(img, labels):
    sigma = np.random.uniform(2, 8)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8), labels

def aug_rotate(img, labels):
    angle = np.random.uniform(-15, 15)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    return rotated, rotate_labels(labels, angle)

AUG_FUNCS = [aug_flip_h, aug_flip_v, aug_brightness, aug_contrast, aug_noise, aug_rotate]

print(f"\nAugmentation uygulanıyor (train, her goruntu icin {N_AUG} versiyon)...")
aug_count = 0
image_files = [
    f for f in os.listdir(os.path.join(SRC_DIR, "images"))
    if f.endswith(".png") or f.endswith(".jpg")
]

for fname in image_files:
    is_negative = fname.startswith("negative_")
    img_path = os.path.join(SRC_DIR, "images", fname)
    lbl_path = os.path.join(
        SRC_DIR, "labels",
        fname.replace(".png", ".txt").replace(".jpg", ".txt")
    )

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    labels = read_yolo_labels(lbl_path)

    funcs = [aug_flip_h, aug_brightness, aug_noise] if is_negative else AUG_FUNCS
    chosen = random.sample(funcs, min(N_AUG, len(funcs)))

    for j, aug_fn in enumerate(chosen):
        aug_img, aug_labels = aug_fn(img.copy(), list(labels))
        base = fname.replace(".png", "").replace(".jpg", "")
        new_name = f"{base}_aug{j+1}.png"
        new_lbl = f"{base}_aug{j+1}.txt"

        cv2.imwrite(os.path.join(DST_DIR, "train", "images", new_name), aug_img)
        write_yolo_labels(os.path.join(DST_DIR, "train", "labels", new_lbl), aug_labels)
        aug_count += 1

train_total = orig_count + aug_count
print(f"  {aug_count} augmented goruntu uretildi")
print()
print("SONUC:")
print(f"  train/: {train_total} goruntu ({orig_count} orijinal + {aug_count} augmented)")
print(f"  val/  : {val_count} goruntu (temiz, augmentation yok)")
print(f"  Toplam: {train_total + val_count} goruntu")
print(f"\nCikti: {DST_DIR}/")