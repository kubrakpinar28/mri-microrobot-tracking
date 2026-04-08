"""
Veri Artirma (Data Augmentation) — Sadece Train | YOLO-Seg
===========================================================
synthetic_dataset_yolo_split/train/ klasorundeki goruntuleri
augmentation ile cogaltir.

ONEMLI: Augmentation SADECE train'e uygulanir.
Val seti temiz kalir — veri sizintisi olmaz.

YOLO-seg label formati: 0 x1 y1 x2 y2 x3 y3 ...
Her nokta normalize edilmis (0-1 arasi) koordinat.

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

for folder in ["train/images", "train/labels", "val/images", "val/labels"]:
    os.makedirs(os.path.join(DST_DIR, folder), exist_ok=True)

# ── Label okuma / yazma ──────────────────────────────────────────────────────

def read_seg_labels(label_path):
    """
    YOLO-seg label oku.
    Her satir: class_id x1 y1 x2 y2 ...
    Donus: [(cls, [(x1,y1), (x2,y2), ...]), ...]
    """
    if not os.path.exists(label_path):
        return []
    instances = []
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
            # Koordinatlar cift olmali
            if len(coords) < 6 or len(coords) % 2 != 0:
                continue
            pts = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
            instances.append((cls, pts))
    return instances


def write_seg_labels(label_path, instances):
    """
    YOLO-seg label yaz.
    instances: [(cls, [(x1,y1), (x2,y2), ...]), ...]
    """
    with open(label_path, "w") as f:
        for cls, pts in instances:
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
            f.write(f"{cls} {coords}\n")


# ── Polygon donusturuculer ───────────────────────────────────────────────────

def flip_pts_h(pts):
    """Yatay flip: x -> 1-x"""
    return [(1.0 - x, y) for x, y in pts]

def flip_pts_v(pts):
    """Dikey flip: y -> 1-y"""
    return [(x, 1.0 - y) for x, y in pts]

def rotate_pts(pts, angle_deg):
    """Merkez (0.5, 0.5) etrafinda her polygon noktasini dondur."""
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    cx, cy = 0.5, 0.5
    new_pts = []
    for x, y in pts:
        dx, dy = x - cx, y - cy
        nx = cx + dx * cos_a - dy * sin_a
        ny = cy + dx * sin_a + dy * cos_a
        # Goruntu siniri disina cikmasin
        nx = float(np.clip(nx, 0.0, 1.0))
        ny = float(np.clip(ny, 0.0, 1.0))
        new_pts.append((nx, ny))
    return new_pts


# ── Augmentation fonksiyonlari ───────────────────────────────────────────────

def aug_flip_h(img, instances):
    new_inst = [(cls, flip_pts_h(pts)) for cls, pts in instances]
    return cv2.flip(img, 1), new_inst

def aug_flip_v(img, instances):
    new_inst = [(cls, flip_pts_v(pts)) for cls, pts in instances]
    return cv2.flip(img, 0), new_inst

def aug_brightness(img, instances):
    f = np.random.uniform(0.7, 1.3)
    return np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8), instances

def aug_contrast(img, instances):
    mean = img.mean()
    f = np.random.uniform(0.8, 1.2)
    return np.clip((img.astype(np.float32) - mean) * f + mean, 0, 255).astype(np.uint8), instances

def aug_noise(img, instances):
    sigma = np.random.uniform(2, 8)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8), instances

def aug_rotate(img, instances):
    angle = np.random.uniform(-15, 15)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
    new_inst = [(cls, rotate_pts(pts, angle)) for cls, pts in instances]
    return rotated, new_inst

AUG_FUNCS = [aug_flip_h, aug_flip_v, aug_brightness,
             aug_contrast, aug_noise, aug_rotate]
NEG_FUNCS = [aug_flip_h, aug_brightness, aug_noise]


# ── Val kopyala ──────────────────────────────────────────────────────────────

print("Val seti kopyalaniyor (augmentation yok)...")
val_src   = "synthetic_dataset_yolo_split/val"
val_count = 0
for fname in os.listdir(os.path.join(val_src, "images")):
    if not (fname.endswith(".png") or fname.endswith(".jpg")):
        continue
    shutil.copy2(os.path.join(val_src, "images", fname),
                 os.path.join(DST_DIR, "val", "images", fname))
    lbl = fname.replace(".png", ".txt").replace(".jpg", ".txt")
    src_lbl = os.path.join(val_src, "labels", lbl)
    dst_lbl = os.path.join(DST_DIR, "val", "labels", lbl)
    if os.path.exists(src_lbl):
        shutil.copy2(src_lbl, dst_lbl)
    else:
        open(dst_lbl, "w").close()
    val_count += 1
print(f"  {val_count} val goruntu kopyalandi")


# ── Train orijinallerini kopyala ─────────────────────────────────────────────

print("\nTrain orijinalleri kopyalaniyor...")
orig_count = 0
for fname in os.listdir(os.path.join(SRC_DIR, "images")):
    if not (fname.endswith(".png") or fname.endswith(".jpg")):
        continue
    shutil.copy2(os.path.join(SRC_DIR, "images", fname),
                 os.path.join(DST_DIR, "train", "images", fname))
    lbl = fname.replace(".png", ".txt").replace(".jpg", ".txt")
    src_lbl = os.path.join(SRC_DIR, "labels", lbl)
    dst_lbl = os.path.join(DST_DIR, "train", "labels", lbl)
    if os.path.exists(src_lbl):
        shutil.copy2(src_lbl, dst_lbl)
    else:
        open(dst_lbl, "w").close()
    orig_count += 1
print(f"  {orig_count} train goruntu kopyalandi")


# ── Augmentation ─────────────────────────────────────────────────────────────

print(f"\nAugmentation uygulanıyor (train, her goruntu icin {N_AUG} versiyon)...")
aug_count   = 0
image_files = [f for f in os.listdir(os.path.join(SRC_DIR, "images"))
               if f.endswith(".png") or f.endswith(".jpg")]

for fname in image_files:
    is_negative = fname.startswith("negative_")

    img_path = os.path.join(SRC_DIR, "images", fname)
    lbl_path = os.path.join(SRC_DIR, "labels",
                            fname.replace(".png", ".txt").replace(".jpg", ".txt"))

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    instances = read_seg_labels(lbl_path)

    funcs  = NEG_FUNCS if is_negative else AUG_FUNCS
    chosen = random.sample(funcs, min(N_AUG, len(funcs)))

    for j, aug_fn in enumerate(chosen):
        aug_img, aug_inst = aug_fn(img.copy(), list(instances))

        base     = fname.replace(".png", "").replace(".jpg", "")
        new_name = f"{base}_aug{j+1}.png"
        new_lbl  = f"{base}_aug{j+1}.txt"

        cv2.imwrite(os.path.join(DST_DIR, "train", "images", new_name), aug_img)
        write_seg_labels(os.path.join(DST_DIR, "train", "labels", new_lbl), aug_inst)
        aug_count += 1

train_total = orig_count + aug_count
print(f"  {aug_count} augmented goruntu uretildi")
print()
print("SONUC:")
print(f"  train/: {train_total} goruntu ({orig_count} orijinal + {aug_count} augmented)")
print(f"  val/  : {val_count} goruntu (temiz, augmentation yok)")
print(f"  Toplam: {train_total + val_count} goruntu")
print(f"\nCikti: {DST_DIR}/")