"""
Veri Artirma (Data Augmentation) — YOLO Detection BBox
======================================================

Label formati:
    class_id x_center y_center width height

Bu sürüm segmentation/polygon label kullanmaz.
Rotation uygulanmaz; böylece bbox aspect ratio bozulmaz.
Sadece train setine augmentation uygulanır.
Val/test setleri temiz kalır.

Çalıştırma:
    python augment_dataset.py
"""

import os
import cv2
import numpy as np
import shutil
import random

# Öncelik sırası: önce filtrelenmiş dataset, yoksa doğrudan YOLO dataset.
# Eğer elinde hazır splitli klasör varsa SRC_DIR'i elle ona da çevirebilirsin.
SRC_DIR = "synthetic_dataset_yolo_filtered"
DST_DIR = "synthetic_dataset_yolo_augmented"

N_AUG = 2
SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(SEED)
np.random.seed(SEED)


def ensure_src_dir():
    """Varsayılan filtrelenmiş klasör yoksa alternatifleri dene."""
    global SRC_DIR
    candidates = [SRC_DIR, "synthetic_dataset_yolo", "synthetic_dataset_yolo_split"]
    for c in candidates:
        if os.path.exists(c):
            SRC_DIR = c
            return
    raise FileNotFoundError(
        "Kaynak dataset bulunamadı. Beklenenlerden biri olmalı: "
        "synthetic_dataset_yolo_filtered / synthetic_dataset_yolo / synthetic_dataset_yolo_split"
    )


def reset_output_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)

    for folder in [
        "train/images", "train/labels",
        "val/images", "val/labels",
        "test/images", "test/labels",
    ]:
        os.makedirs(os.path.join(path, folder), exist_ok=True)


def read_bbox_labels(label_path):
    """
    YOLO detection label oku.
    Her satır kesinlikle 5 eleman olmalı:
        class_id x_center y_center width height
    """
    labels = []

    if not os.path.exists(label_path):
        return labels

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                print(f"[UYARI] Detection formatı değil, atlandı: {label_path} -> {line}")
                continue

            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])

            # Güvenlik: değerleri normalize aralıkta tut.
            x = float(np.clip(x, 0.0, 1.0))
            y = float(np.clip(y, 0.0, 1.0))
            w = float(np.clip(w, 0.0, 1.0))
            h = float(np.clip(h, 0.0, 1.0))

            labels.append([cls, x, y, w, h])

    return labels


def write_bbox_labels(label_path, labels):
    with open(label_path, "w", encoding="utf-8") as f:
        for cls, x, y, w, h in labels:
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def aug_flip_h(img, labels):
    """Yatay flip: x_center -> 1 - x_center. w/h değişmez."""
    new_labels = []
    for cls, x, y, w, h in labels:
        new_labels.append([cls, 1.0 - x, y, w, h])
    return cv2.flip(img, 1), new_labels


def aug_flip_v(img, labels):
    """Dikey flip: y_center -> 1 - y_center. w/h değişmez."""
    new_labels = []
    for cls, x, y, w, h in labels:
        new_labels.append([cls, x, 1.0 - y, w, h])
    return cv2.flip(img, 0), new_labels


def aug_brightness(img, labels):
    factor = np.random.uniform(0.75, 1.25)
    out = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return out, labels


def aug_contrast(img, labels):
    mean = img.mean()
    factor = np.random.uniform(0.85, 1.15)
    out = np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
    return out, labels


def aug_noise(img, labels):
    sigma = np.random.uniform(2, 6)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out, labels


AUG_FUNCS = [aug_flip_h, aug_flip_v, aug_brightness, aug_contrast, aug_noise]
NEG_FUNCS = [aug_flip_h, aug_brightness, aug_noise]


def image_to_label_name(fname):
    base, _ = os.path.splitext(fname)
    return base + ".txt"


def list_images(img_dir):
    if not os.path.exists(img_dir):
        return []
    return sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])


def copy_pair(fname, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir):
    src_img = os.path.join(src_img_dir, fname)
    lbl_name = image_to_label_name(fname)
    src_lbl = os.path.join(src_lbl_dir, lbl_name)

    shutil.copy2(src_img, os.path.join(dst_img_dir, fname))

    dst_lbl = os.path.join(dst_lbl_dir, lbl_name)
    if os.path.exists(src_lbl):
        shutil.copy2(src_lbl, dst_lbl)
    else:
        open(dst_lbl, "w", encoding="utf-8").close()


def has_existing_split(src_dir):
    return os.path.exists(os.path.join(src_dir, "train", "images"))


def split_flat_dataset(src_dir):
    """Flat klasör yapısı: images/ labels/ -> train/val/test listeleri."""
    src_img_dir = os.path.join(src_dir, "images")
    src_lbl_dir = os.path.join(src_dir, "labels")

    image_files = list_images(src_img_dir)
    random.shuffle(image_files)

    n_total = len(image_files)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]

    return {
        "train": (src_img_dir, src_lbl_dir, train_files),
        "val": (src_img_dir, src_lbl_dir, val_files),
        "test": (src_img_dir, src_lbl_dir, test_files),
    }


def read_existing_split(src_dir):
    """Hazır splitli klasör yapısı: train/images, val/images, test/images."""
    split_data = {}
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(src_dir, split, "images")
        lbl_dir = os.path.join(src_dir, split, "labels")
        files = list_images(img_dir)
        split_data[split] = (img_dir, lbl_dir, files)
    return split_data


def augment_train_file(fname, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir):
    is_negative = fname.startswith("negative_")

    img_path = os.path.join(src_img_dir, fname)
    lbl_path = os.path.join(src_lbl_dir, image_to_label_name(fname))

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0

    labels = read_bbox_labels(lbl_path)

    funcs = NEG_FUNCS if is_negative else AUG_FUNCS
    chosen = random.sample(funcs, min(N_AUG, len(funcs)))

    base = os.path.splitext(fname)[0]
    made = 0

    for j, aug_fn in enumerate(chosen):
        aug_img, aug_labels = aug_fn(img.copy(), [l.copy() for l in labels])

        new_img_name = f"{base}_aug{j+1}.png"
        new_lbl_name = f"{base}_aug{j+1}.txt"

        cv2.imwrite(os.path.join(dst_img_dir, new_img_name), aug_img)
        write_bbox_labels(os.path.join(dst_lbl_dir, new_lbl_name), aug_labels)
        made += 1

    return made


def main():
    ensure_src_dir()
    reset_output_dir(DST_DIR)

    if has_existing_split(SRC_DIR):
        print(f"Kaynak splitli dataset olarak algılandı: {SRC_DIR}")
        split_data = read_existing_split(SRC_DIR)
    else:
        print(f"Kaynak flat dataset olarak algılandı, burada split yapılacak: {SRC_DIR}")
        split_data = split_flat_dataset(SRC_DIR)

    print("Split dağılımı:")
    for split, (_, _, files) in split_data.items():
        print(f"  {split}: {len(files)}")

    # Orijinalleri kopyala
    for split, (src_img_dir, src_lbl_dir, files) in split_data.items():
        for fname in files:
            copy_pair(
                fname,
                src_img_dir,
                src_lbl_dir,
                os.path.join(DST_DIR, split, "images"),
                os.path.join(DST_DIR, split, "labels"),
            )

    # Sadece train augmentation
    train_img_dir, train_lbl_dir, train_files = split_data["train"]
    dst_train_img = os.path.join(DST_DIR, "train", "images")
    dst_train_lbl = os.path.join(DST_DIR, "train", "labels")

    aug_count = 0
    for fname in train_files:
        aug_count += augment_train_file(
            fname,
            train_img_dir,
            train_lbl_dir,
            dst_train_img,
            dst_train_lbl,
        )

    print()
    print("Augmentation tamamlandı.")
    print(f"  Kaynak: {SRC_DIR}")
    print(f"  Çıktı : {DST_DIR}")
    print(f"  Augmented train görüntü: {aug_count}")
    print()
    print("Not: Rotation kullanılmadı; bbox w/h oranı korunur.")


if __name__ == "__main__":
    main()
