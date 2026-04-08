"""
Train/Val Split
===============
synthetic_dataset_yolo_filtered/ klasorunu
%80 train / %20 val olarak boler.

Augmentation SADECE train klasorune uygulanacak.
Val seti temiz kalir — veri sizintisi olmaz.

Kullanim:
  py split_dataset.py
"""

import os
import shutil
import random

SRC_DIR = "synthetic_dataset_yolo_filtered"
DST_DIR = "synthetic_dataset_yolo_split"
TRAIN_RATIO = 0.80
SEED = 42

random.seed(SEED)

# Eski split dataset varsa sil
if os.path.exists(DST_DIR):
    shutil.rmtree(DST_DIR)

# Klasorler
for split in ["train", "val"]:
    for folder in ["images", "labels"]:
        os.makedirs(os.path.join(DST_DIR, split, folder), exist_ok=True)

# Tum goruntuleri listele
all_images = [
    f for f in os.listdir(os.path.join(SRC_DIR, "images"))
    if f.endswith(".png") or f.endswith(".jpg")
]

# Negatif ve pozitifi ayir — ayri shuffle, sonra birlestir
# Boylece her iki grup da train/val'e dengeli dagilir
positives = [f for f in all_images if not f.startswith("negative_")]
negatives = [f for f in all_images if f.startswith("negative_")]

random.shuffle(positives)
random.shuffle(negatives)

def split_list(lst, ratio):
    n = int(len(lst) * ratio)
    return lst[:n], lst[n:]

pos_train, pos_val = split_list(positives, TRAIN_RATIO)
neg_train, neg_val = split_list(negatives, TRAIN_RATIO)

train_files = pos_train + neg_train
val_files = pos_val + neg_val

print(f"Toplam: {len(all_images)} goruntu")
print(f"  Pozitif: {len(positives)} → train: {len(pos_train)}, val: {len(pos_val)}")
print(f"  Negatif: {len(negatives)} → train: {len(neg_train)}, val: {len(neg_val)}")
print(f"  Train toplam: {len(train_files)}")
print(f"  Val toplam  : {len(val_files)}")
print()

def copy_split(file_list, split_name):
    copied = 0
    missing_lbl = 0

    for fname in file_list:
        src_img = os.path.join(SRC_DIR, "images", fname)
        src_lbl = os.path.join(
            SRC_DIR,
            "labels",
            fname.replace(".png", ".txt").replace(".jpg", ".txt")
        )

        dst_img = os.path.join(DST_DIR, split_name, "images", fname)
        dst_lbl = os.path.join(
            DST_DIR,
            split_name,
            "labels",
            fname.replace(".png", ".txt").replace(".jpg", ".txt")
        )

        shutil.copy2(src_img, dst_img)

        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)
        else:
            # Bos label olustur (negatif ornekler icin)
            open(dst_lbl, "w").close()
            missing_lbl += 1

        copied += 1

    print(f"  {split_name}: {copied} goruntu kopyalandi (label eksik: {missing_lbl})")
    return copied

copy_split(train_files, "train")
copy_split(val_files, "val")

print()
print(f"Split tamamlandi: {DST_DIR}/")
print(f"  train/images/ → {len(train_files)} goruntu")
print(f"  val/images/   → {len(val_files)} goruntu")
