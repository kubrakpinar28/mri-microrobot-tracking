# WARNING:
# This script deletes OUTPUT_DIR before creating dataset.
# Make sure OUTPUT_DIR is correct.

import os
import shutil
import glob
import random

OUTPUT_DIR = r"C:\Users\kmfm2\Downloads\all_mri"

if os.path.basename(OUTPUT_DIR.rstrip("/\\")) != "all_mri":
    raise ValueError(f"Safety check failed: Unexpected directory is about to be deleted -> {OUTPUT_DIR}")

if os.path.exists(OUTPUT_DIR):
    print("Removing existing dataset...")
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

sources = [
    r"C:\Users\kmfm2\Downloads\archive (1)\MR-ART Dataset\Standard\sagittal",
    r"C:\Users\kmfm2\Downloads\heart_png",
    r"C:\Users\kmfm2\Downloads\archive (8)\Breast Cancer Patients MRI's\train\Healthy",
    r"C:\Users\kmfm2\Downloads\knee_png",
]

count = 0
for src in sources:
    files = glob.glob(os.path.join(src, "*.png")) + \
            glob.glob(os.path.join(src, "*.jpg"))

    # Randomly sample up to 5000 knee MRI images for each run
    if "knee_png" in src:
        files = random.sample(files, min(5000, len(files)))

    folder_name = os.path.basename(src)
    for f in files:
        ext = os.path.splitext(f)[1]
        new_name = f"{folder_name}_{count:05d}{ext}"
        shutil.copy(f, os.path.join(OUTPUT_DIR, new_name))
        count += 1
    print(f"{folder_name}: {len(files)} files copied")

print(f"\nToplam {count} dosya -> {OUTPUT_DIR}")