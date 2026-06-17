import os
import glob
import cv2

DATASET_DIR = "synthetic_dataset_yolo_augmented"

splits = ["train", "val", "test"]

total = 0
bad_format = 0
bad_value = 0
bad_pixel_ar = 0

for split in splits:
    img_dir = os.path.join(DATASET_DIR, split, "images")
    lbl_dir = os.path.join(DATASET_DIR, split, "labels")

    for lbl_path in glob.glob(os.path.join(lbl_dir, "*.txt")):
        img_name = os.path.basename(lbl_path).replace(".txt", ".png")
        img_path = os.path.join(img_dir, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img_h, img_w = img.shape

        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()

                if not parts:
                    continue

                total += 1

                if len(parts) != 5:
                    bad_format += 1
                    print("Format hatası:", lbl_path, line.strip())
                    continue

                cls, x, y, w, h = parts
                x, y, w, h = map(float, [x, y, w, h])

                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    bad_value += 1
                    print("Değer hatası:", lbl_path, line.strip())
                    continue

                # Piksel karşılığı kare mi?
                w_px = w * img_w
                h_px = h * img_h

                if abs(w_px - h_px) > 1.5:
                    bad_pixel_ar += 1
                    print("Pixel AR hatası:", lbl_path, "w_px=", w_px, "h_px=", h_px)

print("Toplam label:", total)
print("Format hatası:", bad_format)
print("Değer hatası:", bad_value)
print("Pixel aspect ratio hatası:", bad_pixel_ar)