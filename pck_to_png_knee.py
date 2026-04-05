"""
KneeMRI .pck -> PNG Converter
"""

import pickle
import numpy as np
import cv2
import os
import glob

INPUT_FOLDER  = r"C:\Users\kmfm2\Downloads\archive (9)"
OUTPUT_FOLDER = r"C:\Users\kmfm2\Downloads\knee_png"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# vol01 - vol08 klasorlerindeki tum .pck dosyalarini bul
pck_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "**", "*.pck"), recursive=True))

print(f"{len(pck_files)} adet .pck dosyasi bulundu")

total_png = 0
for pck_path in pck_files:
    base_name = os.path.splitext(os.path.basename(pck_path))[0]

    try:
        with open(pck_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    except Exception as e:
        try:
            with open(pck_path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e2:
            print(f"  Atlandi: {base_name} ({e2})")
            continue

    # Veri tipine gore isle
    if isinstance(data, np.ndarray):
        vol = data
    elif isinstance(data, dict):
        vol = None
        for key in ['data', 'volume', 'img', 'image', 'array']:
            if key in data and isinstance(data[key], np.ndarray):
                vol = data[key]
                break
        if vol is None:
            for v in data.values():
                if isinstance(v, np.ndarray) and v.ndim >= 2:
                    vol = v
                    break
    else:
        print(f"  Bilinmeyen format: {base_name} ({type(data)})")
        continue

    if vol is None:
        print(f"  Volume bulunamadi: {base_name}")
        continue

    # Slice'lara bol
    if vol.ndim == 2:
        slices = [vol]
    elif vol.ndim == 3:
        # En kucuk boyut slice ekseni mi?
        if vol.shape[2] <= min(vol.shape[0], vol.shape[1]):
            slices = [vol[:, :, i] for i in range(vol.shape[2])]
        else:
            slices = [vol[i] for i in range(vol.shape[0])]
    else:
        print(f"  Beklenmeyen boyut: {base_name} {vol.shape}")
        continue

    slice_count = 0
    for i, sl in enumerate(slices):
        sl = sl.astype(np.float32)
        mn, mx = sl.min(), sl.max()
        if mx > mn:
            sl_norm = ((sl - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            sl_norm = np.zeros_like(sl, dtype=np.uint8)

        # Cok karanlik slice'lari atla
        if sl_norm.mean() < 8:
            continue

        out_path = os.path.join(OUTPUT_FOLDER, f"knee_{base_name}_slice{i:03d}.png")
        cv2.imwrite(out_path, sl_norm)
        slice_count += 1

    total_png += slice_count
    print(f"  {base_name}: {slice_count} slice")

print(f"\nBitti! Toplam {total_png} PNG -> {OUTPUT_FOLDER}")