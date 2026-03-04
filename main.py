import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("translated_random_dark_1.png", cv2.IMREAD_GRAYSCALE)

# 1. ADIM: Beyaz alanı kırptık
_, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
biggest = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(biggest)
cropped = img[y:y+h, x:x+w]

# 2. ADIM: İç bölge
margin = 25
inner = cropped[margin:h-margin, margin:w-margin]

# 3. ADIM: HSV + Maske
bgr = cv2.cvtColor(inner, cv2.COLOR_GRAY2BGR)
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
_, _, v_ch = cv2.split(hsv)
_, mask = cv2.threshold(v_ch, 80, 255, cv2.THRESH_BINARY_INV)

# 4. ADIM: Parçaları birleştir - küçük kernel
kernel = np.ones((7,7), np.uint8)  # küçülttük
mask_dilated = cv2.dilate(mask, kernel, iterations=1)

# 5. ADIM: Tüm konturları al, hepsini kapsayan box bul
contours2, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# En büyük 3 konturu al (üst yuvarlak + orta + alt yuvarlak)
contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)[:3]

# Hepsini kapsayan tek bounding box
all_x, all_y, all_w, all_h = [], [], [], []
for c in contours2:
    cx, cy, cw, ch = cv2.boundingRect(c)
    all_x.append(cx)
    all_y.append(cy)
    all_w.append(cx + cw)
    all_h.append(cy + ch)

# Min/max ile hepsini kapsayan kutu
final_x = min(all_x) - 5
final_y = min(all_y) - 5
final_w = max(all_w) - final_x + 5
final_h = max(all_h) - final_y + 5

# 6. ADIM: Kopar
miknatıs = inner[final_y:final_y+final_h, final_x:final_x+final_w]
arka_plan = cv2.inpaint(inner, mask, 5, cv2.INPAINT_TELEA)

print(f"Mıknatıs boyutu: {final_w}x{final_h} piksel")

# Kaydet
cv2.imwrite("miknatıs_desen.png", miknatıs)
cv2.imwrite("arka_plan.png", arka_plan)

# Görselleştir
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
axes[0].imshow(inner, cmap='gray')
axes[0].set_title("İç Bölge")
axes[1].imshow(mask, cmap='gray')
axes[1].set_title("Orijinal Maske")
axes[2].imshow(arka_plan, cmap='gray')
axes[2].set_title("Arka Plan")
axes[3].imshow(miknatıs, cmap='gray')
axes[3].set_title("Mıknatıs Deseni")
plt.tight_layout()
plt.show()