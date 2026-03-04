import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

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
# Maskeyi büyüt - daha geniş alan temizlensin
kernel_inpaint = np.ones((15,15), np.uint8)
mask_big = cv2.dilate(mask, kernel_inpaint, iterations=2)
arka_plan = cv2.inpaint(inner, mask_big, 10, cv2.INPAINT_TELEA)

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

def normalize_kspace(image):
    """1. Normalizasyon - k-space maksimum değere böl"""
    image = image.astype(np.float32)
    max_val = np.max(np.abs(image))
    if max_val > 0:
        image = image / max_val
    return image

def random_translation(image, max_shift=20):
    """2. Rastgele 2D öteleme"""
    h, w = image.shape
    tx = np.random.randint(-max_shift, max_shift)
    ty = np.random.randint(-max_shift, max_shift)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(image, M, (w, h))
    return translated, tx, ty

def random_flip(image):
    """3. X ekseninde rastgele çevirme"""
    if np.random.rand() > 0.5:
        return cv2.flip(image, 1), True
    return image, False

def random_crop_with_noise(image, crop_size=80):
    """4. Rastgele kırpma + Gaussian gürültü"""
    h, w = image.shape
    if h < crop_size or w < crop_size:
        crop_size = min(h, w) - 10
    
    x = np.random.randint(0, w - crop_size)
    y = np.random.randint(0, h - crop_size)
    cropped = image[y:y+crop_size, x:x+crop_size]
    
    # Orijinal görüntü gürültüsüne eşit Gaussian gürültü
    noise = np.random.normal(0, 0.02, cropped.shape).astype(np.float32)
    noisy = np.clip(cropped + noise, 0, 1)
    return noisy

def sinusoidal_contrast_map(image):
    """5. Sinüzoidal kontrast haritası - doku heterojenliğini taklit eder"""
    h, w = image.shape
    
    # Periyot 0 ile 4π arasında rastgele
    period = np.random.uniform(0, 4 * np.pi)
    
    # Açı 0 ile 90° arasında rastgele
    angle = np.random.uniform(0, np.pi / 2)
    
    # Koordinat ızgarası
    x = np.linspace(0, period, w)
    y = np.linspace(0, period, h)
    xx, yy = np.meshgrid(x, y)
    
    # Döndürülmüş sinüzoidal harita
    contrast_map = 0.85 + 0.15 * np.sin(xx * np.cos(angle) + yy * np.sin(angle))
    
    # Görüntüyle çarp
    result = image * contrast_map.astype(np.float32)
    return np.clip(result, 0, 1)

def generate_synthetic_data(background, magnet_patch, num_samples=50, output_dir="synthetic"):
    """Tüm augmentation'ları birleştirip sentetik veri üret"""
    os.makedirs(output_dir, exist_ok=True)
    
    bg_h, bg_w = background.shape
    mg_h, mg_w = magnet_patch.shape
    
    labels = []
    
    for i in range(num_samples):
        # Arka planı normalize et
        bg = normalize_kspace(background.copy().astype(np.float32))
        
        # Mıknatısı rastgele konuma yerleştir
        max_x = bg_w - mg_w - 10
        max_y = bg_h - mg_h - 10
        pos_x = np.random.randint(10, max_x)
        pos_y = np.random.randint(10, max_y)
        
        # Mıknatısı yapıştır
        synthetic = bg.copy()
        mg_norm = normalize_kspace(magnet_patch.astype(np.float32))
        
        # Sadece koyu pikselleri yapıştır 
        region = synthetic[pos_y:pos_y+mg_h, pos_x:pos_x+mg_w].copy()
        mg_mask = mg_norm < 0.6  # eşiği düşürdük
        # Yumuşak geçiş için blend
        alpha = np.where(mg_mask, 0.85, 0.0).astype(np.float32)
        region = region * (1 - alpha) + mg_norm * alpha
        synthetic[pos_y:pos_y+mg_h, pos_x:pos_x+mg_w] = region
        
        # Flip uygula
        synthetic, flipped = random_flip(synthetic)
        if flipped:
            pos_x = bg_w - pos_x - mg_w
        
        # Sinüzoidal kontrast haritası uygula
        synthetic = sinusoidal_contrast_map(synthetic)
        
        # Merkez koordinatı hesapla
        cx = pos_x + mg_w // 2
        cy = pos_y + mg_h // 2
        
        # Kaydet
        save_img = (synthetic * 255).astype(np.uint8)
        filename = f"synthetic_{i+1:04d}.png"
        cv2.imwrite(os.path.join(output_dir, filename), save_img)
        labels.append({"dosya": filename, "x": cx, "y": cy})
        
        if (i+1) % 10 == 0:
            print(f"{i+1}/{num_samples} görüntü üretildi...")
    
    # Etiketleri kaydet
    import csv
    with open(os.path.join(output_dir, "labels.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dosya", "x", "y"])
        writer.writeheader()
        writer.writerows(labels)
    
    print(f"\nTamamlandı! {num_samples} sentetik görüntü '{output_dir}' klasörüne kaydedildi.")
    return labels

# Arka plan ve mıknatıs desenini yükle
background = cv2.imread("arka_plan.png", cv2.IMREAD_GRAYSCALE)
magnet = cv2.imread("miknatıs_desen.png", cv2.IMREAD_GRAYSCALE)

print(f"Arka plan boyutu: {background.shape}")
print(f"Mıknatıs deseni boyutu: {magnet.shape}")

# 50 sentetik görüntü üret
labels = generate_synthetic_data(background, magnet, num_samples=50)

# İlk 4 sonucu göster
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for idx in range(8):
    img = cv2.imread(f"synthetic/synthetic_{idx+1:04d}.png", cv2.IMREAD_GRAYSCALE)
    row, col = idx // 4, idx % 4
    axes[row][col].imshow(img, cmap='gray')
    axes[row][col].set_title(f"x={labels[idx]['x']}, y={labels[idx]['y']}")
    axes[row][col].axis('off')
plt.suptitle("Sentetik Görüntüler - Data Augmentation")
plt.tight_layout()
plt.show()