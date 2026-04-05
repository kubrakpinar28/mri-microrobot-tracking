import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

img = cv2.imread("translated_random_dark_1.png", cv2.IMREAD_GRAYSCALE)

# STEP 1: Crop white background region
_, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
biggest = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(biggest)
cropped = img[y:y+h, x:x+w]

# STEP 2: Extract inner region (remove edge artifacts)
margin = 25
inner = cropped[margin:h-margin, margin:w-margin]

# STEP 3: Convert to HSV and extract V channel mask
bgr = cv2.cvtColor(inner, cv2.COLOR_GRAY2BGR)
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
_, _, v_ch = cv2.split(hsv)
_, mask = cv2.threshold(v_ch, 80, 255, cv2.THRESH_BINARY_INV)

# STEP 4: Merge magnet parts using dilation
kernel = np.ones((7, 7), np.uint8)
mask_dilated = cv2.dilate(mask, kernel, iterations=1)

# STEP 5: Find all contours and compute bounding box covering all parts
contours2, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Take the 3 largest contours (top circle + middle bar + bottom circle)
contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)[:3]

# Compute single bounding box covering all contours
all_x, all_y, all_w, all_h = [], [], [], []
for c in contours2:
    cx, cy, cw, ch = cv2.boundingRect(c)
    all_x.append(cx)
    all_y.append(cy)
    all_w.append(cx + cw)
    all_h.append(cy + ch)

# Min/max bounding box
final_x = min(all_x) - 5
final_y = min(all_y) - 5
final_w = max(all_w) - final_x + 5
final_h = max(all_h) - final_y + 5

# STEP 6: Extract magnet patch and clean background using inpainting
magnet_patch = inner[final_y:final_y+final_h, final_x:final_x+final_w]

# Expand mask for cleaner inpainting
kernel_inpaint = np.ones((15, 15), np.uint8)
mask_big = cv2.dilate(mask, kernel_inpaint, iterations=2)
background = cv2.inpaint(inner, mask_big, 10, cv2.INPAINT_TELEA)

print(f"Magnet size: {final_w}x{final_h} pixels")

# Save results
cv2.imwrite("magnet_pattern.png", magnet_patch)
cv2.imwrite("arka_plan.png", background)

# Visualize
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
axes[0].imshow(inner, cmap='gray')
axes[0].set_title("Inner Region")
axes[1].imshow(mask, cmap='gray')
axes[1].set_title("HSV Mask")
axes[2].imshow(background, cmap='gray')
axes[2].set_title("Background")
axes[3].imshow(magnet_patch, cmap='gray')
axes[3].set_title("Magnet Pattern")
plt.tight_layout()
plt.show()


def normalize_kspace(image):
    """1. Normalization - divide by maximum absolute value in k-space"""
    image = image.astype(np.float32)
    max_val = np.max(np.abs(image))
    if max_val > 0:
        image = image / max_val
    return image

def random_translation(image, max_shift=20):
    """2. Random 2D translation"""
    h, w = image.shape
    tx = np.random.randint(-max_shift, max_shift)
    ty = np.random.randint(-max_shift, max_shift)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(image, M, (w, h))
    return translated, tx, ty

def random_flip(image):
    """3. Random flip around x-axis"""
    if np.random.rand() > 0.5:
        return cv2.flip(image, 1), True
    return image, False

def random_crop_with_noise(image, crop_size=80):
    """4. Random crop with Gaussian noise injection"""
    h, w = image.shape
    if h < crop_size or w < crop_size:
        crop_size = min(h, w) - 10
    x = np.random.randint(0, w - crop_size)
    y = np.random.randint(0, h - crop_size)
    cropped = image[y:y+crop_size, x:x+crop_size]
    noise = np.random.normal(0, 0.02, cropped.shape).astype(np.float32)
    noisy = np.clip(cropped + noise, 0, 1)
    return noisy

def sinusoidal_contrast_map(image):
    """5. Sinusoidal contrast map - mimics real tissue heterogeneity"""
    h, w = image.shape
    period = np.random.uniform(0, 4 * np.pi)
    angle = np.random.uniform(0, np.pi / 2)
    x = np.linspace(0, period, w)
    y = np.linspace(0, period, h)
    xx, yy = np.meshgrid(x, y)
    contrast_map = 0.85 + 0.15 * np.sin(xx * np.cos(angle) + yy * np.sin(angle))
    result = image * contrast_map.astype(np.float32)
    return np.clip(result, 0, 1)

def generate_synthetic_data(background, magnet_patch, num_samples=50, output_dir="synthetic"):
    """Generate synthetic training data using all augmentation techniques"""
    os.makedirs(output_dir, exist_ok=True)

    bg_h, bg_w = background.shape
    mg_h, mg_w = magnet_patch.shape

    labels = []

    for i in range(num_samples):
        bg = normalize_kspace(background.copy().astype(np.float32))

        max_x = bg_w - mg_w - 10
        max_y = bg_h - mg_h - 10
        pos_x = np.random.randint(10, max_x)
        pos_y = np.random.randint(10, max_y)

        synthetic = bg.copy()
        mg_norm = normalize_kspace(magnet_patch.astype(np.float32))

        region = synthetic[pos_y:pos_y+mg_h, pos_x:pos_x+mg_w].copy()
        mg_mask = mg_norm < 0.6
        alpha = np.where(mg_mask, 0.85, 0.0).astype(np.float32)
        region = region * (1 - alpha) + mg_norm * alpha
        synthetic[pos_y:pos_y+mg_h, pos_x:pos_x+mg_w] = region

        synthetic, flipped = random_flip(synthetic)
        if flipped:
            pos_x = bg_w - pos_x - mg_w

        noise = np.random.normal(0, 0.02, synthetic.shape).astype(np.float32)
        synthetic = np.clip(synthetic + noise, 0, 1)

        synthetic = sinusoidal_contrast_map(synthetic)

        cx = pos_x + mg_w // 2
        cy = pos_y + mg_h // 2

        save_img = (synthetic * 255).astype(np.uint8)
        filename = f"synthetic_{i+1:04d}.png"
        cv2.imwrite(os.path.join(output_dir, filename), save_img)
        labels.append({"file": filename, "x": cx, "y": cy})

        if (i + 1) % 10 == 0:
            print(f"{i+1}/{num_samples} images generated...")

    import csv
    with open(os.path.join(output_dir, "labels.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "x", "y"])
        writer.writeheader()
        writer.writerows(labels)

    print(f"\nDone! {num_samples} synthetic images saved to '{output_dir}' folder.")
    return labels


# Load background and magnet patch
bg_img   = cv2.imread("arka_plan.png",      cv2.IMREAD_GRAYSCALE)
magnet   = cv2.imread("magnet_pattern.png", cv2.IMREAD_GRAYSCALE)

print(f"Background size: {bg_img.shape}")
print(f"Magnet patch size: {magnet.shape}")

# Generate 50 synthetic images
labels = generate_synthetic_data(bg_img, magnet, num_samples=50)

# Visualize first 8 results
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for idx in range(8):
    img_s = cv2.imread(f"synthetic/synthetic_{idx+1:04d}.png", cv2.IMREAD_GRAYSCALE)
    row, col = idx // 4, idx % 4
    axes[row][col].imshow(img_s, cmap='gray')
    axes[row][col].set_title(f"x={labels[idx]['x']}, y={labels[idx]['y']}")
    axes[row][col].axis('off')
plt.suptitle("Synthetic Images - Data Augmentation")
plt.tight_layout()
plt.show()

# Visualize magnet pattern on black background
m = cv2.imread("magnet_pattern.png", cv2.IMREAD_GRAYSCALE)
_, m_mask = cv2.threshold(m, 150, 255, cv2.THRESH_BINARY_INV)
m_black = cv2.bitwise_and(m, m, mask=m_mask)

plt.figure()
plt.imshow(m_black, cmap='gray')
plt.title("Magnet Pattern - Black Background")
plt.axis('off')
plt.show()