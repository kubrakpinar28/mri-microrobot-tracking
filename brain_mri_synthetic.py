import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import csv

# ── Robot size constants ───────────────────────────────────────────────────────
# Pixel spacing values (mm/px) — sources and justification:
#   brain  : MR-ART (Nárai et al., Sci Data 2022), FOV=256mm, 320px crop → 0.80 mm/px
#   heart  : Medical Decathlon Task02_Heart, literature value → 1.25 mm/px
#   breast : Kaggle Breast MRI, no DICOM metadata, literature estimate → 0.75 mm/px
#   knee   : KneeMRI Rijeka, ROI ~88px, FOV=160mm → 1.80 mm/px effective
PIXEL_SPACING_MM = {
    "brain":   0.80,
    "heart":   1.25,
    "breast":  0.75,
    "knee":    1.80,
    "default": 0.80,
}
ROBOT_DIAMETER_MM_MIN = 10.0
ROBOT_DIAMETER_MM_MAX = 20.0


def keep_largest_component(mask):
    """
    Keep only the largest connected component in a binary mask.

    This removes small isolated regions and noise, ensuring that
    subsequent processing operates on the primary anatomical structure.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.uint8) * 255


def extract_magnet_patch(magnet_img_path):
    """
    Extract and normalize the magnetic artifact template from a reference image.

    The function identifies the main artifact region, removes background
    influence, generates an artifact mask, and returns a square normalized
    patch that can later be inserted into MRI images.
    """
    img = cv2.imread(magnet_img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {magnet_img_path}")
    H, W = img.shape
    if H < 300 and W < 300:
        scale_factor = max(1, int(np.ceil(150 / max(H, W))))
        if scale_factor > 1:
            img = cv2.resize(img, (W * scale_factor, H * scale_factor),
                             interpolation=cv2.INTER_CUBIC)
            H, W = img.shape
        crop = img.astype(np.float32)
        b    = max(5, H // 8)
        edges = [crop[:b,:].mean(), crop[-b:,:].mean(),
                 crop[:,:b].mean(), crop[:,-b:].mean()]
        bg = max(float(np.mean(edges)), 20.0)
    else:
        _, ph = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
        ph    = cv2.morphologyEx(ph, cv2.MORPH_CLOSE, np.ones((20,20), np.uint8))
        contours, _ = cv2.findContours(ph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ph_filled   = np.zeros_like(img)
        cv2.drawContours(ph_filled, [max(contours, key=cv2.contourArea)], -1, 255, -1)
        ph_inner  = cv2.erode(ph_filled, np.ones((30,30), np.uint8))
        masked    = np.where(ph_inner > 0, img.astype(np.int32), 255)
        dark_mask = (masked < 90).astype(np.uint8) * 255
        dark_c, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not dark_c:
            raise ValueError("Magnet dark region not found!")
        centers = []
        for c in dark_c:
            area = cv2.contourArea(c)
            if area > 100:
                M = cv2.moments(c)
                if M['m00'] > 0:
                    centers.append((int(M['m10']/M['m00']),
                                    int(M['m01']/M['m00']), area))
        if not centers:
            centers = [(W//2, H//2, 1)]
        centers.sort(key=lambda x: -x[2])
        top     = centers[:4]
        total_a = sum(c[2] for c in top)
        mx   = int(sum(c[0]*c[2] for c in top) / total_a)
        my   = int(sum(c[1]*c[2] for c in top) / total_a)
        half = 110
        y1, y2 = max(0, my-half), min(H, my+half)
        x1, x2 = max(0, mx-half), min(W, mx+half)
        crop    = img[y1:y2, x1:x2].astype(np.float32)
        b       = 20
        edges   = [crop[:b,:].mean(), crop[:,:b].mean(), crop[:,-b:].mean()]
        bg      = max(float(np.mean(edges)), 30.0)

    patch_norm = np.clip(crop / (bg * 2.0), 0.0, 1.0).astype(np.float32)
    eff        = (np.abs(patch_norm - 0.5) > 0.07).astype(np.uint8) * 255
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    effect_mask = cv2.morphologyEx(eff, cv2.MORPH_OPEN,  k1)
    effect_mask = cv2.morphologyEx(effect_mask, cv2.MORPH_CLOSE, k2)
    effect_mask = keep_largest_component(effect_mask)

    # Convert the extracted artifact patch to a square shape
    # to preserve a fixed aspect ratio during resizing.
    pH, pW = patch_norm.shape
    side   = min(pH, pW)
    cy_p, cx_p = pH//2, pW//2
    y1_c, x1_c = cy_p - side//2, cx_p - side//2
    patch_norm  = patch_norm[y1_c:y1_c+side, x1_c:x1_c+side]
    effect_mask = effect_mask[y1_c:y1_c+side, x1_c:x1_c+side]
    center      = (side//2, side//2)

    print(f"[Magnet] BG~{bg:.0f} | Patch={patch_norm.shape} (kare) | "
          f"AR={patch_norm.shape[1]/patch_norm.shape[0]:.3f} | "
          f"Void(<0.4)={(patch_norm < 0.4).sum()} | "
          f"Halo(>0.6)={(patch_norm > 0.6).sum()}")
    return patch_norm, effect_mask, center


def get_tissue_mask(image, source_type="default"):
    """
    Generate an anatomical tissue mask for a given MRI image.

    Different MRI modalities require different thresholding and
    morphological operations. The resulting mask defines the valid
    region where synthetic artifacts may be inserted.
    """
    blurred = cv2.GaussianBlur(image, (15,15), 0)
    H, W    = image.shape

    if source_type == "heart":
        otsu_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, mask = cv2.threshold(blurred, int(otsu_val*0.35), 255, cv2.THRESH_BINARY)
        k_close = max(20, int(min(H,W)*0.09))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k_close,k_close), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5,5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        if contours:
            big_c = [c for c in contours if cv2.contourArea(c) > H*W*0.005]
            if not big_c: big_c = [max(contours, key=cv2.contourArea)]
            hull = cv2.convexHull(np.vstack(big_c))
            cv2.drawContours(filled, [hull], -1, 255, -1)
        s = max(11, int(min(H,W)*0.04)); s += s%2==0
        filled = cv2.GaussianBlur(filled, (s,s), 0)
        _, filled = cv2.threshold(filled, 127, 255, cv2.THRESH_BINARY)
        return filled

    elif source_type == "breast":
        otsu_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, mask = cv2.threshold(blurred, int(otsu_val*0.60), 255, cv2.THRESH_BINARY)
        k = max(9, int(min(H,W)*0.07)); k += k%2==0
        ellipse_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ellipse_k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            big_c = [c for c in contours if cv2.contourArea(c) > H*W*0.01]
            if not big_c: big_c = [max(contours, key=cv2.contourArea)]
            filled = np.zeros_like(mask)
            cv2.drawContours(filled, big_c, -1, 255, -1)
            s = max(21, int(min(H,W)*0.09)); s += s%2==0
            filled = cv2.GaussianBlur(filled, (s,s), 0)
            _, filled = cv2.threshold(filled, 100, 255, cv2.THRESH_BINARY)
            mask = filled
        return mask

    elif source_type == "knee":
        otsu_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, mask = cv2.threshold(blurred, int(otsu_val*0.45), 255, cv2.THRESH_BINARY)
        k_close = max(20, int(min(H,W)*0.08))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k_close,k_close), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5,5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        if contours:
            big_c = [c for c in contours if cv2.contourArea(c) > H*W*0.03]
            if not big_c: big_c = [max(contours, key=cv2.contourArea)]
            cv2.drawContours(filled, big_c, -1, 255, -1)
        s = max(11, int(min(H,W)*0.04)); s += s%2==0
        filled = cv2.GaussianBlur(filled, (s,s), 0)
        _, filled = cv2.threshold(filled, 127, 255, cv2.THRESH_BINARY)
        return keep_largest_component(filled)

    else:  # brain / default
        otsu_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, mask = cv2.threshold(blurred, int(otsu_val*0.65), 255, cv2.THRESH_BINARY)
        k = max(9, int(min(H,W)*0.08)); k += k%2==0
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k,k), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((7,7), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            filled = np.zeros_like(mask)
            cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, -1)
            s = max(15, int(min(H,W)*0.06)); s += s%2==0
            filled = cv2.GaussianBlur(filled, (s,s), 0)
            _, filled = cv2.threshold(filled, 127, 255, cv2.THRESH_BINARY)
            mask = filled
        return mask


def add_rice_noise(image_u8, sigma=12.0):
    """
    Apply Rician noise to simulate MRI acquisition noise.

    MRI magnitude images commonly exhibit Rician-distributed noise,
    making this augmentation more realistic than standard Gaussian noise.
    """
    img = image_u8.astype(np.float32)
    n1  = np.random.normal(0, sigma, img.shape).astype(np.float32)
    n2  = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(np.sqrt((img+n1)**2 + n2**2), 0, 255).astype(np.uint8)


def place_magnet(image_f32, tissue_mask,
                 patch_norm, effect_mask, patch_center,
                 cx, cy, scale=1.0, h=0.0, hthr=9.0,
                 motion_vec=None, alpha=0.92, contrast=1.0,
                 source_type="default"):
    """
    Insert a synthetic magnetic microrobot artifact into an MRI image.

    The artifact is resized according to the desired robot scale,
    blended into the surrounding tissue, and evaluated using a
    contrast-to-noise ratio (CNR) visibility metric.

    Returns:
        result      : MRI image with inserted artifact
        visible     : visibility flag based on organ-specific CNR threshold
        cnr         : computed contrast-to-noise ratio
        artifact_mask : final artifact region used for evaluation
    """
    result     = image_f32.copy()
    if h >= hthr:
        return result, False, 0.0, None
    visibility = float(np.clip(1.0 - (h/hthr)**1.2, 0.0, 1.0))

    pH, pW = patch_norm.shape          # patch zaten kare: pH == pW
    new_h  = max(20, int(pH * scale))
    new_w  = max(20, int(pW * scale))
    p  = cv2.resize(patch_norm,  (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    em = cv2.resize(effect_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    em = (em > 127).astype(bool)

    # Motion-based stretching is intentionally disabled to preserve
    # the physical aspect ratio of the artifact template.
    # motion_vec is retained only for backward compatibility.
    H, W  = result.shape
    x1i   = cx - new_w//2;  x2i = x1i + new_w
    y1i   = cy - new_h//2;  y2i = y1i + new_h
    ix1   = max(0, x1i);    ix2 = min(W, x2i)
    iy1   = max(0, y1i);    iy2 = min(H, y2i)
    px1   = ix1 - x1i;      px2 = px1 + (ix2 - ix1)
    py1   = iy1 - y1i;      py2 = py1 + (iy2 - iy1)
    if ix2 <= ix1 or iy2 <= iy1:
        return result, False, 0.0, None

    roi        = result[iy1:iy2, ix1:ix2].copy()
    p_c        = p[py1:py2, px1:px2]
    em_c       = em[py1:py2, px1:px2]
    tissue_roi = (tissue_mask[iy1:iy2, ix1:ix2] > 0)
    eff_alpha  = min(alpha * visibility * contrast * 1.3, 1.0)
    ph_r, pw_r = p_c.shape
    cy_p, cx_p = ph_r//2, pw_r//2

    yy, xx      = np.mgrid[0:ph_r, 0:pw_r].astype(np.float32)
    x_rel, y_rel = xx - cx_p, yy - cy_p
    h_px         = h / 0.94 if h > 0 else 0.5
    r_3d         = np.sqrt(x_rel**2 + y_rel**2 + h_px**2) + 1e-6   # 3D distance from the artifact center.
    dipole_mag   = np.abs(3*(x_rel/r_3d)**2 - 1) / (r_3d**2 + 1e-6) # Dipole-inspired magnetic field intensity approximation.
    dipole_norm  = np.clip(dipole_mag / (np.percentile(dipole_mag, 97)+1e-9), 0.0, 1.0)
    r_falloff    = np.exp(-r_3d / (max(ph_r, pw_r)*0.4))
    pen_str      = eff_alpha * 0.65 * dipole_norm * (0.6 + 0.4*r_falloff)
    pen_result   = np.clip(roi*(1.0-pen_str) - 0.04*dipole_norm, 0.0, 1.0)
    blended      = np.where(tissue_roi, pen_result, roi)

    local_mean  = float(roi[tissue_roi].mean()) if tissue_roi.any() else 0.5
    local_scale = np.clip(local_mean / 0.5, 0.4, 1.6)
    ksize       = max(3, int(min(ph_r, pw_r)*0.25) | 1)
    if ksize % 2 == 0: ksize += 1
    soft_mask   = np.clip(cv2.GaussianBlur(em_c.astype(np.float32),(ksize,ksize),0), 0.0, 1.0)

    void_region = (p_c < 0.45) & tissue_roi
    if void_region.any():
        w       = np.clip(eff_alpha*(1.0 - p_c/0.45)*1.9, 0.0, 1.0) * soft_mask
        fv      = blended*(1.0-w) + p_c*0.04*local_scale*w
        blended = np.where(void_region, fv*0.90 + blended*0.10, blended)

    halo_region = (p_c > 0.60) & tissue_roi
    if halo_region.any():
        w       = np.clip(eff_alpha*(p_c-0.60)/0.35, 0.0, 1.0) * soft_mask
        fh      = blended*(1.0-w) + np.clip(p_c*1.5*local_scale,0.0,1.0)*w
        blended = np.where(halo_region, fh*0.88 + blended*0.12, blended)

    blended = np.where(tissue_roi, blended, roi)
    result[iy1:iy2, ix1:ix2] = np.clip(blended, 0.0, 1.0)

    # Compute artifact visibility using the contrast-to-noise ratio (CNR).
    artifact_mask = em_c & tissue_roi
    if not artifact_mask.any():
        artifact_mask = tissue_roi
    artifact_u8   = artifact_mask.astype(np.uint8) * 255
    ring_size     = max(5, int(min(ph_r, pw_r)*0.4))
    dilated       = cv2.dilate(artifact_u8, np.ones((ring_size,ring_size), np.uint8))
    surround_mask = (dilated > 0) & ~artifact_mask & tissue_roi
    after_roi     = result[iy1:iy2, ix1:ix2]

    if artifact_mask.any() and surround_mask.any():
        cnr = abs(float(after_roi[artifact_mask].mean()) -
                  float(roi[surround_mask].mean())) / (float(roi[surround_mask].std()) + 1e-6)
    else:
        cnr = 0.0

    thr = {"brain": 0.8, "heart": 1.5, "knee": 1.5}.get(source_type, 3.0)
    visible = cnr > thr

    return result, visible, round(cnr, 3), em_c


def detect_source_type(filename):
    """
    Infer the MRI modality from the source filename.

    The detected modality determines tissue segmentation parameters,
    pixel spacing values, and visibility thresholds used during
    synthetic artifact generation.
    """
    name = filename.lower()
    if any(k in name for k in ("sagittal", "brain", "mr-art")):  return "brain"
    if any(k in name for k in ("heart", "la_", "sa_")):           return "heart"
    if any(k in name for k in ("healthy", "breast", "sick")):     return "breast"
    if "knee_" in name:                                            return "knee"
    return "default"


def generate_synthetic_mri(mri_image, patch_norm, effect_mask, patch_center,
                            num_robots=3, scales=None, h_values=None,
                            motion_vecs=None, alphas=None, contrasts=None,
                            hthr=9.0, contrast_alpha=1.0, contrast_beta=0,
                            min_dist=40, seed=None, source_type="default"):
    """
    Generate a synthetic MRI image containing one or more magnetic
    microrobot artifacts.

    The function selects valid tissue locations, inserts synthetic
    artifacts, evaluates their visibility using CNR, and generates
    bounding-box annotations for object detection training.
    """
    if seed is not None:
        np.random.seed(seed)
    img = np.clip(contrast_alpha * mri_image.astype(np.float32) + contrast_beta,
                  0, 255).astype(np.uint8)
    tissue_mask       = get_tissue_mask(img, source_type=source_type)
    all_tissue_pixels = np.argwhere(tissue_mask > 0)
    if len(all_tissue_pixels) == 0:
        raise ValueError("Tissue mask is empty!")

    # Convert the desired physical robot diameter (10–20 mm)
    # into image-space dimensions using modality-specific pixel spacing.
    if scales is None:
        pixel_spacing = PIXEL_SPACING_MM.get(source_type, 0.80)
        patch_side    = patch_norm.shape[0]   # patch kare olduğundan tek boyut yeterli
        scales = []
        for _ in range(num_robots):
            robot_mm = np.random.uniform(ROBOT_DIAMETER_MM_MIN, ROBOT_DIAMETER_MM_MAX)
            scales.append(round((robot_mm / pixel_spacing) / patch_side, 3))

    if h_values   is None: h_values   = [round(np.random.uniform(0.0, hthr*0.30), 2) for _ in range(num_robots)]
    if alphas     is None: alphas     = [round(np.random.uniform(0.93, 1.0),  3)     for _ in range(num_robots)]
    if contrasts  is None: contrasts  = [round(np.random.uniform(1.0,  1.2),  3)     for _ in range(num_robots)]

    # Motion-based deformation is disabled to maintain
    # a fixed square artifact geometry.
    motion_vecs = [None] * num_robots

    img_f  = img.astype(np.float32) / 255.0
    labels = []
    placed = []

    for i in range(num_robots):
        found = False
        for _ in range(800):
            idx    = np.random.randint(0, len(all_tissue_pixels))
            py, px = all_tissue_pixels[idx]
            if all(np.sqrt((px-ox)**2 + (py-oy)**2) >= min_dist for ox, oy in placed):
                found = True
                break
        if not found:
            print(f"  [Warning] No valid position for robot {i+1}, skipping.")
            continue

        img_f_test, visible, cnr, em_placed = place_magnet(
            img_f, tissue_mask, patch_norm, effect_mask, patch_center,
            cx=px, cy=py, scale=scales[i], h=h_values[i], hthr=hthr,
            motion_vec=None, alpha=alphas[i], contrast=contrasts[i],
            source_type=source_type)

        if not visible:
            print(f"  Robot {i+1}: pos=({px},{py}) not visible (CNR={cnr:.2f}), skipped.")
            continue

        img_f = img_f_test
        placed.append((px, py))

        # Generate the bounding box from the theoretical physical robot size
        # rather than the visible artifact mask.
        # This ensures:
        #   1) The box center always matches the true robot position.
        #   2) The bounding box remains square (AR = 1.0).
        #   3) Edge effects do not shift the annotation center.
        pixel_spacing    = PIXEL_SPACING_MM.get(source_type, 0.80)
        theoretical_side = int(patch_norm.shape[0] * scales[i])
        theoretical_side = max(8, theoretical_side)

        bbox_cx_px   = float(px)
        bbox_cy_px   = float(py)
        bbox_w_px    = theoretical_side
        bbox_h_px    = theoretical_side   # w == h → AR = 1.0
        robot_mm     = round(theoretical_side * pixel_spacing, 2)
        aspect_ratio = 1.0

        print(f"  Robot {i+1}: pos=({px},{py})  scale={scales[i]:.3f}  "
              f"bbox={bbox_w_px}x{bbox_h_px}px ({robot_mm}mm)  AR={aspect_ratio}  "
              f"h={h_values[i]}mm  CNR={cnr:.2f}")
        # Store robot metadata for CSV export and YOLO annotation generation.
        labels.append({
            "robot_id":     i + 1,
            "x":            int(px),
            "y":            int(py),
            "bbox_cx_px":   bbox_cx_px,
            "bbox_cy_px":   bbox_cy_px,
            "bbox_w_px":    bbox_w_px,
            "bbox_h_px":    bbox_h_px,
            "robot_mm":     robot_mm,
            "aspect_ratio": aspect_ratio,
            "scale":        scales[i],
            "h_mm":         h_values[i],
            "visible":      visible,
            "cnr":          cnr,
            "alpha":        alphas[i],
            "contrast":     contrasts[i],
            "motion":       None,
        })

    synthetic_u8 = (img_f * 255).astype(np.uint8)
    synthetic_u8 = add_rice_noise(synthetic_u8, float(np.random.uniform(2.0, 5.0)))
    return synthetic_u8, tissue_mask, labels


def generate_negative_samples(all_mri_files, n_samples, output_folder,
                               annotated_folder, yolo_folder, rice_sigma=3.0,
                               train_ratio=0.72, val_ratio=0.18):
    """
    Generate negative MRI samples without microrobot artifacts.

    These images are used as background-only examples for object
    detection training and receive empty YOLO annotation files.
    Mild Rician noise is added to preserve MRI realism.
    """
    import random
    chosen    = random.choices(all_mri_files, k=n_samples)
    neg_count = 0
    for src in chosen:
        mri = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        if mri is None: continue
        mf = mri.astype(np.float32)
        N1 = np.random.normal(0, rice_sigma, mf.shape).astype(np.float32)
        N2 = np.random.normal(0, rice_sigma, mf.shape).astype(np.float32)
        noisy = np.clip(np.sqrt((mf+N1)**2 + N2**2), 0, 255).astype(np.uint8)
        neg_count += 1
        fn = f"negative_{neg_count:04d}.png"
        cv2.imwrite(os.path.join(output_folder,    fn), noisy)
        cv2.imwrite(os.path.join(annotated_folder, fn), noisy)

        # Assign each negative sample to the train, validation,
        # or test subset using the predefined split ratios.
        split = "train" if rnd < train_ratio else ("val" if rnd < train_ratio+val_ratio else "test")
        yolo_img = os.path.join(yolo_folder, split, "images", fn)
        yolo_lbl = os.path.join(yolo_folder, split, "labels", fn.replace(".png", ".txt"))
        cv2.imwrite(yolo_img, noisy)
        open(yolo_lbl, "w").close()
    print(f"  Negatif ornek: {neg_count} goruntu eklendi (robot yok, label yok)")
    return neg_count


def visualize(original, tissue_mask, synthetic, labels,
              patch_norm=None, save_path=None, title_prefix=""):
    """
    Visualize the original MRI, tissue mask, synthetic result,
    and generated annotations for qualitative inspection.

    Bounding boxes, robot centers, and physical robot sizes
    are overlaid to simplify validation of generated samples.
    """
    ncols  = 4 if patch_norm is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 6))
    fig.patch.set_facecolor('#1a1a1a')

    imgs   = [original, tissue_mask, synthetic]
    titles = [f"Original MRI{' - '+title_prefix if title_prefix else ''}",
              "Tissue Mask", f"Synthetic ({len(labels)} robots)"]
    if patch_norm is not None:
        pshow = (patch_norm*255).astype(np.uint8) if patch_norm.max()<=1.0 else patch_norm
        imgs.append(pshow)
        titles.append("Magnet Patch (square, AR=1.0)")

    for ax, im, t in zip(axes, imgs, titles):
        ax.imshow(im, cmap='gray', vmin=0, vmax=255)
        ax.set_title(t, color='white', fontsize=12, pad=6)
        ax.axis('off')

    import matplotlib.patches as mpatches
    ax_syn = axes[2]
    for lbl in labels:
        col = 'lime' if lbl['visible'] else 'red'
        bx, by = lbl['bbox_cx_px'], lbl['bbox_cy_px']
        bw, bh = lbl['bbox_w_px'],  lbl['bbox_h_px']

        # Draw the ground-truth robot center.
        ax_syn.plot(bx, by, 'o', color='cyan', markersize=4,
                    markeredgecolor='white', markeredgewidth=0.8, zorder=6)
        # Draw the theoretical square bounding box.
        rect = mpatches.Rectangle(
            (bx - bw/2, by - bh/2), bw, bh,
            fill=False, edgecolor=col, linewidth=1.5)
        ax_syn.add_patch(rect)
        ax_syn.annotate(
            f"R{lbl['robot_id']} {lbl['robot_mm']}mm\nAR={lbl['aspect_ratio']:.2f}",
            (bx, by), textcoords="offset points", xytext=(5, -16),
            color=col, fontsize=7,
            bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.65))

    plt.tight_layout(pad=1.2)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"Saved: {save_path}")
    plt.show()


if __name__ == "__main__":

    MAGNET_IMG       = "magnet_pattern.png"
    MRI_FOLDER       = r"C:\Users\kmfm2\Downloads\all_mri"
    OUTPUT_FOLDER    = "synthetic_dataset_all"
    ANNOTATED_FOLDER = "synthetic_dataset_annotated"
    YOLO_FOLDER      = "synthetic_dataset_yolo"
    NUM_PER_ORGAN    = 500
    NUM_BRAIN        = 5000
    NUM_HEART        = 1500
    NUM_KNEE         = 1500
    HTHR             = 9.0
    DOT_RADIUS       = 2

    TRAIN_RATIO = 0.72
    VAL_RATIO   = 0.18
    TEST_RATIO  = 0.10

    os.makedirs(OUTPUT_FOLDER,    exist_ok=True)
    os.makedirs(ANNOTATED_FOLDER, exist_ok=True)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(YOLO_FOLDER, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(YOLO_FOLDER, split, "labels"), exist_ok=True)
    os.makedirs(os.path.join(YOLO_FOLDER, "images_annotated"), exist_ok=True)

    print("Loading magnet pattern...")
    patch_norm, effect_mask, patch_center = extract_magnet_patch(MAGNET_IMG)
    print(f"  Patch: {patch_norm.shape}  AR={patch_norm.shape[1]/patch_norm.shape[0]:.3f}\n")

    all_files = sorted(glob.glob(os.path.join(MRI_FOLDER,"*.jpg")) +
                       glob.glob(os.path.join(MRI_FOLDER,"*.png")))
    if not all_files:
        raise FileNotFoundError(f"No images found in: {MRI_FOLDER}")

    organ_files = {"brain":[], "heart":[], "breast":[], "knee":[]}
    for f in all_files:
        t = detect_source_type(os.path.basename(f))
        organ_files[t if t in organ_files else "brain"].append(f)
    for org, fl in organ_files.items():
        print(f"  {org}: {len(fl)} images")
    print()

    organ_lists = {}
    for org, fl in organ_files.items():
        if not fl:
            print(f"  [Warning] {org}: no images, skipping.")
            continue
        n = {"brain":NUM_BRAIN,"heart":NUM_HEART,"knee":NUM_KNEE}.get(org, NUM_PER_ORGAN)
        organ_lists[org] = list(zip([np.random.choice(fl) for _ in range(n)], [org]*n))

    combined  = []
    max_len   = max(len(v) for v in organ_lists.values())
    keys      = list(organ_lists.keys())
    for i in range(max_len):
        for k in keys:
            if i < len(organ_lists[k]):
                combined.append(organ_lists[k][i])

    balanced_files, organ_types = zip(*combined) if combined else ([],[])
    total = len(balanced_files)
    print(f"Total {total} images to generate\n")

    # Create a metadata CSV containing robot positions,
    # bounding-box dimensions, visibility status, and CNR values.
    csv_path   = os.path.join(OUTPUT_FOLDER, "labels.csv")
    csv_file   = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    MAX_ROBOTS = 25
    header = ["filename","split","source_mri","organ","img_w","img_h","num_robots"]
    for r in range(1, MAX_ROBOTS+1):
        header += [f"r{r}_true_cx", f"r{r}_true_cy",
                   f"r{r}_bbox_w_px", f"r{r}_bbox_h_px",
                   f"r{r}_robot_mm", f"r{r}_aspect_ratio",
                   f"r{r}_h_mm", f"r{r}_scale",
                   f"r{r}_visible", f"r{r}_cnr"]
    csv_writer.writerow(header)

    preview_examples = {"brain":None,"heart":None,"breast":None,"knee":None}
    organ_labels_tr  = {"brain":"Brain","heart":"Heart","breast":"Breast","knee":"Knee"}
    last_mri = last_tissue = last_synthetic = last_labels = None
    global_idx = 0

    for src_path, organ in zip(balanced_files, organ_types):
        mri = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if mri is None: continue
        src_name = os.path.basename(src_path)
        num_r    = int(np.random.choice(
            [2,3,4,5,6,7,8,10,12,15,20,25],
            p=[0.10,0.12,0.12,0.12,0.10,0.09,0.08,0.07,0.06,0.05,0.05,0.04]))
        MIN_VIS  = max(1, num_r-1) if organ=="heart" else num_r
        success  = False

        for attempt in range(50):
            try:
                synthetic, tissue_mask, labels = generate_synthetic_mri(
                    mri_image=mri, patch_norm=patch_norm, effect_mask=effect_mask,
                    patch_center=patch_center, num_robots=num_r,
                    hthr=HTHR, min_dist=25, seed=None, source_type=organ)
                if sum(1 for l in labels if l["visible"]) >= MIN_VIS:
                    success = True; break
            except Exception as e:
                print(f"  [Warning] attempt {attempt+1}: {e}")
            new_src = np.random.choice(organ_files[organ])
            mri     = cv2.imread(new_src, cv2.IMREAD_GRAYSCALE)
            src_name = os.path.basename(new_src)
            if mri is None: continue
            num_r   = int(np.random.choice(
                [2,3,4,5,6,7,8,10,12,15,20,25],
                p=[0.10,0.12,0.12,0.12,0.10,0.09,0.08,0.07,0.06,0.05,0.05,0.04]))
            MIN_VIS = max(1, num_r-1) if organ=="heart" else num_r

        if not success: continue

        global_idx += 1
        filename     = f"synthetic_{global_idx:04d}.png"
        img_h, img_w = mri.shape

        rnd   = np.random.random()
        split = "train" if rnd < TRAIN_RATIO else ("val" if rnd < TRAIN_RATIO+VAL_RATIO else "test")

        cv2.imwrite(os.path.join(OUTPUT_FOLDER, filename), synthetic)

        # Generate a visualization image containing bounding boxes
        # and robot center markers for manual inspection.
        annotated = cv2.cvtColor(synthetic, cv2.COLOR_GRAY2BGR)
        for lbl in labels:
            bx, by = int(round(lbl["bbox_cx_px"])), int(round(lbl["bbox_cy_px"]))
            bw, bh = lbl["bbox_w_px"], lbl["bbox_h_px"]
            x1, y1 = int(round(bx-bw/2)), int(round(by-bh/2))
            x2, y2 = int(round(bx+bw/2)), int(round(by+bh/2))
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 1)
            cv2.circle(annotated, (bx,by), DOT_RADIUS,   (0,255,255), -1)
            cv2.circle(annotated, (bx,by), DOT_RADIUS+1, (255,255,255), 1)
        cv2.imwrite(os.path.join(ANNOTATED_FOLDER, filename), annotated)

        # Export YOLO-format object detection annotations:
        # class_id x_center y_center width height
        # All coordinates are normalized to image dimensions.
        yolo_img = os.path.join(YOLO_FOLDER, split, "images", filename)
        yolo_lbl = os.path.join(YOLO_FOLDER, split, "labels",
                                filename.replace(".png",".txt"))
        yolo_ann = os.path.join(YOLO_FOLDER, "images_annotated", filename)
        cv2.imwrite(yolo_img, synthetic)
        cv2.imwrite(yolo_ann, annotated)

        lines = []
        for lbl in labels:
            if not lbl["visible"]: continue
            # Preserve the true robot center location regardless
            # of clipping effects near image boundaries.
            xc  = float(np.clip(lbl["bbox_cx_px"] / img_w, 0.001, 0.999))
            yc  = float(np.clip(lbl["bbox_cy_px"] / img_h, 0.001, 0.999))
            bwn = float(np.clip(lbl["bbox_w_px"]  / img_w, 0.001, 0.500))
            bhn = float(np.clip(lbl["bbox_h_px"]  / img_h, 0.001, 0.500))
            lines.append(f"0 {xc:.6f} {yc:.6f} {bwn:.6f} {bhn:.6f}")
        with open(yolo_lbl, "w") as f:
            f.write("\n".join(lines))

        # CSV
        # Store complete robot metadata for later analysis,
        # filtering, and dataset quality assessment.
        row     = [filename, split, src_name, organ, img_w, img_h, len(labels)]
        lbl_map = {l["robot_id"]: l for l in labels}
        for r_id in range(1, MAX_ROBOTS+1):
            if r_id in lbl_map:
                l = lbl_map[r_id]
                row += [round(l["bbox_cx_px"],2), round(l["bbox_cy_px"],2),
                        l["bbox_w_px"], l["bbox_h_px"],
                        l["robot_mm"], l["aspect_ratio"],
                        l["h_mm"], l["scale"],
                        int(l["visible"]), l.get("cnr",-1)]
            else:
                row += [-1]*10
        csv_writer.writerow(row)

        if preview_examples[organ] is None:
            preview_examples[organ] = (mri, tissue_mask, synthetic, labels, src_name)
        last_mri, last_tissue = mri, tissue_mask
        last_synthetic, last_labels = synthetic, labels

        if global_idx % 100 == 0 or global_idx == 1:
            vc = sum(1 for l in labels if l["visible"])
            print(f"  [{global_idx}/{total}] {filename} | {split} | {organ} | "
                  f"robots={vc}/{len(labels)}")

    csv_file.close()

    print("\nGenerating negative samples...")
    all_flat  = [f for fl in organ_files.values() for f in fl]
    n_neg     = 225
    generate_negative_samples(all_flat, n_neg, OUTPUT_FOLDER,
                               ANNOTATED_FOLDER, YOLO_FOLDER)

    print(f"\nDone! {global_idx} positive + {n_neg} negative images")
    print(f"  Split: train~{int(TRAIN_RATIO*100)}% / val~{int(VAL_RATIO*100)}% / test~{int(TEST_RATIO*100)}%")
    print(f"  YOLO  -> {YOLO_FOLDER}/train|val|test/")
    print(f"  CSV   -> {csv_path}")

    for organ, val in preview_examples.items():
        lbl_tr = organ_labels_tr.get(organ, organ)
        if val:
            mri_, tm_, syn_, lbl_, sn_ = val
        elif last_mri is not None:
            mri_, tm_, syn_, lbl_, sn_ = last_mri, last_tissue, last_synthetic, last_labels, "last"
        else:
            continue
        print(f"\nShowing preview: {lbl_tr} ({sn_})")
        visualize(mri_, tm_, syn_, lbl_, patch_norm=patch_norm,
                  save_path=f"synthetic_result_{organ}.png", title_prefix=lbl_tr)