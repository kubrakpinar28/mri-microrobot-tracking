import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import csv


def extract_magnet_patch(magnet_img_path):
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
        b = max(5, H // 8)
        edges = [crop[:b, :].mean(), crop[-b:, :].mean(),
                 crop[:, :b].mean(), crop[:, -b:].mean()]
        bg = max(float(np.mean(edges)), 20.0)

    else:
        _, ph = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
        ph = cv2.morphologyEx(ph, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))
        contours, _ = cv2.findContours(ph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ph_filled = np.zeros_like(img)
        cv2.drawContours(ph_filled, [max(contours, key=cv2.contourArea)], -1, 255, -1)
        ph_inner  = cv2.erode(ph_filled, np.ones((30, 30), np.uint8))
        masked    = np.where(ph_inner > 0, img.astype(np.int32), 255)
        dark_mask = (masked < 90).astype(np.uint8) * 255
        dark_c, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
        if not dark_c:
            raise ValueError("Magnet dark region not found!")
        centers = []
        for c in dark_c:
            area = cv2.contourArea(c)
            if area > 100:
                M = cv2.moments(c)
                if M['m00'] > 0:
                    centers.append((int(M['m10'] / M['m00']),
                                    int(M['m01'] / M['m00']), area))
        if not centers:
            centers = [(W // 2, H // 2, 1)]
        centers.sort(key=lambda x: -x[2])
        top     = centers[:4]
        total_a = sum(c[2] for c in top)
        mx = int(sum(c[0] * c[2] for c in top) / total_a)
        my = int(sum(c[1] * c[2] for c in top) / total_a)
        half = 110
        y1, y2 = max(0, my - half), min(H, my + half)
        x1, x2 = max(0, mx - half), min(W, mx + half)
        crop    = img[y1:y2, x1:x2].astype(np.float32)
        b       = 20
        edges   = [crop[:b, :].mean(), crop[:, :b].mean(), crop[:, -b:].mean()]
        bg      = max(float(np.mean(edges)), 30.0)

    H2, W2 = crop.shape
    patch_norm = np.clip(crop / (bg * 2.0), 0.0, 1.0).astype(np.float32)

    eff = (np.abs(patch_norm - 0.5) > 0.07).astype(np.uint8) * 255
    k   = np.ones((3, 3), np.uint8)
    effect_mask = cv2.morphologyEx(eff, cv2.MORPH_OPEN,  k)
    effect_mask = cv2.morphologyEx(effect_mask, cv2.MORPH_CLOSE, k * 3)

    center = (W2 // 2, H2 // 2)
    print(f"[Magnet] BG~{bg:.0f} | Patch={patch_norm.shape} | "
          f"Void(<0.4)={(patch_norm < 0.4).sum()} | "
          f"Halo(>0.6)={(patch_norm > 0.6).sum()}")
    return patch_norm, effect_mask, center


def get_tissue_mask(image, source_type="default"):
    """
    source_type:
      "brain"   -> sagittal beyin
      "heart"   -> kalp MRI
      "breast"  -> gogus MRI
      "knee"    -> diz MRI (liver'in yerine eklendi, ayni strateji)
      "default" -> otomatik Otsu
    """
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    H, W = image.shape

    if source_type == "heart":
        otsu_val, _ = cv2.threshold(blurred, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = int(otsu_val * 0.35)
        _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

        k_close = max(20, int(min(H, W) * 0.09))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((k_close, k_close), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        if contours:
            min_area = H * W * 0.005
            big_c = [c for c in contours if cv2.contourArea(c) > min_area]
            if not big_c:
                big_c = [max(contours, key=cv2.contourArea)]
            all_points = np.vstack(big_c)
            hull = cv2.convexHull(all_points)
            cv2.drawContours(filled, [hull], -1, 255, -1)

        erode_k = max(8, int(min(H, W) * 0.04))
        filled = cv2.erode(filled, np.ones((erode_k, erode_k), np.uint8))
        s = max(11, int(min(H, W) * 0.04))
        if s % 2 == 0:
            s += 1
        filled = cv2.GaussianBlur(filled, (s, s), 0)
        _, filled = cv2.threshold(filled, 127, 255, cv2.THRESH_BINARY)
        return filled

    elif source_type == "breast":
        otsu_val, _ = cv2.threshold(blurred, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = int(otsu_val * 0.75)
        _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

        k = max(9, int(min(H, W) * 0.07))
        if k % 2 == 0:
            k += 1
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((7, 7), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            min_area = H * W * 0.03
            big_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            if not big_contours:
                big_contours = [max(contours, key=cv2.contourArea)]
            filled = np.zeros_like(mask)
            cv2.drawContours(filled, big_contours, -1, 255, -1)
            erode_k = max(5, int(min(H, W) * 0.03))
            filled = cv2.erode(filled, np.ones((erode_k, erode_k), np.uint8))
            s = max(11, int(min(H, W) * 0.05))
            if s % 2 == 0:
                s += 1
            filled = cv2.GaussianBlur(filled, (s, s), 0)
            _, filled = cv2.threshold(filled, 127, 255, cv2.THRESH_BINARY)
            mask = filled
        return mask

    elif source_type == "knee":
        # Diz MRI: karacigerin yerine eklendi, benzer strateji
        # Parlak eklem dokusu — yuksek threshold ile sadece diz dokusunu al
        otsu_val, _ = cv2.threshold(blurred, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = int(otsu_val * 0.85)
        _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

        k_close = max(15, int(min(H, W) * 0.06))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((k_close, k_close), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > H * W * 0.05:
                cv2.drawContours(filled, [largest], -1, 255, -1)
            else:
                for c in contours:
                    if cv2.contourArea(c) > H * W * 0.02:
                        cv2.drawContours(filled, [c], -1, 255, -1)

        erode_k = max(8, int(min(H, W) * 0.04))
        filled = cv2.erode(filled, np.ones((erode_k, erode_k), np.uint8))
        s = max(11, int(min(H, W) * 0.04))
        if s % 2 == 0:
            s += 1
        filled = cv2.GaussianBlur(filled, (s, s), 0)
        _, filled = cv2.threshold(filled, 127, 255, cv2.THRESH_BINARY)
        return filled

    else:
        # Brain / default
        otsu_val, _ = cv2.threshold(blurred, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = int(otsu_val * 0.65)
        _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

        k = max(9, int(min(H, W) * 0.08))
        if k % 2 == 0:
            k += 1
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((7, 7), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            filled = np.zeros_like(mask)
            cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, -1)
            erode_k = max(5, int(min(H, W) * 0.03))
            filled = cv2.erode(filled, np.ones((erode_k, erode_k), np.uint8))
            s = max(15, int(min(H, W) * 0.06))
            if s % 2 == 0:
                s += 1
            filled = cv2.GaussianBlur(filled, (s, s), 0)
            _, filled = cv2.threshold(filled, 127, 255, cv2.THRESH_BINARY)
            mask = filled
        return mask


def generate_negative_samples(all_mri_files, n_samples, output_folder,
                               annotated_folder, yolo_folder, rice_sigma=3.0):
    import random
    chosen = random.choices(all_mri_files, k=n_samples)
    neg_count = 0
    for i, src in enumerate(chosen):
        mri = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        if mri is None:
            continue
        mri_f = mri.astype(np.float32)
        N1 = np.random.normal(0, rice_sigma, mri_f.shape).astype(np.float32)
        N2 = np.random.normal(0, rice_sigma, mri_f.shape).astype(np.float32)
        noisy = np.sqrt((mri_f + N1)**2 + N2**2)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        neg_count += 1
        filename = f"negative_{neg_count:04d}.png"

        cv2.imwrite(os.path.join(output_folder, filename), noisy)
        cv2.imwrite(os.path.join(annotated_folder, filename), noisy)

        yolo_img_path = os.path.join(yolo_folder, "images", filename)
        yolo_lbl_path = os.path.join(yolo_folder, "labels", filename.replace(".png", ".txt"))
        cv2.imwrite(yolo_img_path, noisy)
        open(yolo_lbl_path, "w").close()

    print(f"  Negatif ornek: {neg_count} goruntu eklendi (robot yok, label yok)")
    return neg_count


def detect_source_type(filename):
    name = filename.lower()
    if "sagittal" in name or "brain" in name or "mr-art" in name:
        return "brain"
    elif "heart" in name or "la_" in name or "sa_" in name:
        return "heart"
    elif "healthy" in name or "breast" in name or "sick" in name:
        return "breast"
    elif "knee_" in name:
        return "knee"
    else:
        return "default"


def add_rice_noise(image_u8, sigma=12.0):
    img = image_u8.astype(np.float32)
    n1 = np.random.normal(0, sigma, img.shape).astype(np.float32)
    n2 = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = np.sqrt((img + n1) ** 2 + n2 ** 2)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def dirty_image(image_u8, noise_std=8.0, blur_prob=0.0):
    return add_rice_noise(image_u8, sigma=noise_std)


def place_magnet(image_f32, tissue_mask,
                 patch_norm, effect_mask, patch_center,
                 cx, cy,
                 scale=1.0,
                 h=0.0, hthr=9.0,
                 motion_vec=None,
                 alpha=0.92,
                 contrast=1.0,
                 source_type="default"):
    result = image_f32.copy()

    if h >= hthr:
        return result, False, 0.0, None

    visibility = float(np.clip(1.0 - (h / hthr) ** 1.2, 0.0, 1.0))

    pH, pW = patch_norm.shape
    new_h = max(20, int(pH * scale))
    new_w = max(20, int(pW * scale))
    p  = cv2.resize(patch_norm,  (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    em = cv2.resize(effect_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    em = (em > 127).astype(bool)

    if motion_vec is not None:
        dx, dy = motion_vec
        speed  = np.sqrt(dx**2 + dy**2)
        if speed > 0.01:
            sx = 1.0 + abs(dx) * 3.0
            sy = 1.0 + abs(dy) * 3.0
            ww = max(20, int(new_w * sx))
            hh = max(20, int(new_h * sy))
            p  = cv2.resize(p, (ww, hh), interpolation=cv2.INTER_CUBIC)
            em = cv2.resize(em.astype(np.uint8) * 255,
                            (ww, hh), interpolation=cv2.INTER_NEAREST)
            em = (em > 127).astype(bool)
            new_w, new_h = ww, hh

    H, W = result.shape
    x1i  = cx - new_w // 2;  x2i = x1i + new_w
    y1i  = cy - new_h // 2;  y2i = y1i + new_h
    ix1  = max(0, x1i);      ix2 = min(W, x2i)
    iy1  = max(0, y1i);      iy2 = min(H, y2i)
    px1  = ix1 - x1i;        px2 = px1 + (ix2 - ix1)
    py1  = iy1 - y1i;        py2 = py1 + (iy2 - iy1)

    if ix2 <= ix1 or iy2 <= iy1:
        return result, False, 0.0, None

    roi        = result[iy1:iy2, ix1:ix2].copy()
    p_c        = p[py1:py2, px1:px2]
    em_c       = em[py1:py2, px1:px2]
    tissue_roi = (tissue_mask[iy1:iy2, ix1:ix2] > 0)

    eff_alpha = alpha * visibility * contrast * 1.3
    eff_alpha = min(eff_alpha, 1.0)

    blended = roi.copy()

    ph_r, pw_r = p_c.shape
    cy_p, cx_p = ph_r // 2, pw_r // 2
    yy, xx = np.mgrid[0:ph_r, 0:pw_r].astype(np.float32)
    x_rel  = xx - cx_p
    y_rel  = yy - cy_p
    h_px   = h / 0.94 if h > 0 else 0.5
    r_3d   = np.sqrt(x_rel**2 + y_rel**2 + h_px**2) + 1e-6
    cos_th = x_rel / r_3d
    dipole_mag  = np.abs(3 * cos_th**2 - 1) / (r_3d**2 + 1e-6)
    d_max       = np.percentile(dipole_mag, 97) + 1e-9
    dipole_norm = np.clip(dipole_mag / d_max, 0.0, 1.0)
    penumbra_strength = eff_alpha * 0.65 * dipole_norm

    r_falloff = np.exp(-r_3d / (max(ph_r, pw_r) * 0.4))
    penumbra_strength = penumbra_strength * (0.6 + 0.4 * r_falloff)

    pen_result = np.clip(roi * (1.0 - penumbra_strength) - 0.04 * dipole_norm, 0.0, 1.0)
    blended = np.where(tissue_roi, pen_result, roi)

    local_mean = float(roi[tissue_roi].mean()) if tissue_roi.any() else 0.5
    local_scale = np.clip(local_mean / 0.5, 0.4, 1.6)

    ksize = max(3, int(min(ph_r, pw_r) * 0.25) | 1)
    if ksize % 2 == 0:
        ksize += 1
    solid_mask = em_c.astype(np.float32)
    soft_mask = cv2.GaussianBlur(solid_mask, (ksize, ksize), 0)
    soft_mask = np.clip(soft_mask, 0.0, 1.0)

    void_region = (p_c < 0.45) & tissue_roi
    if void_region.any():
        void_target = p_c * 0.04 * local_scale
        w = np.clip(eff_alpha * (1.0 - p_c / 0.45) * 1.9, 0.0, 1.0) * soft_mask
        texture_preserve = 0.10
        final_void = blended * (1.0 - w) + void_target * w
        final_void = final_void * (1.0 - texture_preserve) + blended * texture_preserve
        blended = np.where(void_region, final_void, blended)

    halo_region = (p_c > 0.60) & tissue_roi
    if halo_region.any():
        halo_target = np.clip(p_c * 1.5 * local_scale, 0.0, 1.0)
        w = np.clip(eff_alpha * (p_c - 0.60) / 0.35, 0.0, 1.0) * soft_mask
        texture_preserve = 0.12
        final_halo = blended * (1.0 - w) + halo_target * w
        final_halo = final_halo * (1.0 - texture_preserve) + blended * texture_preserve
        blended = np.where(halo_region, final_halo, blended)

    blended = np.where(tissue_roi, blended, roi)
    result[iy1:iy2, ix1:ix2] = np.clip(blended, 0.0, 1.0).astype(np.float32)

    after_roi = result[iy1:iy2, ix1:ix2]

    artifact_mask = em_c & tissue_roi
    em_placed = em_c
    if not artifact_mask.any():
        artifact_mask = tissue_roi

    artifact_u8   = artifact_mask.astype(np.uint8) * 255
    ring_size     = max(5, int(min(ph_r, pw_r) * 0.4))
    dilated       = cv2.dilate(artifact_u8, np.ones((ring_size, ring_size), np.uint8))
    surround_mask = (dilated > 0) & ~artifact_mask & tissue_roi

    if artifact_mask.any() and surround_mask.any():
        robot_signal   = float(after_roi[artifact_mask].mean())
        surround_mean  = float(roi[surround_mask].mean())
        surround_std   = float(roi[surround_mask].std()) + 1e-6
        cnr = abs(robot_signal - surround_mean) / surround_std
    else:
        cnr = 0.0

    if source_type == "brain":
        cnr_threshold = 0.8
    elif source_type == "heart":
        cnr_threshold = 1.5
    elif source_type == "knee":
        cnr_threshold = 1.5
    else:
        cnr_threshold = 3.0
    visible = cnr > cnr_threshold
    return result, visible, round(cnr, 3), em_placed


def generate_synthetic_mri(
    mri_image,
    patch_norm, effect_mask, patch_center,
    num_robots    = 3,
    scales        = None,
    h_values      = None,
    motion_vecs   = None,
    alphas        = None,
    contrasts     = None,
    hthr          = 9.0,
    contrast_alpha= 1.0,
    contrast_beta = 0,
    min_dist      = 40,
    seed          = None,
    source_type   = "default"
):
    if seed is not None:
        np.random.seed(seed)

    img = np.clip(contrast_alpha * mri_image.astype(np.float32) + contrast_beta,
                  0, 255).astype(np.uint8)

    # 320x320'ye normalize — orantisiz goruntulerde merkezi kare crop
    if img.shape[0] != 320 or img.shape[1] != 320:
        h_orig, w_orig = img.shape
        if max(h_orig, w_orig) / max(min(h_orig, w_orig), 1) > 1.3:
            min_dim = min(h_orig, w_orig)
            y_start = (h_orig - min_dim) // 2
            x_start = (w_orig - min_dim) // 2
            img = img[y_start:y_start+min_dim, x_start:x_start+min_dim]
        img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_LINEAR)

    tissue_mask = get_tissue_mask(img, source_type=source_type)

    H_img, W_img = img.shape
    margin_px = 12

    eroded_mask = cv2.erode(tissue_mask,
                            np.ones((margin_px, margin_px), np.uint8),
                            iterations=1)
    tissue_pixels = np.argwhere(eroded_mask > 0)
    if len(tissue_pixels) == 0:
        tissue_pixels = np.argwhere(tissue_mask > 0)
    if len(tissue_pixels) == 0:
        raise ValueError("Tissue mask is empty!")

    if scales is None:
        scales = [round(np.random.uniform(0.08, 0.13), 3) for _ in range(num_robots)]
    if h_values is None:
        h_values = [round(np.random.uniform(0.0, hthr * 0.30), 2)
                    for _ in range(num_robots)]
    if alphas is None:
        alphas = [round(np.random.uniform(0.93, 1.0), 3) for _ in range(num_robots)]
    if contrasts is None:
        contrasts = [round(np.random.uniform(1.0, 1.2), 3) for _ in range(num_robots)]
    if motion_vecs is None:
        motion_vecs = []
        for _ in range(num_robots):
            if np.random.rand() < 0.45:
                angle = np.random.uniform(0, 2 * np.pi)
                spd   = np.random.uniform(0.05, 0.35)
                motion_vecs.append((np.cos(angle) * spd, np.sin(angle) * spd))
            else:
                motion_vecs.append(None)

    img_f  = img.astype(np.float32) / 255.0
    labels = []
    placed = []

    for i in range(num_robots):
        found = False
        for _ in range(800):
            idx = np.random.randint(0, len(tissue_pixels))
            py, px = tissue_pixels[idx]
            if tissue_mask[py, px] == 0:
                continue
            if all(np.sqrt((px - ox)**2 + (py - oy)**2) >= min_dist
                   for ox, oy in placed):
                found = True
                break
        if not found:
            print(f"  [Warning] No valid position for robot {i + 1}, skipping.")
            continue

        img_f_test, visible, cnr, em_placed = place_magnet(
            img_f, tissue_mask,
            patch_norm, effect_mask, patch_center,
            cx=px, cy=py,
            scale=scales[i],
            h=h_values[i],
            hthr=hthr,
            motion_vec=motion_vecs[i],
            alpha=alphas[i],
            contrast=contrasts[i],
            source_type=source_type
        )

        if not visible:
            print(f"  Robot {i + 1}: pos=({px},{py}) gorunmez (CNR={cnr:.2f}), atlaniyor.")
            continue

        img_f = img_f_test
        placed.append((px, py))

        print(f"  Robot {i + 1}: pos=({px},{py})  scale={scales[i]}  "
              f"h={h_values[i]}mm  CNR={cnr:.2f}  visible=True")

        if em_placed is not None and em_placed.any():
            ys, xs = np.where(em_placed)
            bbox_x1, bbox_x2 = xs.min(), xs.max()
            bbox_y1, bbox_y2 = ys.min(), ys.max()
            bbox_w_px = int(bbox_x2 - bbox_x1 + 1)
            bbox_h_px = int(bbox_y2 - bbox_y1 + 1)
        else:
            bbox_w_px = max(8, int(patch_norm.shape[1] * scales[i]))
            bbox_h_px = max(8, int(patch_norm.shape[0] * scales[i]))

        labels.append({
            "robot_id": i + 1,
            "x":        int(px),
            "y":        int(py),
            "scale":    scales[i],
            "h_mm":     h_values[i],
            "visible":  visible,
            "cnr":      cnr,
            "alpha":    alphas[i],
            "contrast": contrasts[i],
            "motion":   motion_vecs[i],
            "bbox_w_px": bbox_w_px,
            "bbox_h_px": bbox_h_px,
        })

    synthetic_u8 = (img_f * 255).astype(np.uint8)
    sigma = float(np.random.uniform(2.0, 5.0))
    synthetic_u8 = add_rice_noise(synthetic_u8, sigma=sigma)

    return synthetic_u8, tissue_mask, labels


def visualize(original, tissue_mask, synthetic, labels,
              patch_norm=None, save_path=None, title_prefix=""):
    ncols = 4 if patch_norm is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 6))
    fig.patch.set_facecolor('#1a1a1a')

    pshow = None
    if patch_norm is not None:
        pshow = (patch_norm * 255).astype(np.uint8) if patch_norm.max() <= 1.0 \
                else patch_norm

    imgs   = [original, tissue_mask, synthetic]
    titles = [f"Original MRI{' - ' + title_prefix if title_prefix else ''}",
              "Tissue Mask",
              f"Synthetic ({len(labels)} robots)"]
    if pshow is not None:
        imgs.append(pshow)
        titles.append("Magnet Pattern")

    for ax, im, t in zip(axes, imgs, titles):
        ax.imshow(im, cmap='gray', vmin=0, vmax=255)
        ax.set_title(t, color='white', fontsize=12, pad=6)
        ax.axis('off')

    ax_syn = axes[2]
    for lbl in labels:
        col = 'lime' if lbl['visible'] else 'red'
        mrk = 'o'    if lbl['visible'] else 'x'
        ax_syn.plot(lbl['x'], lbl['y'], mrk, color=col,
                    markersize=4, markeredgewidth=1)
        ax_syn.annotate(
            f"R{lbl['robot_id']}\nh={lbl['h_mm']}mm",
            (lbl['x'], lbl['y']),
            textcoords="offset points", xytext=(7, -14),
            color=col, fontsize=8,
            bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.65)
        )

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
    BBOX_SIZE_RATIO  = 0.08

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(ANNOTATED_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(YOLO_FOLDER, "images"), exist_ok=True)
    os.makedirs(os.path.join(YOLO_FOLDER, "labels"), exist_ok=True)
    os.makedirs(os.path.join(YOLO_FOLDER, "images_annotated"), exist_ok=True)

    print("Loading magnet pattern...")
    patch_norm, effect_mask, patch_center = extract_magnet_patch(MAGNET_IMG)

    all_files = sorted(glob.glob(os.path.join(MRI_FOLDER, "*.jpg")) +
                       glob.glob(os.path.join(MRI_FOLDER, "*.png")))
    if not all_files:
        raise FileNotFoundError(f"No images found in: {MRI_FOLDER}")

    organ_files = {"brain": [], "heart": [], "breast": [], "knee": []}
    for f in all_files:
        t = detect_source_type(os.path.basename(f))
        if t in organ_files:
            organ_files[t].append(f)
        else:
            organ_files["brain"].append(f)

    for organ, flist in organ_files.items():
        print(f"  {organ}: {len(flist)} kaynak goruntu")

    toplam_grouped = sum(len(v) for v in organ_files.values())
    print(f"  Toplam taninan: {toplam_grouped} / {len(all_files)}")
    print()
    print("Organ basina uretilecek goruntu sayisi:")
    for organ, flist in organ_files.items():
        n = NUM_BRAIN if organ=="brain" else NUM_HEART if organ=="heart" else NUM_KNEE if organ=="knee" else NUM_PER_ORGAN
        print(f"  {organ}: {n} uretilecek (kaynak: {len(flist)})")
    print()

    organ_lists = {}
    for organ, flist in organ_files.items():
        if not flist:
            print(f"  [Warning] {organ} icin goruntu bulunamadi, atlaniyor.")
            continue
        if organ == "brain":
            n = NUM_BRAIN
        elif organ == "heart":
            n = NUM_HEART
        elif organ == "knee":
            n = NUM_KNEE
        else:
            n = NUM_PER_ORGAN
        chosen = [np.random.choice(flist) for _ in range(n)]
        organ_lists[organ] = list(zip(chosen, [organ] * n))

    combined = []
    organ_keys = list(organ_lists.keys())
    max_len = max(len(v) for v in organ_lists.values())
    for i in range(max_len):
        for key in organ_keys:
            if i < len(organ_lists[key]):
                combined.append(organ_lists[key][i])

    balanced_files, organ_types = zip(*combined) if combined else ([], [])
    total = len(balanced_files)
    print(f"\nToplam {total} goruntu uretilecek "
          f"({NUM_BRAIN} beyin + {NUM_HEART} kalp + {NUM_PER_ORGAN} gogus + {NUM_KNEE} diz)\n")

    csv_path   = os.path.join(OUTPUT_FOLDER, "labels.csv")
    csv_file   = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    MAX_ROBOTS = 25
    header = ["filename", "source_mri", "organ", "img_w", "img_h", "num_robots"]
    for r in range(1, MAX_ROBOTS + 1):
        header += [f"r{r}_x", f"r{r}_y", f"r{r}_h_mm", f"r{r}_scale", f"r{r}_visible", f"r{r}_cnr"]
    csv_writer.writerow(header)

    last_mri = last_tissue = last_synthetic = last_labels = None
    preview_examples = {"brain": None, "heart": None, "breast": None, "knee": None}
    organ_labels_tr  = {"brain": "Beyin", "heart": "Kalp", "breast": "Gogus", "knee": "Diz"}
    global_idx = 0

    for i, (src_path, organ) in enumerate(zip(balanced_files, organ_types)):
        mri = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if mri is None:
            print(f"  [Warning] Could not read: {src_path}")
            continue

        src_name = os.path.basename(src_path)

        num_robots_this = int(np.random.choice(
            [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25],
            p=[0.10, 0.12, 0.12, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04]
        ))

        MIN_VISIBLE = max(1, num_robots_this - 1) if organ == "heart" else num_robots_this
        max_attempts = 50
        success = False
        for attempt in range(max_attempts):
            try:
                synthetic, tissue_mask, labels = generate_synthetic_mri(
                    mri_image    = mri,
                    patch_norm   = patch_norm,
                    effect_mask  = effect_mask,
                    patch_center = patch_center,
                    num_robots   = num_robots_this,
                    hthr         = HTHR,
                    min_dist     = 25,
                    seed         = None,
                    source_type  = organ
                )
                visible_count = sum(1 for lbl in labels if lbl["visible"])
                if visible_count >= MIN_VISIBLE:
                    success = True
                    break
                else:
                    new_src = np.random.choice(organ_files[organ])
                    mri = cv2.imread(new_src, cv2.IMREAD_GRAYSCALE)
                    src_name = os.path.basename(new_src)
                    if mri is None:
                        continue
                    num_robots_this = int(np.random.choice(
                        [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25],
                        p=[0.10, 0.12, 0.12, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04]
                    ))
                    MIN_VISIBLE = max(1, num_robots_this - 1) if organ == "heart" else num_robots_this
            except Exception as e:
                print(f"  [Warning] attempt {attempt+1} failed: {e}")
                continue
        if not success:
            print(f"  [Warning] {MIN_VISIBLE} gorunur robot saglanamadi, atlaniyor.")
            continue

        global_idx += 1
        filename     = f"synthetic_{global_idx:04d}.png"
        img_h, img_w = mri.shape

        cv2.imwrite(os.path.join(OUTPUT_FOLDER, filename), synthetic)

        annotated = cv2.cvtColor(synthetic, cv2.COLOR_GRAY2BGR)
        for lbl in labels:
            color = (0, 255, 0) if lbl["visible"] else (0, 0, 255)
            cv2.circle(annotated, (lbl["x"], lbl["y"]), DOT_RADIUS, color, -1)
            cv2.circle(annotated, (lbl["x"], lbl["y"]), DOT_RADIUS + 1, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(ANNOTATED_FOLDER, filename), annotated)

        yolo_img_path        = os.path.join(YOLO_FOLDER, "images", filename)
        yolo_ann_path        = os.path.join(YOLO_FOLDER, "images_annotated", filename)
        yolo_label_path      = os.path.join(YOLO_FOLDER, "labels",
                                            filename.replace(".png", ".txt"))
        cv2.imwrite(yolo_img_path, synthetic)

        yolo_annotated = cv2.cvtColor(synthetic, cv2.COLOR_GRAY2BGR)
        for lbl in labels:
            if not lbl["visible"]:
                continue
            cv2.circle(yolo_annotated, (lbl["x"], lbl["y"]), DOT_RADIUS, (0, 255, 0), -1)
            cv2.circle(yolo_annotated, (lbl["x"], lbl["y"]), DOT_RADIUS + 1, (255, 255, 255), 1)
        cv2.imwrite(yolo_ann_path, yolo_annotated)

        yolo_lines = []
        for lbl in labels:
            if not lbl["visible"]:
                continue
            x_center = lbl["x"] / img_w
            y_center = lbl["y"] / img_h
            bw = float(np.clip(lbl["bbox_w_px"] / img_w, 0.01, 0.5))
            bh = float(np.clip(lbl["bbox_h_px"] / img_h, 0.01, 0.5))
            x_center = float(np.clip(x_center, bw/2, 1 - bw/2))
            y_center = float(np.clip(y_center, bh/2, 1 - bh/2))
            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

        with open(yolo_label_path, "w") as f:
            f.write("\n".join(yolo_lines))

        row = [filename, src_name, organ, img_w, img_h, len(labels)]
        lbl_map = {lbl["robot_id"]: lbl for lbl in labels}
        for r in range(1, MAX_ROBOTS + 1):
            if r in lbl_map:
                lbl = lbl_map[r]
                row += [lbl["x"], lbl["y"], lbl["h_mm"], lbl["scale"],
                        int(lbl["visible"]), lbl.get("cnr", -1)]
            else:
                row += [-1, -1, -1, -1, -1, -1]
        csv_writer.writerow(row)

        if preview_examples[organ] is None:
            preview_examples[organ] = (mri, tissue_mask, synthetic, labels, src_name)

        last_mri = mri; last_tissue = tissue_mask
        last_synthetic = synthetic; last_labels = labels

        if global_idx % 100 == 0 or global_idx == 1:
            visible_count = sum(1 for lbl in labels if lbl["visible"])
            print(f"  [{global_idx}/{total}] {filename} | organ={organ} | "
                  f"gorunur robot={visible_count}/{len(labels)}")

    csv_file.close()

    print("\nNegatif ornekler uretiliyor...")
    all_files_flat = [f for flist in organ_files.values() for f in flist]
    n_neg = 225
    generate_negative_samples(
        all_mri_files   = all_files_flat,
        n_samples       = n_neg,
        output_folder   = OUTPUT_FOLDER,
        annotated_folder= ANNOTATED_FOLDER,
        yolo_folder     = YOLO_FOLDER,
        rice_sigma      = 3.0
    )

    print(f"\nTamamlandi! {global_idx} pozitif + {n_neg} negatif goruntu kaydedildi.")
    print(f"  Clean PNG   -> {OUTPUT_FOLDER}/")
    print(f"  Annotated   -> {ANNOTATED_FOLDER}/")
    print(f"  YOLO        -> {YOLO_FOLDER}/images/ + /images_annotated/ + /labels/")
    print(f"  CSV         -> {csv_path}")

    for organ, val in preview_examples.items():
        label = organ_labels_tr.get(organ, organ)
        if val is not None:
            mri_, tissue_, syn_, lbl_, sname_ = val
        elif last_mri is not None:
            mri_, tissue_, syn_, lbl_, sname_ = last_mri, last_tissue, last_synthetic, last_labels, "son_goruntu"
            print(f"  [Uyari] {label} icin ornek bulunamadi, son goruntu gosteriliyor.")
        else:
            continue
        print(f"\nGosteriliyor: {label} ({sname_}) -- pencereyi kapatinca sonraki acilir.")
        visualize(mri_, tissue_, syn_, lbl_,
                  patch_norm=patch_norm,
                  save_path=f"synthetic_result_{organ}.png",
                  title_prefix=label)
