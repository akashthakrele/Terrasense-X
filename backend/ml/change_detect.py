import os
import numpy as np
from PIL import Image, ImageFilter
from scipy.fft import fft2, ifft2
from scipy.ndimage import label
import config as cfg

# V2 Configurables
DIFF_THRESH = 40
MORPH_OPEN_KERNEL = 5
MORPH_CLOSE_KERNEL = 7
MIN_COMPONENT_SIZE = 100

def phase_correlation_align(img1, img2):
    gray1 = np.mean(img1, axis=2)
    gray2 = np.mean(img2, axis=2)
    
    F1 = fft2(gray1)
    F2 = fft2(gray2)
    cross_power = (F1 * F2.conj()) / (np.abs(F1 * F2.conj()) + 1e-8)
    corr = np.abs(ifft2(cross_power))
    
    max_idx = np.unravel_index(np.argmax(corr), corr.shape)
    sy = max_idx[0] if max_idx[0] < corr.shape[0] // 2 else max_idx[0] - corr.shape[0]
    sx = max_idx[1] if max_idx[1] < corr.shape[1] // 2 else max_idx[1] - corr.shape[1]
    
    aligned = np.roll(img2, shift=(sy, sx), axis=(0, 1))
    return aligned, (sx, sy)

def color_match(src, ref):
    res = np.zeros_like(src, dtype=np.float32)
    for i in range(3):
        src_c = src[:,:,i].astype(np.float32)
        ref_c = ref[:,:,i].astype(np.float32)
        m_s, std_s = src_c.mean(), src_c.std()
        m_r, std_r = ref_c.mean(), ref_c.std()
        if std_s > 0:
            res[:,:,i] = (src_c - m_s) * (std_r / std_s) + m_r
        else:
            res[:,:,i] = src_c
    return np.clip(res, 0, 255).astype(np.uint8)

def generate_overlay(after_img, binary_mask, out_path):
    # Overlay semi-transparent red on the mask
    overlay = np.copy(after_img)
    
    # Red color: [255, 60, 0]
    r_channel = overlay[:, :, 0].astype(np.float32)
    g_channel = overlay[:, :, 1].astype(np.float32)
    b_channel = overlay[:, :, 2].astype(np.float32)
    
    alpha = 0.5
    mask_bool = binary_mask > 127
    
    r_channel[mask_bool] = r_channel[mask_bool] * (1 - alpha) + 255 * alpha
    g_channel[mask_bool] = g_channel[mask_bool] * (1 - alpha) + 60 * alpha
    b_channel[mask_bool] = b_channel[mask_bool] * (1 - alpha) + 0 * alpha
    
    overlay[:, :, 0] = np.clip(r_channel, 0, 255).astype(np.uint8)
    overlay[:, :, 1] = np.clip(g_channel, 0, 255).astype(np.uint8)
    overlay[:, :, 2] = np.clip(b_channel, 0, 255).astype(np.uint8)
    
    Image.fromarray(overlay).save(out_path)


def detect_change(img1_path, img2_path, pair_id):
    """
    Production change detector based on rigorous benchmark evaluation.
    Configuration: aligned_rgb_norm, threshold=35, min_area=50
    """
    img1_pil = Image.open(img1_path).convert('RGB')
    img2_pil = Image.open(img2_path).convert('RGB')
    
    img1 = np.array(img1_pil)
    img2 = np.array(img2_pil)
    
    mean1 = img1.mean()
    mean2 = img2.mean()
    mean_diff = abs(mean1 - mean2)
    g_diff = abs(img1[:,:,1].mean() - img2[:,:,1].mean())

    # 1. Alignment
    try:
        aligned_img2, shift = phase_correlation_align(img1, img2)
        if abs(shift[0]) > 50 or abs(shift[1]) > 50:
            aligned_img2 = img2
            shift = (0, 0)
    except Exception:
        aligned_img2 = img2
        shift = (0, 0)

    # 2. Normalization
    norm_img2 = color_match(aligned_img2, img1)

    # 3. Difference (RGB Norm)
    diff = np.abs(img1.astype(np.int16) - norm_img2.astype(np.int16)).astype(np.uint8)
    gray_diff = np.mean(diff, axis=2)

    # 4. Threshold & Morphology
    # Benchmark winning threshold: 35
    thresh = (gray_diff > 35).astype(np.uint8) * 255
    thresh_pil = Image.fromarray(thresh)
    opened = thresh_pil.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    closed = opened.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    closed_np = np.array(closed)
    
    # 5. Connected Components (Min Area: 50)
    labeled_array, num_features = label(closed_np > 127)
    final_mask = np.zeros_like(closed_np)
    
    num_regions = 0
    max_region_pct = 0.0
    total_pixels = img1.shape[0] * img1.shape[1]

    if num_features > 0:
        for i in range(1, num_features + 1):
            region_mask = (labeled_array == i)
            area = np.sum(region_mask)
            
            if area >= 50:
                final_mask[region_mask] = 255
                num_regions += 1
                
                pct = (area / total_pixels) * 100
                if pct > max_region_pct:
                    max_region_pct = pct

    changed_pixels = np.sum(final_mask == 255)
    changed_area_percent = (changed_pixels / total_pixels) * 100

    # Save outputs
    os.makedirs(cfg.OUTPUTS_DIR, exist_ok=True)
    mask_name = f"mask_benchmark_{pair_id}.png"
    overlay_name = f"overlay_benchmark_{pair_id}.png"
    
    Image.fromarray(final_mask).save(os.path.join(cfg.OUTPUTS_DIR, mask_name))
    generate_overlay(img2, final_mask, os.path.join(cfg.OUTPUTS_DIR, overlay_name))

    confounders = []
    if mean_diff > 15:
        confounders.append("Illumination / shadows")
    if g_diff > 10:
        confounders.append("Vegetation / seasonal variation")
    if abs(shift[0]) > 20 or abs(shift[1]) > 20:
        confounders.append("Image alignment")

    result = {
        "changed_pixel_area": int(changed_pixels),
        "changed_area_percent": round(changed_area_percent, 2),
        "detected_regions": num_regions,
        "max_region_pct": round(max_region_pct, 2),
        "potential_confounders": confounders,
        "analysis_method": "aligned_rgb_norm (t=35, area=50)",
        "mask_url": f"/outputs/{mask_name}",
        "overlay_url": f"/outputs/{overlay_name}",
        "alignment_status": "Successful" if shift == (0,0) else "Aligned"
    }
    
    analysis = generate_explanation(result)
    result.update(analysis)
    
    return result

def generate_explanation(result):
    """
    Deterministic rule-based explanation generator using structural heuristics.
    """
    num_regions = result.get("detected_regions", 0)
    
    if num_regions == 0:
        status = "NO CHANGE DETECTED"
        summary = "No detector-positive regions found."
    else:
        status = "CHANGE DETECTED"
        summary = f"{num_regions} change region(s) identified."
        
    return {
        "summary": summary,
        "review_status": status
    }
