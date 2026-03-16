import pandas as pd
import os
import cv2
import numpy as np

# Shared Helpers for Task C2 & C3

def load_image_with_black_bg(path):
    # Reads image with alpha channel and blends it onto a black background
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        black_bg = np.zeros_like(b, dtype=np.uint8)
        alpha_factor = a.astype(np.float32) / 255.0
        b = (b.astype(np.float32) * alpha_factor + black_bg.astype(np.float32) * (1.0 - alpha_factor)).astype(np.uint8)
        g = (g.astype(np.float32) * alpha_factor + black_bg.astype(np.float32) * (1.0 - alpha_factor)).astype(np.uint8)
        r = (r.astype(np.float32) * alpha_factor + black_bg.astype(np.float32) * (1.0 - alpha_factor)).astype(np.uint8)
        return cv2.merge([b, g, r])
    return img


def rotate_image(image, angle_degrees):
    # Rotates image around center, expanding canvas to prevent clipping corners
    angle_degrees = float(angle_degrees) % 360.0
    if angle_degrees == 0:
        return image
    if angle_degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle_degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle_degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    h, w = image.shape[:2]
    center = (w * 0.5, h * 0.5)
    M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w * 0.5) - center[0]
    M[1, 2] += (new_h * 0.5) - center[1]

    return cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )


def get_visual_crop(img):
    # Isolates the non-zero icon pixels from the large background canvas
    if img is None:
        return None
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray > 10).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, ww, hh = cv2.boundingRect(coords)
        return (x, y, ww, hh)
    return (0, 0, w, h)


def compute_iou(boxA, boxB):
    # Standard Intersection over Union metric for detection overlap assessment
    lA, tA, rA, bA = boxA
    lB, tB, rB, bB = boxB
    inter_l = max(lA, lB)
    inter_t = max(tA, tB)
    inter_r = min(rA, rB)
    inter_b = min(bA, bB)
    inter_area = max(0, inter_r - inter_l) * max(0, inter_b - inter_t)
    areaA = max(0, rA - lA) * max(0, bA - tA)
    areaB = max(0, rB - lB) * max(0, bB - tB)
    union = areaA + areaB - inter_area + 1e-6
    return inter_area / union


def load_ground_truth_C3(csv_path):
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path)
    gts = []
    for _, row in df.iterrows():
        label = str(row["classname"]).split('-', 1)[-1]
        gts.append({
            "label": label,
            "bbox": [int(row["left"]), int(row["top"]), int(row["right"]), int(row["bottom"])],
            "matched": False
        })
    return gts

def load_ground_truth_C2(csv_path):
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path)
    gts = []
    for _, row in df.iterrows():
        label = str(row["classname"]).split('-', 1)[-1]
        gts.append({
            "label": label,
            "bbox": [int(row["top"]), int(row["left"]), int(row["bottom"]), int(row["right"])],
            "matched": False
        })
    return gts


def nms(detections, iou_thresh=0.1):
    # Class-specific Non-Maxima Suppression to pick highest confidence labels
    if not detections:
        return []

    dets = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept = []

    while dets:
        best = dets.pop(0)
        kept.append(best)

        remaining = []
        for d in dets:
            if d.get("label", None) != best.get("label", None):
                remaining.append(d)
                continue

            lA, tA, rA, bA = d["bbox"]
            lB, tB, rB, bB = best["bbox"]

            inter_area = max(0, min(rA, rB) - max(lA, lB)) * max(0, min(bA, bB) - max(tA, tB))
            areaA = max(0, rA - lA) * max(0, bA - tA)

            if areaA > 0 and (inter_area / (areaA + 1e-6)) > 0.7:
                continue  

            if compute_iou(best["bbox"], d["bbox"]) >= iou_thresh:
                continue  

            remaining.append(d)

        dets = remaining

    return kept


def manual_convolve2d(image, kernel):
    # Manual convolution implementation using zero-padding and sliding window
    i_h, i_w = image.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros_like(image)

    for y in range(i_h):
        for x in range(i_w):
            region = padded_img[y:y + k_h, x:x + k_w]
            output[y, x] = np.sum(region * kernel)

    return output


# Task C2 helpers

def mask_white_background_to_zero_bgr(img_bgr, tol=8):
    # Fills corner regions to ensure icon templates have zero backgrounds
    h, w = img_bgr.shape[:2]
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    filled = img_bgr.copy()

    for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        cv2.floodFill(
            filled, ff_mask, seedPoint=seed, newVal=(0, 0, 0),
            loDiff=(tol, tol, tol), upDiff=(tol, tol, tol),
            flags=4 | cv2.FLOODFILL_FIXED_RANGE
        )
    return filled

def crop_nonzero_gray(T, eps=1e-3):
    m = (np.abs(T) > eps).astype(np.uint8)
    ys, xs = np.where(m)
    if ys.size == 0:
        return T
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    return T[y0:y1 + 1, x0:x1 + 1]


def rotate_mask(mask_u8, angle):
    m = rotate_image(mask_u8, angle)
    return (m > 0).astype(np.uint8) * 255


def to_gray_float(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32)


def build_bgr_pyramid_from_rgb_with_level_mask(img_bgr, max_levels, fg_thresh):
    # Generates a Gaussian pyramid where each level preserves icon-only foreground
    def make_mask(bgr):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return ((g > fg_thresh).astype(np.uint8) * 255)

    pyr_bgr = []
    cur_img = img_bgr.copy()
    cur_mask = make_mask(cur_img)

    for _ in range(max_levels + 1):
        cur_bgr = cur_img.astype(np.float32)
        m = (cur_mask.astype(np.float32) / 255.0)[..., None]  
        cur_bgr = cur_bgr * m
        pyr_bgr.append(cur_bgr)

        cur_img = cv2.pyrDown(cur_img)
        h, w = cur_img.shape[:2]
        cur_mask = cv2.resize(cur_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return pyr_bgr


def ncc_score(patch, tmpl_norm):
    p_mean = float(np.mean(patch))
    p_std = float(np.std(patch)) + 1e-6
    p_norm = (patch - p_mean) / p_std
    return float(np.mean(p_norm * tmpl_norm))

def match_template_multiscale_intensity(
    test_pyr,
    tmpl_pyr,
    label,
    score_thresh=0.72,
    max_peaks_per_map=20,
    min_coverage_frac=0.88,
    fg_thresh=10.0,
    std_floor=5.0,
):
    # Multi-channel Normalized Cross-Correlation (NCC) search across scales
    detections = []

    def ensure_bgr_float(img):
        if img is None:
            return None
        if img.ndim == 2:
            return np.repeat(img.astype(np.float32)[..., None], 3, axis=2)
        if img.ndim == 3 and img.shape[2] == 3:
            return img.astype(np.float32)
        return None

    def crop_to_foreground_rgb(T):
        M = (np.max(T, axis=2) > 1e-3).astype(np.float32) 
        ys, xs = np.where(M > 0)
        if ys.size == 0:
            return None, None
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        Tc = T[y0:y1+1, x0:x1+1].astype(np.float32)
        Mc = M[y0:y1+1, x0:x1+1].astype(np.float32)
        return Tc, Mc

    for tl, test_img in enumerate(test_pyr):
        test_img = ensure_bgr_float(test_img)
        if test_img is None:
            continue

        H, W = test_img.shape[:2]
        scale_to_orig = 2 ** tl
        # Calculate foreground mask to ignore black/background pixels during matching
        test_fg = (np.max(test_img, axis=2) > fg_thresh).astype(np.float32)

        for tmpl_level_img in tmpl_pyr:
            tmpl_level_img = ensure_bgr_float(tmpl_level_img)
            if tmpl_level_img is None:
                continue

            T_rot = tmpl_level_img
            T_rot = ensure_bgr_float(T_rot)
            if T_rot is None:
                continue
            # Crop template to tightest bounding box of non-zero pixels
            T, M = crop_to_foreground_rgb(T_rot)
            if T is None:
                continue

            th, tw = T.shape[:2]
            if th < 8 or tw < 8:
                continue
            if th > H or tw > W:
                continue

            tf = float(np.sum(M))
            if tf < 20.0:
                continue

            kcy, kcx = th // 2, tw // 2
            valid_h = H - th + 1
            valid_w = W - tw + 1
            # Use filter2D for sliding-window summation of overlap area
            cov_full = cv2.filter2D(test_fg, ddepth=cv2.CV_32F, kernel=M, borderType=cv2.BORDER_CONSTANT)
            cov = cov_full[kcy:kcy + valid_h, kcx:kcx + valid_w]

            min_cov = float(min_coverage_frac * tf)
            ok_cov = (cov >= min_cov)

            if not np.any(ok_cov):
                continue

            per_ch_ncc = []
            per_ch_ok = []
            # Per-channel NCC calculation: Numerator = (I-meanI)*(T-meanT)
            for c in range(3):
                Tc = T[:, :, c]
                t_mean = float(np.sum(Tc * M) / (tf + 1e-6))
                t0 = (Tc - t_mean) * M
                t_var = float(np.sum(t0 * t0) / (tf + 1e-6))
                t_std = float(np.sqrt(max(t_var, 0.0)) + 1e-6)
                if t_std < 2.0:
                    continue

                Ic = test_img[:, :, c]
                num_full = cv2.filter2D(Ic, ddepth=cv2.CV_32F, kernel=t0, borderType=cv2.BORDER_CONSTANT)
                sumI_full = cv2.filter2D(Ic, ddepth=cv2.CV_32F, kernel=M, borderType=cv2.BORDER_CONSTANT)
                sumI2_full = cv2.filter2D(Ic * Ic, ddepth=cv2.CV_32F, kernel=M, borderType=cv2.BORDER_CONSTANT)
                num = num_full[kcy:kcy + valid_h, kcx:kcx + valid_w]
                sumI = sumI_full[kcy:kcy + valid_h, kcx:kcx + valid_w]
                sumI2 = sumI2_full[kcy:kcy + valid_h, kcx:kcx + valid_w]
                meanI = sumI / (tf + 1e-6)
                varI = (sumI2 / (tf + 1e-6)) - (meanI * meanI)
                varI = np.maximum(varI, 0.0)
                stdI = np.sqrt(varI) + 1e-6

                ok_std = (stdI >= std_floor)
                ok = ok_cov & ok_std

                ncc_c = np.full((valid_h, valid_w), -1.0, dtype=np.float32)
                denom = stdI * t_std * (tf + 1e-6)
                ncc_c[ok] = num[ok] / denom[ok]
                per_ch_ncc.append(ncc_c)
                per_ch_ok.append(ok.astype(np.float32))

            if len(per_ch_ncc) == 0:
                continue

            ncc_stack = np.stack(per_ch_ncc, axis=0)
            ok_stack = np.stack(per_ch_ok, axis=0)
            ok_count = np.sum(ok_stack, axis=0)
            min_valid_channels = max(1, int(np.ceil(0.67 * len(per_ch_ncc))))
            ok_combined = (ok_count >= float(min_valid_channels)) & ok_cov

            if not np.any(ok_combined):
                continue
            # Combine channels and identify local maxima (peaks) in the similarity map
            weighted_sum = np.sum(ncc_stack * ok_stack, axis=0)
            ncc = np.full((valid_h, valid_w), -1.0, dtype=np.float32)
            ncc[ok_combined] = weighted_sum[ok_combined] / (ok_count[ok_combined] + 1e-6)
            dil = cv2.dilate(ncc, np.ones((3, 3), np.uint8))
            peaks = (ncc >= score_thresh) & (ncc >= (dil - 1e-6))
            ys, xs = np.where(peaks)
            if ys.size == 0:
                continue

            if ys.size > max_peaks_per_map:
                scores = ncc[ys, xs]
                idx = np.argpartition(scores, -max_peaks_per_map)[-max_peaks_per_map:]
                ys, xs = ys[idx], xs[idx]

            for y, x in zip(ys.tolist(), xs.tolist()):
                score = float(ncc[y, x])
                l = int(x * scale_to_orig)
                t = int(y * scale_to_orig)
                r = int((x + tw) * scale_to_orig)
                b = int((y + th) * scale_to_orig)

                detections.append({
                    "score": score,
                    "bbox": [l, t, r, b],
                    "label": label
                })

    return detections

def apply_feature_method(img_bgr, method='intensity'):
    # Returns the image unaltered for intensity matching
    if method == 'intensity':
        return img_bgr.copy()
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    if method == 'sobel':
        sobelx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        edges = cv2.magnitude(sobelx, sobely)
    elif method == 'laplacian':
        edges = np.abs(cv2.Laplacian(blurred, cv2.CV_32F, ksize=3))
        kernel = np.ones((3, 3), np.float32)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
    cv2.normalize(edges, edges, 0, 255, cv2.NORM_MINMAX)
    
    # Convert the 1D edge map back into 3 identical channels
    edges_u8 = edges.astype(np.uint8)
    return cv2.merge([edges_u8, edges_u8, edges_u8])

# Task C3 helpers
def nms_class_agnostic(detections, iou_thresh=0.35):
    # Non-Maximum Suppression to remove redundant detections
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept = []
    for d in dets:
        if any(compute_iou(d["bbox"], k["bbox"]) >= iou_thresh for k in kept):
            continue
        kept.append(d)
    return kept

def rootsift(des, eps=1e-12):
    # Hellinger distance mapping for SIFT descriptors to improve matching robustness
    if des is None or len(des) == 0:
        return des
    des = des.astype(np.float32)
    des /= (np.sum(des, axis=1, keepdims=True) + eps) 
    des = np.sqrt(des) 
    return des


def foreground_mask_no_speckles(img_bgr, min_area=120):
    # Noise reduction via connected components to keep only stable icon structures
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > 10).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean[labels == i] = 255
    return clean


def fit_partial_affine_least_squares(src_xy, dst_xy):
    n = src_xy.shape[0]
    if n < 2:
        return None
    A = np.zeros((2 * n, 4), dtype=np.float64)
    A[0::2, 0] = src_xy[:, 0]
    A[0::2, 1] = -src_xy[:, 1]
    A[0::2, 2] = 1.0
    
    A[1::2, 0] = src_xy[:, 1]
    A[1::2, 1] = src_xy[:, 0]
    A[1::2, 3] = 1.0
    
    B = np.zeros((2 * n,), dtype=np.float64)
    B[0::2] = dst_xy[:, 0]
    B[1::2] = dst_xy[:, 1]
    
    try:
        sol, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    except np.linalg.LinAlgError:
        return None
        
    a, b, tx, ty = sol
    return np.array([[a, -b, tx],
                     [b,  a, ty]], dtype=np.float32)

def fit_partial_affine_2pts(src_pts, dst_pts):
    A = np.array([
        [src_pts[0,0], -src_pts[0,1], 1, 0],
        [src_pts[0,1],  src_pts[0,0], 0, 1],
        [src_pts[1,0], -src_pts[1,1], 1, 0],
        [src_pts[1,1],  src_pts[1,0], 0, 1]
    ], dtype=np.float64)
    B = np.array([
        dst_pts[0,0],
        dst_pts[0,1],
        dst_pts[1,0],
        dst_pts[1,1]
    ], dtype=np.float64)
    
    try:
        sol = np.linalg.solve(A, B)
        a, b, tx, ty = sol
        return np.array([[a, -b, tx],
                         [b,  a, ty]], dtype=np.float32)
    except np.linalg.LinAlgError:
        return None


def keep_unique_train(matches):
    matches = sorted(matches, key=lambda m: m.distance)
    used = set()
    out = []
    for m in matches:
        if m.trainIdx in used:
            continue
        used.add(m.trainIdx)
        out.append(m)
    return out

def mutual_ratio_matches(des_t, des_i, ratio=0.72):
    # Manual Euclidean distance calculation and Lowe's Ratio Test filtering
    if des_t is None or des_i is None or len(des_t) == 0 or len(des_i) == 0:
        return []

    # Vectorized computation of the Euclidean distance squared matrix between all descriptors.
    # We use the mathematical expansion: ||a - b||^2 = ||a||^2 + ||b||^2 - 2(a dot b)
    d1_sq = np.sum(des_t ** 2, axis=1, keepdims=True)
    d2_sq = np.sum(des_i ** 2, axis=1, keepdims=True).T
    dist_sq = np.maximum(d1_sq + d2_sq - 2.0 * np.dot(des_t, des_i.T), 0.0)
    if dist_sq.shape[1] < 2 or dist_sq.shape[0] < 2:
        return []

    # --- Forward Matching (Query to Train) ---
    part_t = np.argpartition(dist_sq, 1, axis=1)
    best_idx_t = part_t[:, 0]
    second_idx_t = part_t[:, 1]

    row_idx = np.arange(dist_sq.shape[0])
    best_dist_sq_t = dist_sq[row_idx, best_idx_t]
    second_dist_sq_t = dist_sq[row_idx, second_idx_t]
    
    ratio_sq = ratio ** 2
    valid_t = best_dist_sq_t < ratio_sq * second_dist_sq_t
    
    # --- Backward Matching (Train to Query) for Cross-Checking ---
    part_i = np.argpartition(dist_sq, 1, axis=0)
    best_idx_i = part_i[0, :]
    second_idx_i = part_i[1, :]
    
    col_idx = np.arange(dist_sq.shape[1])
    best_dist_sq_i = dist_sq[best_idx_i, col_idx]
    second_dist_sq_i = dist_sq[second_idx_i, col_idx]
    
    valid_i = best_dist_sq_i < ratio_sq * second_dist_sq_i
    
    # --- Mutual Consistency Check ---
    out = []
    valid_t_idx = np.where(valid_t)[0]
    
    for q in valid_t_idx:
        train_idx = best_idx_t[q]
        
        if valid_i[train_idx] and best_idx_i[train_idx] == q:
            dist = np.sqrt(best_dist_sq_t[q])
            out.append(cv2.DMatch(int(q), int(train_idx), 0, float(dist)))
            
    return keep_unique_train(out)

def ransac_partial_affine_refit(src_pts, dst_pts, iterations=800, threshold=5.0):
    if len(src_pts) < 2:
        return None, np.array([], dtype=bool)
    # Project all source points and calculate Euclidean error against target points
    src = src_pts.reshape(-1, 2).astype(np.float32)
    dst = dst_pts.reshape(-1, 2).astype(np.float32)
    n = src.shape[0]

    best_inliers = np.zeros((n,), dtype=bool)
    best_count = 0
    best_M = None

    ones = np.ones((n, 1), dtype=np.float32)
    src_h = np.hstack([src, ones])

    for _ in range(iterations):
        idx = np.random.choice(n, 2, replace=False)
        M_candidate = fit_partial_affine_2pts(src[idx], dst[idx])
        if M_candidate is None:
            continue

        pred = src_h @ M_candidate.T
        errs = np.linalg.norm(dst - pred, axis=1)

        # Count inliers where projection error is below the pixel threshold
        inliers = errs < threshold
        count = int(np.sum(inliers))

        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_M = M_candidate

    if best_M is None or best_count < 2:
        return None, np.zeros((n,), dtype=bool)

    M_refit = fit_partial_affine_least_squares(src[best_inliers], dst[best_inliers])
    if M_refit is None:
        return best_M, best_inliers
    return M_refit, best_inliers