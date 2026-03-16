import argparse
import pandas as pd
import cv2
import numpy as np
import time
import os
from t1_utils import edge_detection, hough_lines, calculate_geometric_angle
from t2_3_utils import *
from optimiseC3 import grid_search_c3

def test_task_c1(folder_name):
    annotation_path = os.path.join(folder_name, "list.txt")
    try:
        task1_data = pd.read_csv(annotation_path)
    except Exception as e:
        print(f"Error reading list.txt: {e}")
        return None

    errors = []

    for index, row in task1_data.iterrows():
        try:
            filename = row.iloc[0] 
            true_angle = float(row.iloc[1])
        except ValueError:
            continue
        
        img_path = os.path.join(folder_name, filename)
        img = cv2.imread(img_path)
        if img is None: continue

        # 1. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Canny Edge Detection with Hysteresis Thresholding
        edges = edge_detection(gray, low_thresh=75, high_thresh=150)
        
        # 3. Hough Transform
        lines = hough_lines(edges, theta_res=1, threshold=40)
        
        # 4. Angle Calculation
        pred_angle = calculate_geometric_angle(lines, edges)
        
        error = abs(pred_angle - true_angle)
        errors.append(error)

        print(f"Img: {filename} | Pred: {pred_angle:.2f} | True: {true_angle:.2f} | Diff: {error:.2f}")

    if not errors: return None
        
    total_squared_error = sum([e**2 for e in errors])
    
    print(f"\n--- Task 1 Results ---")
    print(f"Total Squared Error: {total_squared_error:.4f}")
    return total_squared_error


def test_task_c2(icon_dir, test_dir):
    outputs_folder = "C2_Results"
    os.makedirs(outputs_folder, exist_ok=True)

    img_dir = os.path.join(test_dir, "images")
    ann_dir = os.path.join(test_dir, "annotations")
    EVAL_METHOD = 'intensity'  # Change to 'intensity', 'sobel', or 'laplacian'
    PYRAMID_LEVELS = 3
    pyramid_saved = False 

    templates = {}
    png_subdir = os.path.join(icon_dir, "png")
    if os.path.exists(png_subdir):
        icon_dir = png_subdir

    for fname in sorted(os.listdir(icon_dir)):
        if not fname.lower().endswith((".png", ".jpg")):
            continue
        label = os.path.splitext(fname)[0].split('-', 1)[-1]
        path = os.path.join(icon_dir, fname)

        img = load_image_with_black_bg(path)
        if img is None:
            continue
        
        # Isolate the icon from its original 512x512 canvas to create a tight template
        x, y, w, h = get_visual_crop(img)
        crop = img[y:y + h, x:x + w]
        
        # Zero-out background noise to ensure NCC similarity scores focus only on icon structure
        crop_masked = mask_white_background_to_zero_bgr(crop, tol=2)
        crop_features = apply_feature_method(crop_masked, method=EVAL_METHOD)
        
        # Build pyramid
        templates[label] = build_bgr_pyramid_from_rgb_with_level_mask(crop_features, max_levels=PYRAMID_LEVELS, fg_thresh=10)
        if not pyramid_saved and label == "supermarket":
            for lvl, pyr_img in enumerate(templates[label]):
                cv2.imwrite(f"{outputs_folder}/report_pyramid_{label}_lvl_{lvl}_{EVAL_METHOD}.png", pyr_img)
            pyramid_saved = True
    TP, FP, FN = 0, 0, 0
    matched_ious = []

    start_total = time.time()
    image_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".png")])
    
    # Test Image Processing Loop
    for i, img_name in enumerate(image_files):
        test_bgr = load_image_with_black_bg(os.path.join(img_dir, img_name))
        if test_bgr is None:
            continue
        
        # Mirror pre-processing on test image for consistency with templates
        test_bgr = mask_white_background_to_zero_bgr(test_bgr, tol=2)
        test_features = apply_feature_method(test_bgr, method=EVAL_METHOD)
        test_pyr = build_bgr_pyramid_from_rgb_with_level_mask(test_features, max_levels=PYRAMID_LEVELS, fg_thresh=10)
        
        # Visualization: mask background to highlight detected regions
        vis = test_bgr.copy()
        clean_mask = foreground_mask_no_speckles(test_bgr, min_area=120)
        vis = cv2.bitwise_and(vis, vis, mask=clean_mask)

        all_dets = []
        
        # Multi-scale search across all icon classes
        for label, tmpl_pyr in templates.items():
            all_dets.extend(
                match_template_multiscale_intensity(
                    test_pyr=test_pyr,
                    tmpl_pyr=tmpl_pyr,
                    label=label,
                    #score_thresh=0.4,  # only uncomment for Laplacian algorithm
                    #std_floor=1.0  # only uncomment for Laplacian algorithm
                )
            )
        
        # NMS (IoU=0.2) to suppress sub-matches within large icons
        final_dets = nms(all_dets, iou_thresh=0.2)
        gts = load_ground_truth_C2(os.path.join(ann_dir, os.path.splitext(img_name)[0] + ".csv"))
        img_tp, img_fp = 0, 0
        
        # Evaluation Metrics
        for det in final_dets:
            match_found = False
            best_iou = 0.0
            best_gt = None
            
            for gt in gts:
                if gt["matched"]:
                    continue
                if det["label"] in gt["label"] or gt["label"] in det["label"]:
                    iou = compute_iou(det["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt

            if best_gt is not None and best_iou > 0.85:
                match_found = True
                best_gt["matched"] = True
                img_tp += 1
                matched_ious.append(best_iou)
            else:
                img_fp += 1
           
            # Bounding box visual: Green = Correct, Red = False Positive
            color = (0, 255, 0) if match_found else (0, 0, 255)
            cv2.rectangle(vis, (det["bbox"][0], det["bbox"][1]), (det["bbox"][2], det["bbox"][3]), color, 2)
            cv2.putText(
                vis, det["label"], (det["bbox"][0], max(0, det["bbox"][1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
        # FN = Ground Truths that were never found by a detection
        img_fn = len([g for g in gts if not g["matched"]])

        for gt in gts:
            gt_bbox = gt["bbox"]
            # Draw GT box in Blue (255, 0, 0)
            cv2.rectangle(vis, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (255, 0, 0), 1)
            cv2.putText(vis, f"GT:{gt['label']}", (gt_bbox[0], gt_bbox[3] + 12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        TP += img_tp
        FP += img_fp
        FN += img_fn

        cv2.imwrite(f"{outputs_folder}/c2_result_{i}.png", vis)
    
    # Results
    eps = 1e-6
    acc = TP / (TP + FP + FN + eps)
    tpr = TP / (TP + FN + eps)
    fpr = FP / (TP + FP + eps)
    fnr = FN / (TP + FN + eps)
    avg_time = (time.time() - start_total) / max(1, len(image_files))

    mean_iou = float(np.mean(matched_ious)) if len(matched_ious) else 0.0
    print(
        f"Task C2 Results: ACC: {acc:.4f}, TPR: {tpr:.4f}, FPR: {fpr:.4f}, FNR: {fnr:.4f}, "
        f"Mean IoU(TP): {mean_iou:.4f}, Avg Time: {avg_time:.4f}s"
    )
    return acc, tpr, fpr, fnr

def test_task_c3(icon_folder_name, test_folder_name, 
                 contrast_threshold=0.02,
                 n_octave_layers=10,
                 ransac_threshold=2,
                 ransac_iterations=100,
                 lowe_ratio=0.75,
                 edge_threshold=20,
                 gaussian_blur=1.3,
                 verbose=True,
                 tuning=False):

    if not tuning:
        outputs_folder = "C3_Results"
        os.makedirs(outputs_folder, exist_ok=True)
    
    # Initialize SIFT with parameters tuned for synthetic icon detection: low contrastThreshold to capture subtle features, edgeThreshold to keep keypoints on borders
    sift = cv2.SIFT_create(
        nfeatures=0,
        nOctaveLayers=n_octave_layers,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
        sigma=gaussian_blur
    )

    templates = []

    # Handle both root and 'png' subdirectories in IconDataset
    png_subdir = os.path.join(icon_folder_name, "png")
    if os.path.exists(png_subdir):
        icon_folder_name = png_subdir

    # Template Feature Extraction
    for fname in sorted(os.listdir(icon_folder_name)):
        if not fname.lower().endswith((".png", ".jpg")):
            continue
        label = os.path.splitext(fname)[0].split('-', 1)[-1]
        path = os.path.join(icon_folder_name, fname)

        img = load_image_with_black_bg(path)
        if img is None:
            continue

        # Isolate icon and resize to a consistent target resolution (160px long edge)
        x, y, w, h = get_visual_crop(img)
        crop = img[y:y + h, x:x + w]
        ch, cw = crop.shape[:2]
        target_long = 160
        scale = target_long / max(1, max(ch, cw))
        new_w = max(32, int(cw * scale))
        new_h = max(32, int(ch * scale))
        crop_resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
        mask = foreground_mask_no_speckles(crop_resized)
        kp, des = sift.detectAndCompute(gray, mask)

        if des is not None and len(kp) >= 3:
            templates.append({
                "label": label,
                "kp": kp,
                "des": rootsift(des),
                "size": gray.shape[:2], 
                "img": crop_resized,
                "mask": mask
            })

    img_dir = os.path.join(test_folder_name, "images")
    ann_dir = os.path.join(test_folder_name, "annotations")
    image_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    # Metrics for performance analysis
    TP, FP, FN = 0, 0, 0
    matched_ious = []

    start_total = time.time()

    # Hyperparameters for matching
    demo_written = False

    # Test Image Processing Loop
    for i, img_name in enumerate(image_files):
        test_bgr = load_image_with_black_bg(os.path.join(img_dir, img_name))
        if test_bgr is None:
            continue

        # Standardize test image background for feature stability
        test_bgr = mask_white_background_to_zero_bgr(test_bgr, tol=2)
        base_mask = foreground_mask_no_speckles(test_bgr, min_area=3)
        test_bgr_clean = cv2.bitwise_and(test_bgr, test_bgr, mask=base_mask)

        detections = []

        # SIFT is scale-invariant, so we only need to extract features once at original scale
        test_gray = cv2.cvtColor(test_bgr_clean, cv2.COLOR_BGR2GRAY)
        min_area = 12
        mask = foreground_mask_no_speckles(test_bgr_clean, min_area=min_area)
        
        kp_test, des_test = sift.detectAndCompute(test_gray, mask)
        des_test = rootsift(des_test) if des_test is not None else None

        if des_test is None or len(kp_test) < 3:
            continue

        # Feature Matching & RANSAC
        for tmpl in templates:
            # Get matches using Lowe's Ratio and Mutual Consistency
            good_matches = mutual_ratio_matches(tmpl["des"], des_test, ratio=lowe_ratio)
            min_match = max(3, min(10, int(0.08 * len(tmpl["kp"]))))
            if len(good_matches) < min_match:
                continue

            src_pts = np.float32([tmpl["kp"][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_test[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # RANSAC implementation for Affine Transformation
            M, inliers = ransac_partial_affine_refit(
                src_pts, dst_pts, 
                iterations=ransac_iterations, 
                threshold=ransac_threshold, 
            )

            if M is None or inliers.size == 0:
                continue

            # Geometric Validity
            inlier_count = int(np.sum(inliers))
            if inlier_count < max(4, int(0.7 * min_match)):
                continue
            if inlier_count / max(1, len(good_matches)) < 0.52:
                continue

            src_in = src_pts[inliers].reshape(-1, 2)
            if src_in.shape[0] >= 4:
                tmpl_h, tmpl_w = tmpl["size"]
                sx = (src_in[:, 0].max() - src_in[:, 0].min()) / max(1.0, float(tmpl_w - 1))
                sy = (src_in[:, 1].max() - src_in[:, 1].min()) / max(1.0, float(tmpl_h - 1))
                if sx < 0.18 or sy < 0.18:
                    continue

            # Matrix Determinant and scale validation
            det = float(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])
            scale_x = float(np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2))
            scale_y = float(np.sqrt(M[1, 0] ** 2 + M[1, 1] ** 2))

            if det <= 0.001:
                continue
            if abs(np.log((scale_x + 1e-6) / (scale_y + 1e-6))) > 0.25:
                continue
            if not (0.08 < scale_x < 8.0):
                continue

            # Transform template foreground pixels to test image space to get tight bounding box
            mask_fg = tmpl["mask"]
            ys, xs = np.where(mask_fg > 0)
            if len(ys) == 0:
                continue
                
            pts_fg = np.column_stack((xs, ys)).astype(np.float32)
            ones_fg = np.ones((pts_fg.shape[0], 1), dtype=np.float32)
            pts_fg_h = np.hstack([pts_fg, ones_fg])
            dst_fg = (pts_fg_h @ M.T)
            
            x_min = int(np.min(dst_fg[:, 0]))
            y_min = int(np.min(dst_fg[:, 1]))
            x_max = int(np.max(dst_fg[:, 0]))
            y_max = int(np.max(dst_fg[:, 1]))
            bbox = [x_min, y_min, x_max, y_max]

            # Transform original template corners for polygon checks
            h, w = tmpl["size"]
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            pts2 = pts.reshape(-1, 2)
            ones = np.ones((pts2.shape[0], 1), dtype=np.float32)
            pts_h = np.hstack([pts2, ones])
            dst = (pts_h @ M.T)
            dst = dst.reshape(-1, 1, 2).astype(np.float32)
            H0, W0 = test_bgr.shape[:2]
            l, t, r, b = bbox
            if r <= 0 or b <= 0 or l >= W0 or t >= H0:
                continue

            l2, t2 = max(0, l), max(0, t)
            r2, b2 = min(W0, r), min(H0, b)
            area = (r2 - l2) * (b2 - t2)

            if area < 200 or area > 0.7 * W0 * H0:
                continue

            poly_i32 = np.int32(dst).reshape(-1, 2)
            if len(poly_i32) != 4 or not cv2.isContourConvex(poly_i32):
                continue

            poly_area = abs(cv2.contourArea(poly_i32.astype(np.float32)))
            bbox_area = max(1.0, float((r2 - l2) * (b2 - t2)))
            fill_ratio = poly_area / bbox_area

            if poly_area < 80:
                continue
            if not (0.22 <= fill_ratio <= 1.25):
                continue

            # Scoring by the inlier count and the ratio of good matches
            inlier_ratio = inlier_count / float(max(1, len(good_matches)))
            score = 0.65 * inlier_ratio + 0.35 * min(1.0, inlier_count / 12.0)

            detections.append({
                "label": tmpl["label"],
                "bbox": bbox,
                "poly": dst,
                "score": score,
                "matches": good_matches,
                "inliers": inliers,
                "tmpl": tmpl
            })

        # First NMS within classes, then class-agnostic NMS to pick the best label
        final_dets = nms(detections, iou_thresh=0.2)
        final_dets = nms_class_agnostic(final_dets, iou_thresh=0.25)


        gts = load_ground_truth_C3(os.path.join(ann_dir, os.path.splitext(img_name)[0] + ".csv"))
        img_tp, img_fp = 0, 0

        vis = test_bgr.copy()

        # True Positives using the 0.85 IoU threshold
        IOU_THRESH = 0.85
        for det in final_dets:
            match_found = False
            best_iou = 0.0
            best_gt = None

            for gt in gts:
                if gt["matched"]:
                    continue
                if det["label"] in gt["label"] or gt["label"] in det["label"]:
                    iou = compute_iou(det["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt

                # Draw ground truth boxes in blue
                gt_bbox = gt["bbox"]
                cv2.rectangle(vis, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (255, 0, 0), 1)
                cv2.putText(vis, f"GT:{gt['label']}", (gt_bbox[0], gt_bbox[3] + 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            if best_gt is not None and best_iou > IOU_THRESH:
                match_found = True
                best_gt["matched"] = True
                img_tp += 1
                matched_ious.append(best_iou)
            else:
                img_fp += 1

            # Visualize result with polygons
            color = (0, 255, 0) if match_found else (0, 0, 255)
            l, t, r, b = det["bbox"]
            cv2.rectangle(vis, (l, t), (r, b), color, 2)
            cv2.putText(
                vis, det["label"], (det["bbox"][0], max(0, det["bbox"][1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

        img_fn = len([g for g in gts if not g["matched"]])           

        TP += img_tp
        FP += img_fp
        FN += img_fn

        if not tuning:
            cv2.imwrite(f"{outputs_folder}/c3_result_{i}.png", vis)
        print(f"Image {i+1}/{len(image_files)} processed", end='\r', flush=True)

        if not demo_written and len(final_dets) > 0:
            best = sorted(final_dets, key=lambda d: d["score"], reverse=True)[0]
            tmpl = best["tmpl"]

            pre = cv2.drawMatches(tmpl["img"], tmpl["kp"], test_bgr_clean, kp_test, best["matches"], None, flags=2)
            if not tuning:
                cv2.imwrite(f"{outputs_folder}/c3_match_demo_pre.png", pre)

            inlier_matches = [m for m, is_in in zip(best["matches"], best["inliers"].tolist()) if is_in]
            post = cv2.drawMatches(tmpl["img"], tmpl["kp"], test_bgr_clean, kp_test, inlier_matches, None, flags=2)
            if not tuning:
                cv2.imwrite(f"{outputs_folder}/c3_match_demo_post.png", post)

            demo_written = True
            
    print()
    # Result Aggregation
    eps = 1e-6
    acc = TP / (TP + FP + FN + eps)
    tpr = TP / (TP + FN + eps)
    fpr = FP / (TP + FP + eps)
    fnr = FN / (TP + FN + eps)
    avg_time = (time.time() - start_total) / max(1, len(image_files))

    mean_iou = float(np.mean(matched_ious)) if len(matched_ious) else 0.0
    if verbose:
        print(
            f"Task C3 Results: ACC: {acc:.4f}, TPR: {tpr:.4f}, FPR: {fpr:.4f}, FNR: {fnr:.4f}, "
            f"Mean IoU(TP): {mean_iou:.4f}, Avg Time: {avg_time:.4f}s"
        )
    return acc, tpr, fpr, fnr


if __name__ == "__main__":
    # parsing the command line path to directories and invoking the test scripts for each task
    parser = argparse.ArgumentParser("Data Parser")
    parser.add_argument("--Task1Dataset", help="Provide a folder that contains the Task 1 Dataset.", type=str,
                        required=False)
    parser.add_argument("--IconDataset", help="Provide a folder that contains the Icon Dataset for Task2 and Task3.",
                        type=str, required=False)
    parser.add_argument("--Task2Dataset", help="Provide a folder that contains the Task 2 test Dataset.", type=str,
                        required=False)
    parser.add_argument("--Task3Dataset", help="Provide a folder that contains the Task 3 test Dataset.", type=str,
                        required=False)
    parser.add_argument("--optimise_c3", action="store_true", help="Run C3 hyperparameter optimization grid search.")
    args = parser.parse_args()

    if args.Task1Dataset is not None:
        # This dataset has a list of png files and a txt file that has annotations of filenames and angle
        test_task_c1(args.Task1Dataset)

    if args.IconDataset is not None and args.Task2Dataset is not None:
        # The Icon dataset has a directory that contains the icon image for each file
        # The Task2 dataset directory has two directories, an annotation directory that contains the annotation and a
        # png directory with list of images
        test_task_c2(args.IconDataset, args.Task2Dataset)

    if args.IconDataset is not None and args.Task3Dataset is not None:
        # The Icon dataset directory contains an icon image for each file
        # The Task3 dataset has two directories, an annotation directory that contains the annotation and a png
        # directory with list of images
        if args.optimise_c3:
            grid_search_c3(test_task_c3, args.IconDataset, args.Task3Dataset)
        else:
            test_task_c3(args.IconDataset, args.Task3Dataset)