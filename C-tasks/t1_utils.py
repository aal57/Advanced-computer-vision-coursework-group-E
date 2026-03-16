import cv2
import numpy as np

def edge_detection(img_gray, low_thresh=75, high_thresh=150):
    blurred_img = cv2.GaussianBlur(img_gray, (5, 5), 0.25)
    
    sobel_x = np.array([[-1, 0, 1], 
                   [-2, 0, 2], 
                   [-1, 0, 1]], dtype=np.float32)
                   
    sobel_y = np.array([[1, 2, 1], 
                   [0, 0, 0], 
                   [-1, -2, -1]], dtype=np.float32)
    
    image_x = cv2.filter2D(blurred_img, cv2.CV_64F, sobel_x)
    image_y = cv2.filter2D(blurred_img, cv2.CV_64F, sobel_y)
    
    magnitude = np.sqrt(image_x**2 + image_y**2)
    
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude = magnitude.astype(np.uint8)
    
    edges = hysteresis_thresholding(magnitude, low_thresh, high_thresh)
    
    return edges

def hysteresis_thresholding(img, low_thresh, high_thresh):
    H, W = img.shape
    output = np.zeros((H, W), dtype=np.uint8)
    
    strong_i, strong_j = np.where(img >= high_thresh)
    
    output[strong_i, strong_j] = 255
    
    # Use a stack to find all weak pixels connected to strong pixels
    stack = list(zip(strong_i, strong_j))
    
    while stack:
        i, j = stack.pop()
        
        # Check 8-connected neighbors
        for diff_i in [-1, 0, 1]:
            for diff_j in [-1, 0, 1]:
                if diff_i == 0 and diff_j == 0: continue
                
                new_i, new_j = i + diff_i, j + diff_j
                
                if 0 <= new_i < H and 0 <= new_j < W:
                    is_weak = img[new_i, new_j] >= low_thresh and img[new_i, new_j] < high_thresh

                    if is_weak and output[new_i, new_j] == 0:
                        output[new_i, new_j] = 255
                        stack.append((new_i, new_j))
                        
    return output

def hough_lines(edges, theta_res=1, rho_res=1, threshold=50):
    H, W = edges.shape
    diag_len = int(np.ceil(np.sqrt(H**2 + W**2)))
    
    thetas = np.deg2rad(np.arange(-90, 90, theta_res))
    rhos = np.linspace(-diag_len, diag_len, int(2 * diag_len / rho_res) + 1)
    
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    
    accumulator = np.zeros((len(rhos), num_thetas), dtype=np.int64)
    
    y_idxs, x_idxs = np.nonzero(edges)
    xs = x_idxs[:, np.newaxis] 
    ys = y_idxs[:, np.newaxis]
    edge_rhos = xs * cos_t + ys * sin_t
    rho_idxs = np.round(edge_rhos + diag_len).astype(np.int64)
    
    for t_idx in range(num_thetas):
        r_indices = rho_idxs[:, t_idx]
        counts = np.bincount(r_indices, minlength=accumulator.shape[0])
        accumulator[:, t_idx] += counts.astype(np.int64)

    detected_lines = []
    acc_copy = accumulator.copy()

    # For suppression window (the area around detected peak to suppress)
    theta_window_deg = 10
    rho_window_px = 20
    theta_range_idxs = int(theta_window_deg / theta_res)
    
    # Find the top two lines
    for _ in range(2): 
        idx = np.argmax(acc_copy)
        max_rho_idx, max_theta_idx = np.unravel_index(idx, acc_copy.shape)
        
        if acc_copy[max_rho_idx, max_theta_idx] < threshold:
            break
            
        rho = rhos[max_rho_idx]
        theta = thetas[max_theta_idx]
        
        detected_lines.append((rho, theta))
        
        # Local suppression to avoid detecting the same line again
        r_min = max(0, max_rho_idx - rho_window_px)
        r_max = min(acc_copy.shape[0], max_rho_idx + rho_window_px)
        t_min = max(0, max_theta_idx - theta_range_idxs)
        t_max = min(acc_copy.shape[1], max_theta_idx + theta_range_idxs)
        acc_copy[r_min:r_max, t_min:t_max] = 0
        
        # If the detected theta is near the edges, suppress the opposite edge
        # to avoid duplicate detections due to angle wrap-around
        if max_theta_idx < theta_range_idxs:
            acc_copy[:, num_thetas - theta_range_idxs:] = 0
            
        elif max_theta_idx > num_thetas - theta_range_idxs:
            acc_copy[:, :theta_range_idxs] = 0
            
    return detected_lines

def get_intersection(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([rho1, rho2])
    
    try:
        x0, y0 = np.linalg.solve(A, b)
        return int(np.round(x0)), int(np.round(y0))
    except np.linalg.LinAlgError:
        return None

def check_ray_for_edge_pixels(img, start_point, angle_rad, check_dist=20):
    H, W = img.shape
    x0, y0 = start_point
    
    # We check at specific points along the ray
    # Check at 50% and 100% of check_dist
    sample_points = [0.5, 1.0] 
    
    hits = 0
    
    for factor in sample_points:
        dist = check_dist * factor
        test_x = int(x0 + dist * np.cos(angle_rad))
        test_y = int(y0 + dist * np.sin(angle_rad))
        
        if 0 <= test_x < W and 0 <= test_y < H:
            # Check 3x3 window around the test point (lines might be slightly off)
            window = img[max(0, test_y-1):min(H, test_y+2), 
                         max(0, test_x-1):min(W, test_x+2)]
            
            if np.sum(window) > 0:
                hits += 1
                
    return hits > 0

def calculate_geometric_angle(lines, edge_img):
    if lines is None or len(lines) < 2:
        return 0.0
        
    # 1. Find the Vertex
    intersection = get_intersection(lines[0], lines[1])
    if intersection is None:
        return 0.0 # Parallel lines
        
    valid_vectors = []
    
    # 2. Check all 4 possible directions
    for rho, theta in lines:
        # Hough theta is the normal vector angle. 
        # The line's actual direction is perpendicular: theta + 90 and theta - 90
        dir_1 = theta + np.pi/2
        dir_2 = theta - np.pi/2
        
        has_pixels_1 = check_ray_for_edge_pixels(edge_img, intersection, dir_1)
        has_pixels_2 = check_ray_for_edge_pixels(edge_img, intersection, dir_2)
        
        if has_pixels_1:
            valid_vectors.append(dir_1)
        elif has_pixels_2:
            valid_vectors.append(dir_2)
        else:
            # Fallback
            valid_vectors.append(dir_1)
            
    if len(valid_vectors) < 2:
        return 0.0
        
    # 3. Calculate angle between the two validated vectors
    v1 = valid_vectors[0]
    v2 = valid_vectors[1]
    
    vec_a = np.array([np.cos(v1), np.sin(v1)])
    vec_b = np.array([np.cos(v2), np.sin(v2)])
    
    dot_product = np.dot(vec_a, vec_b)
    dot_product = max(min(dot_product, 1.0), -1.0)
    
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg