import cv2
import numpy as np
import os
import glob
import json
import math  # For math.hypot
import csv   # For CSV output

# --- Configuration ---
# Preprocessing parameters
GAUSSIAN_KERNEL_SIZE = (15, 15)
THRESHOLD_TYPE = cv2.THRESH_BINARY_INV
USE_OTSU_THRESHOLDING = True
THRESHOLD_VALUE = 127

# Contour filtering parameters
MIN_CONTOUR_AREA = 15
MAX_CONTOUR_AREA = 2000
# User is testing with 180, and CSV will help debug.
BLACK_THRESHOLD = 180

# Parameters for differentiating shapes from lines
MIN_SOLIDITY_FOR_SHAPE = 0.15
MAX_ELONGATION_RATIO = 10
MIN_ROTATED_BBOX_AREA_FACTOR = 0.1

# Deduplication parameter
DEDUPLICATION_DISTANCE_THRESHOLD = 15

# Marking parameters
CROSS_COLOR = (0, 255, 0)  # Green
TEXT_COLOR = (255, 0, 0)   # Blue
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
CROSS_SIZE = 4

def calculate_solidity(contour):
    """Calculates the solidity of a contour."""
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    return float(area) / hull_area if hull_area else 0

def get_region_average_color(image, contour, thickness=-1):
    """
    Calculates the average BGR color of a contour region in the original image.
    """
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, thickness)
    if cv2.countNonZero(mask) == 0:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            if 0 <= cy < image.shape[0] and 0 <= cx < image.shape[1]:
                return image[cy, cx]
        return (128, 128, 128) # Fallback color
    mean_val = cv2.mean(image, mask=mask)
    return mean_val[:3]

def process_image_for_patterns(input_path, output_path):
    """
    Loads an image, detects patterns (excluding black/gray fills and lines),
    marks midpoints, saves output image, and saves pattern data to data.csv.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input image '{input_path}' not found.")
        # Sample image generation remains for convenience
        sample_img = np.full((500, 700, 3), 255, dtype=np.uint8)
        cv2.circle(sample_img, (50, 50), 15, (0,0,0), -1)
        cv2.putText(sample_img, "IGNORE", (30, 80), FONT, 0.7, (0,0,0), 2)
        cv2.rectangle(sample_img, (150, 50), (180, 70), (70,70,70), -1)
        cv2.line(sample_img, (20, 120), (180, 130), (0,0,0), 5)
        cv2.circle(sample_img, (100, 180), 30, (0,0,0), 2)
        cv2.circle(sample_img, (100, 180), 15, (255,255,255), -1)
        cv2.circle(sample_img, (250, 60), 20, (200,0,0), -1)
        cv2.rectangle(sample_img, (350, 50), (400, 90), (0,200,0), -1)
        pts_triangle_red = np.array([[450, 90], [500, 90], [475, 50]], np.int32)
        cv2.drawContours(sample_img, [pts_triangle_red], 0, (0,0,200), -1)
        # ... (other sample drawing commands can be added if needed)
        cv2.imwrite(input_path, sample_img)
        print(f"Sample '{input_path}' created. Run again with this sample or your image.")
        return

    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: OpenCV could not load image at '{input_path}'.")
        return

    output_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL_SIZE, 0)

    if USE_OTSU_THRESHOLDING:
        _, thresh = cv2.threshold(blurred, 0, 255, THRESHOLD_TYPE + cv2.THRESH_OTSU)
    else:
        _, thresh = cv2.threshold(blurred, THRESHOLD_VALUE, 255, THRESHOLD_TYPE)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    initial_detected_patterns = []

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if not (MIN_CONTOUR_AREA < area < MAX_CONTOUR_AREA):
            continue

        avg_b_fill, avg_g_fill, avg_r_fill = get_region_average_color(image, contour, thickness=-1)
        is_fill_black_gray = (avg_b_fill < BLACK_THRESHOLD and \
                              avg_g_fill < BLACK_THRESHOLD and \
                              avg_r_fill < BLACK_THRESHOLD)
        if is_fill_black_gray:
            continue
        
        solidity = calculate_solidity(contour)
        if len(contour) < 5: continue
        rect = cv2.minAreaRect(contour)
        (w_r, h_r) = rect[1]
        if w_r == 0 or h_r == 0: continue
        if w_r > h_r: w_r, h_r = h_r, w_r 
        elongation_ratio = (h_r / w_r) if w_r > 0.01 else float('inf') 
        rotated_bbox_area = w_r * h_r
        contour_to_rotated_bbox_area_ratio = area / rotated_bbox_area if rotated_bbox_area > 0 else 0

        is_line = False
        if elongation_ratio > MAX_ELONGATION_RATIO: is_line = True
        if not is_line and solidity < MIN_SOLIDITY_FOR_SHAPE and elongation_ratio > (MAX_ELONGATION_RATIO / 2.5): is_line = True

        is_desired_pattern = False
        if not is_line and solidity >= MIN_SOLIDITY_FOR_SHAPE and contour_to_rotated_bbox_area_ratio >= MIN_ROTATED_BBOX_AREA_FACTOR:
            is_desired_pattern = True

        if is_desired_pattern:
            M = cv2.moments(contour)
            if M["m00"] == 0: continue 
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            initial_detected_patterns.append({
                "cx": cx, "cy": cy, "area": area,
                "solidity": solidity, "elongation": elongation_ratio,
                "avg_fill_r": avg_r_fill, "avg_fill_g": avg_g_fill, "avg_fill_b": avg_b_fill
            })

    initial_detected_patterns.sort(key=lambda p: p['area'], reverse=True) 
    final_detected_patterns = []
    if initial_detected_patterns:
        for current_pattern in initial_detected_patterns:
            is_duplicate = False
            for existing_pattern in final_detected_patterns:
                dist = math.hypot(current_pattern['cx'] - existing_pattern['cx'], current_pattern['cy'] - existing_pattern['cy'])
                if dist < DEDUPLICATION_DISTANCE_THRESHOLD:
                    is_duplicate = True; break
            if not is_duplicate: final_detected_patterns.append(current_pattern)
    
    if not final_detected_patterns:
        print(f"No patterns matching criteria were detected in '{input_path}' (after deduplication).")
    else:
        print(f"Detected {len(final_detected_patterns)} unique pattern(s) in '{input_path}'. Drawing markings...")
        for info in final_detected_patterns:
            cx, cy = info['cx'], info['cy']
            coords_text = f"({cx},{cy})"
            cv2.line(output_image, (cx - CROSS_SIZE, cy), (cx + CROSS_SIZE, cy), CROSS_COLOR, FONT_THICKNESS)
            cv2.line(output_image, (cx, cy - CROSS_SIZE), (cx, cy + CROSS_SIZE), CROSS_COLOR, FONT_THICKNESS)
            (text_w, text_h), _ = cv2.getTextSize(coords_text, FONT, FONT_SCALE, FONT_THICKNESS)
            text_x, text_y = cx + CROSS_SIZE + 2, cy 
            img_h, img_w = output_image.shape[:2]
            if text_x + text_w > img_w: text_x = cx - CROSS_SIZE - 2 - text_w
            if text_y - text_h < 0: text_y = cy + text_h + CROSS_SIZE + 2
            elif text_y + (text_h//2) > img_h: text_y = cy - (text_h//2) - CROSS_SIZE - 2
            cv2.putText(output_image, coords_text, (text_x, text_y), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    cv2.imwrite(output_path, output_image)
    print(f"Output image with pattern markings saved to '{output_path}'")

    # --- Write data.csv for the current image ---
    csv_output_dir = os.path.dirname(input_path)
    csv_output_path = os.path.join(csv_output_dir, "data.csv")
    
    print(f"Writing pattern data for '{input_path}' to '{csv_output_path}'...")
    fieldnames = ['color', 'x', 'y'] # Columns as per user request
    
    with open(csv_output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        if final_detected_patterns:
            for p_info in final_detected_patterns:
                csv_row = {
                    'color': f"{int(p_info['avg_fill_r'])},{int(p_info['avg_fill_g'])},{int(p_info['avg_fill_b'])}",
                    'x': p_info['cx'],
                    'y': p_info['cy']
                }
                writer.writerow(csv_row)
            print(f"Data for {len(final_detected_patterns)} patterns written to {csv_output_path}")
        else:
            # If no patterns, an empty CSV (with headers only) is created.
            print(f"No patterns detected in '{input_path}' to write to data.csv. Empty CSV (with headers) created.")
    # No explicit return value needed for CSV data handling by caller

def process_folders(base_path):
    """
    Iterates through folders and processes in.png files.
    CSV writing is handled by process_image_for_patterns for each image.
    """
    for i in range(1, 3317): 
        folder_name = f"L{i:04d}"
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.isdir(folder_path): continue
        ys_folders = glob.glob(os.path.join(folder_path, "YS*"))
        if not ys_folders: continue
        for ys_folder in ys_folders:
            if not os.path.isdir(ys_folder): continue
            data_folders = glob.glob(os.path.join(ys_folder, "data*"))
            if not data_folders: continue
            for data_folder in data_folders:
                if not os.path.isdir(data_folder): continue
                txt_file = os.path.join(data_folder, "check_data.txt")
                if not os.path.isfile(txt_file): continue
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f: content = f.read().strip()
                    if content != "No Numerical annotation": continue
                except Exception as e:
                    print(f"Error reading {txt_file}: {e}, skipping")
                    continue
                json_file = os.path.join(data_folder, "graph_out_fixed.json")
                if not os.path.isfile(json_file): continue
                try:
                    with open(json_file, 'r',  encoding='utf-8') as f: json_data = json.load(f)
                    image_type = json_data.get("image_type", "")
                    if image_type not in ["dot-line plot", "scatter plot"]: continue
                except Exception as e: 
                    print(f"Error processing {json_file}: {e}, skipping")
                    continue
                input_file = os.path.join(data_folder, "in.png")
                output_file = os.path.join(data_folder, "out.png") 
                if not os.path.isfile(input_file): continue
                
                print(f"Processing {input_file}...")
                process_image_for_patterns(input_file, output_file) # This now handles its own CSV

# Usage
if __name__ == "__main__":
    base_path = r".\HEA_finetune\Yield_Strength"
    
    # process_folders no longer returns data for a master CSV
    process_folders(base_path)
        
    print("Batch processing finished. Individual data.csv files created per processed image.")