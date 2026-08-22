import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import os
import csv
import glob

# --- Configuration ---
try:
    pytesseract.pytesseract.tesseract_cmd = 'tesseract'
    pytesseract.get_tesseract_version()
except pytesseract.TesseractNotFoundError:
    pytesseract.pytesseract.tesseract_cmd = r'tesseract'

# --- OCR Accuracy Tuning ---
MIN_CONFIDENCE = 30
CROSS_COLOR = (0, 255, 0)  # Green
TEXT_COLOR = (255, 0, 0)   # Blue
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 2
CROSS_SIZE = 8
UPSCALE_FACTOR = 2

def is_numeric_string(s):
    """Checks if the string is a valid number (integers or decimals)."""
    if not s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False

# --- Method 1: Original Adaptive Threshold ---
def preprocess_method_1(gray_image):
    """Adaptive threshold with dilation"""
    upscaled = cv2.resize(gray_image, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(upscaled, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 4)
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    return cv2.bitwise_not(dilated)

# --- Method 2: Otsu's Thresholding ---
def preprocess_method_2(gray_image):
    """Otsu's thresholding - good for bimodal images"""
    upscaled = cv2.resize(gray_image, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(upscaled, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# --- Method 3: Minimal Processing with Denoising ---
def preprocess_method_3(gray_image):
    """Simple upscale + denoise - minimal processing"""
    upscaled = cv2.resize(gray_image, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.fastNlMeansDenoising(upscaled, h=10)
    return denoised

# --- Method 4: Morphological Enhancement ---
def preprocess_method_4(gray_image):
    """Morphological operations to enhance text"""
    upscaled = cv2.resize(gray_image, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(upscaled, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return morph

def run_ocr_with_multiple_psm(preprocessed_image):
    """
    Run OCR with multiple PSM modes and merge results.
    PSM 11 is best for sparse text (scattered numbers).
    """
    psm_modes = [11, 6, 3]  # 11=sparse text, 6=uniform block, 3=auto
    all_detections = []
    
    for psm_mode in psm_modes:
        custom_config = f'--oem 3 --psm {psm_mode} -c tessedit_char_whitelist=0123456789.'
        try:
            ocr_data = pytesseract.image_to_data(preprocessed_image, config=custom_config, output_type=Output.DICT)
            detections = extract_numbers_from_ocr(ocr_data)
            all_detections.extend(detections)
        except Exception as e:
            continue
    
    # Remove duplicates (same position)
    return remove_duplicate_detections(all_detections)

def extract_numbers_from_ocr(ocr_data):
    """Extract valid numbers from OCR data"""
    detected = []
    n_boxes = len(ocr_data['level'])
    
    for i in range(n_boxes):
        text = ocr_data['text'][i].strip()
        
        if not text:
            continue
            
        try:
            conf = int(float(ocr_data['conf'][i]))
        except ValueError:
            conf = -1
        
        if conf > MIN_CONFIDENCE and is_numeric_string(text):
            x_up = ocr_data['left'][i]
            y_up = ocr_data['top'][i]
            w_up = ocr_data['width'][i]
            h_up = ocr_data['height'][i]
            
            # Scale back to original size
            x = x_up // UPSCALE_FACTOR
            y = y_up // UPSCALE_FACTOR
            w = max(1, w_up // UPSCALE_FACTOR)
            h = max(1, h_up // UPSCALE_FACTOR)
            
            mid_x = x + w // 2
            mid_y = y + h // 2
            
            detected.append({
                "text": text,
                "mid_x": mid_x,
                "mid_y": mid_y,
                "confidence": conf
            })
    
    return detected

def remove_duplicate_detections(detections, proximity_threshold=15):
    """
    Remove duplicate detections that are very close together.
    Keep the one with highest confidence.
    """
    if not detections:
        return []
    
    merged = []
    used = set()
    
    for i, det1 in enumerate(detections):
        if i in used:
            continue
        
        cluster = [det1]
        for j, det2 in enumerate(detections):
            if j <= i or j in used:
                continue
            
            dx = det1['mid_x'] - det2['mid_x']
            dy = det1['mid_y'] - det2['mid_y']
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance < proximity_threshold:
                cluster.append(det2)
                used.add(j)
        
        best = max(cluster, key=lambda x: x['confidence'])
        merged.append(best)
    
    return merged

def process_image_with_method(gray_image, method_func, method_name):
    """Process image with a specific preprocessing method"""
    try:
        preprocessed = method_func(gray_image)
        detections = run_ocr_with_multiple_psm(preprocessed)
        return detections, method_name
    except Exception as e:
        print(f"    ✗ {method_name} failed: {e}")
        return [], method_name

def process_image(input_path, output_path, csv_path):
    """
    Process image with all 4 methods, compare results,
    and save the best method's results to CSV.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input image '{input_path}' not found.")
        return

    try:
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error: OpenCV could not load image at '{input_path}'.")
            return

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        print(f"\nProcessing: {os.path.basename(input_path)}")
        print("-" * 70)
        
        # Define all 4 methods
        methods = [
            (preprocess_method_1, "Method 1: Adaptive Threshold + Dilation"),
            (preprocess_method_2, "Method 2: Otsu's Thresholding"),
            (preprocess_method_3, "Method 3: Denoising Only"),
            (preprocess_method_4, "Method 4: Morphological Enhancement")
        ]
        
        # Process with each method and collect results
        results = []
        for method_func, method_name in methods:
            print(f"  Testing {method_name}...", end=" ")
            detections, name = process_image_with_method(gray_image, method_func, method_name)
            count = len(detections)
            print(f"Found {count} number(s)")
            results.append({
                'method_name': name,
                'detections': detections,
                'count': count
            })
        
        # Find the method with MOST detections
        best_result = max(results, key=lambda x: x['count'])
        best_method = best_result['method_name']
        best_detections = best_result['detections']
        best_count = best_result['count']
        
        print("-" * 70)
        print(f"  ★ WINNER: {best_method}")
        print(f"  ★ Total numbers detected: {best_count}")
        print("-" * 70)
        
        # Sort detections by position (top to bottom, left to right)
        if best_detections:
            best_detections.sort(key=lambda x: (x['mid_y'], x['mid_x']))
        
        # Save ONLY the best method's results to CSV
        with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(["Identified Number", "Pixel X", "Pixel Y"])
            
            for det in best_detections:
                csv_writer.writerow([det['text'], det['mid_x'], det['mid_y']])
        
        # Draw markings on output image
        output_image = image.copy()
        for info in best_detections:
            mid_x, mid_y = info['mid_x'], info['mid_y']
            coords_text = f"({mid_x},{mid_y})"
            
            # Draw green cross
            cv2.line(output_image, (mid_x - CROSS_SIZE, mid_y), (mid_x + CROSS_SIZE, mid_y), 
                     CROSS_COLOR, FONT_THICKNESS)
            cv2.line(output_image, (mid_x, mid_y - CROSS_SIZE), (mid_x, mid_y + CROSS_SIZE), 
                     CROSS_COLOR, FONT_THICKNESS)
            
            # Draw coordinate text with boundary checking
            (text_w, text_h), _ = cv2.getTextSize(coords_text, FONT, FONT_SCALE, FONT_THICKNESS)
            text_x = mid_x - text_w // 2
            text_y = mid_y + CROSS_SIZE + text_h + 4
            
            img_h, img_w = output_image.shape[:2]
            text_x = max(0, min(text_x, img_w - text_w))
            if text_y > img_h - 5:
                text_y = mid_y - CROSS_SIZE - 4
            
            cv2.putText(output_image, coords_text, (text_x, text_y), 
                       FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
        
        cv2.imwrite(output_path, output_image)
        
        if best_count > 0:
            print(f"  ✓ Saved {best_count} coordinate(s) to: {os.path.basename(csv_path)}")
            print(f"  ✓ Saved marked image to: {os.path.basename(output_path)}")
        else:
            print(f"  ⚠ No numbers detected - empty files created")

    except pytesseract.TesseractNotFoundError:
        print("\n--- TESSERACT ERROR ---")
        print("Error: Tesseract is not installed or not found in your PATH.")
        print(f"Configured path: '{pytesseract.pytesseract.tesseract_cmd}'")
    except Exception as e:
        print(f"Unexpected error processing {input_path}: {e}")
        import traceback
        traceback.print_exc()

def process_folders(base_path):
    """
    Iterates through folders L0001-L2594, then YS subfolders,
    processes in.png if split_check.csv exists.
    """
    print("=" * 70)
    print(f"Starting OCR Processing")
    print(f"Base directory: {base_path}")
    print("=" * 70)
    
    processed_count = 0
    total_numbers_found = 0
    
    for i in range(1, 3356):
        folder_name = f"L{i:04d}"
        l_folder_path = os.path.join(base_path, folder_name)

        if not os.path.isdir(l_folder_path):
            continue

        ys_folders = glob.glob(os.path.join(l_folder_path, "YS*"))
        if not ys_folders:
            continue

        for ys_folder in ys_folders:
            if not os.path.isdir(ys_folder):
                continue

            split_folders = glob.glob(os.path.join(ys_folder, "split*"))
            
            if split_folders:
                for split_folder in split_folders:
                    if not os.path.isdir(split_folder):
                        continue

                    input_file = os.path.join(split_folder, "in.png")
                    check_file = os.path.join(split_folder, "split_check.csv")

                    if os.path.isfile(input_file) and os.path.isfile(check_file):
                        output_image = os.path.join(split_folder, "in_axis.png")
                        output_csv = os.path.join(split_folder, "axis.csv")
                        process_image(input_file, output_image, output_csv)
                        processed_count += 1
                        
                        # Count numbers found
                        if os.path.exists(output_csv):
                            with open(output_csv, 'r') as f:
                                total_numbers_found += sum(1 for line in f) - 1  # Subtract header
            else:
                input_file = os.path.join(ys_folder, "in.png")
                check_file = os.path.join(ys_folder, "split_check.csv")
                
                if os.path.isfile(input_file) and os.path.isfile(check_file):
                    output_image = os.path.join(ys_folder, "in_axis.png")
                    output_csv = os.path.join(ys_folder, "axis.csv")
                    process_image(input_file, output_image, output_csv)
                    processed_count += 1
                    
                    # Count numbers found
                    if os.path.exists(output_csv):
                        with open(output_csv, 'r') as f:
                            total_numbers_found += sum(1 for line in f) - 1  # Subtract header
    
    print("\n" + "=" * 70)
    print(f"PROCESSING COMPLETE")
    print(f"Total images processed: {processed_count}")
    print(f"Total numbers detected: {total_numbers_found}")
    print("=" * 70)

if __name__ == "__main__":
    base_directory = "."
    process_folders(base_directory)