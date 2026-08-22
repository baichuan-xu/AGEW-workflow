import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import os
import csv
import json
import glob
import traceback

# --- Configuration ---
# IMPORTANT: Set this to your Tesseract installation path if it's not in your system's PATH
pytesseract.pytesseract.tesseract_cmd = r'tesseract' # Make sure this path is correct for your system

MIN_CONFIDENCE = 10
CROSS_COLOR = (0, 255, 0)  # Green
TEXT_COLOR = (255, 0, 0)   # Blue
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
CROSS_SIZE = 8
UPSCALE_FACTOR = 2

# --- HSV Color Ranges ---
# These ranges might need tuning based on the specific colors in your images.
# Format: "color_name": [(lower_hsv_bound, upper_hsv_bound), ...]
# OpenCV HSV ranges: H: 0-179, S: 0-255, V: 0-255
COLOR_RANGES_HSV = {
    "red": [
        (np.array([0, 100, 100]), np.array([10, 255, 255])),  # Lower hue red
        (np.array([160, 100, 100]), np.array([179, 255, 255])) # Upper hue red
    ],
    "green": [(np.array([35, 50, 50]), np.array([85, 255, 255]))],
    "blue": [(np.array([90, 50, 50]), np.array([130, 255, 255]))],
    "yellow": [(np.array([20, 100, 100]), np.array([35, 255, 255]))], # Adjusted from H=25-35
    "orange": [(np.array([10, 100, 100]), np.array([24, 255, 255]))], # H=11-24
    "pink": [ # Hues around magenta/red
        (np.array([140, 50, 70]), np.array([165, 255, 255])) # Broader range for various pinks near magenta
    ],
    "brown": [(np.array([5, 60, 20]), np.array([30, 255, 180]))], # Covers reddish to yellowish browns
    "gray": [(np.array([0, 0, 40]), np.array([179, 70, 220]))], # Low saturation, wide hue and value
    "teal": [(np.array([80, 60, 60]), np.array([105, 255, 255]))], # Cyan/Green-Blue region
    # You had "teal" and "cyan" as potentially similar. Teal is often blue-green.
    # "purple": [(np.array([125, 50, 50]), np.array([155, 255, 255]))], # Example for purple
}

# --- Helper function to check if a string is purely numeric ---
def is_numeric_string(s):
    """Checks if the string contains only digits."""
    return bool(s) and s.isdigit()

# --- Main processing function ---
def process_image(input_path, output_path, csv_path, target_color_name):
    """
    Loads an image, preprocesses it focusing on a target color, detects numbers
    using OCR, marks them, saves the result image, and writes detected numbers
    and coordinates to a CSV file.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input image '{input_path}' not found.")
        return

    try:
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error: OpenCV could not load image at '{input_path}'. Check file integrity and path.")
            return

        output_image = image.copy()

        # --- Color Segmentation ---
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_ranges = COLOR_RANGES_HSV.get(target_color_name.lower())

        if not color_ranges:
            print(f"Warning: Color '{target_color_name}' is not defined in COLOR_RANGES_HSV. OCR will process the image in grayscale.")
            gray_image_for_ocr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            combined_mask = None
            for hsv_lower, hsv_upper in color_ranges:
                mask = cv2.inRange(hsv_image, hsv_lower, hsv_upper)
                if combined_mask is None:
                    combined_mask = mask
                else:
                    combined_mask = cv2.bitwise_or(combined_mask, mask)

            if combined_mask is None or cv2.countNonZero(combined_mask) == 0:
                print(f"No pixels found for color '{target_color_name}' in '{input_path}'. Skipping OCR for this image.")
                with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
                    csv_writer = csv.writer(file)
                    csv_writer.writerow(["Identified Number", "Pixel X", "Pixel Y"])
                cv2.imwrite(output_path, output_image)
                print(f"Empty CSV '{csv_path}' and original image saved to '{output_path}' as no target color found.")
                return

            isolated_color_image = np.full_like(image, (255, 255, 255), dtype=image.dtype)
            cv2.copyTo(image, combined_mask, isolated_color_image)
            gray_image_for_ocr = cv2.cvtColor(isolated_color_image, cv2.COLOR_BGR2GRAY)
            
            # Debug: Save the color-isolated grayscale image
            # debug_iso_path = os.path.join(os.path.dirname(output_path), f"debug_isolated_gray_{target_color_name}.png")
            # cv2.imwrite(debug_iso_path, gray_image_for_ocr)


        # --- Preprocessing for better OCR ---
        # print(f"Original dimensions for OCR: {gray_image_for_ocr.shape}") # Less verbose
        upscaled_gray = cv2.resize(gray_image_for_ocr, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=cv2.INTER_CUBIC)
        # print(f"Upscaled dimensions for OCR: {upscaled_gray.shape}") # Less verbose

        blurred_image = cv2.GaussianBlur(upscaled_gray, (5, 5), 0)
        preprocessed_image = cv2.adaptiveThreshold(
            blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        preprocessed_image = cv2.bitwise_not(preprocessed_image)
        
        # Debug: Save the final preprocessed image sent to Tesseract
        # debug_tess_path = os.path.join(os.path.dirname(output_path), f"debug_tesseract_input_{target_color_name}.png")
        # cv2.imwrite(debug_tess_path, preprocessed_image)

        # --- OCR Configuration ---
        custom_config = r'--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789'
        # print(f"Performing OCR on '{input_path}' for '{target_color_name}' numbers...") # Less verbose
        ocr_data = pytesseract.image_to_data(preprocessed_image, config=custom_config, output_type=Output.DICT)

        n_boxes = len(ocr_data['level'])
        detected_numbers_info = []

        with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(["Identified Number", "Pixel X", "Pixel Y"])

            for i in range(n_boxes):
                if ocr_data['level'][i] == 5:
                    text = ocr_data['text'][i].strip()
                    conf_str = ocr_data['conf'][i]

                    try:
                        conf = int(float(conf_str))
                    except ValueError:
                        conf = -1

                    if conf > MIN_CONFIDENCE and is_numeric_string(text):
                        x_up = ocr_data['left'][i]
                        y_up = ocr_data['top'][i]
                        w_up = ocr_data['width'][i]
                        h_up = ocr_data['height'][i]

                        x = x_up // UPSCALE_FACTOR
                        y = y_up // UPSCALE_FACTOR
                        w = max(1, w_up // UPSCALE_FACTOR)
                        h = max(1, h_up // UPSCALE_FACTOR)

                        mid_x = x + w // 2
                        mid_y = y + h // 2

                        detected_numbers_info.append({
                            "text": text, "x": x, "y": y, "w": w, "h": h,
                            "mid_x": mid_x, "mid_y": mid_y, "conf": conf
                        })
                        # print(f"  Detected: '{text}' at original coords ({mid_x},{mid_y}), Conf: {conf}%") # Less verbose
                        csv_writer.writerow([text, mid_x, mid_y])

        if not detected_numbers_info:
            print(f"No numbers detected for color '{target_color_name}' in '{input_path}' with sufficient confidence.")
        else:
            print(f"Detected {len(detected_numbers_info)} number(s) for '{target_color_name}' in '{input_path}'. Data: '{csv_path}'.")

        for info in detected_numbers_info:
            mid_x, mid_y = info['mid_x'], info['mid_y']
            coords_text = f"({mid_x},{mid_y})"
            cv2.line(output_image, (mid_x - CROSS_SIZE, mid_y), (mid_x + CROSS_SIZE, mid_y), CROSS_COLOR, FONT_THICKNESS)
            cv2.line(output_image, (mid_x, mid_y - CROSS_SIZE), (mid_x, mid_y + CROSS_SIZE), CROSS_COLOR, FONT_THICKNESS)
            (text_w, text_h), _ = cv2.getTextSize(coords_text, FONT, FONT_SCALE, FONT_THICKNESS)
            text_x = mid_x - text_w // 2
            text_y = mid_y + CROSS_SIZE + text_h + 4
            img_h, img_w = output_image.shape[:2]
            if text_x < 0: text_x = 0
            if text_x + text_w > img_w: text_x = img_w - text_w
            if text_y - text_h < 0 : text_y = mid_y - CROSS_SIZE - 4
            if text_y > img_h: text_y = img_h - 5
            cv2.putText(output_image, coords_text, (text_x, text_y), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        cv2.imwrite(output_path, output_image)
        # print(f"Output image with markings for '{target_color_name}' saved to '{output_path}'") # Less verbose

    except pytesseract.TesseractNotFoundError:
        print("Error: Tesseract is not installed or not found in your PATH.")
        print("Please install Tesseract OCR and ensure it's in PATH or set pytesseract.tesseract_cmd.")
    except ImportError as e:
        print(f"Error: A required library is not installed. {e}")
    except Exception as e:
        print(f"An unexpected error occurred in process_image for {input_path} ({target_color_name}): {e}")
        traceback.print_exc()

def process_folders(base_path):
    """Iterates through folders, checks JSON for color, and processes in.png files."""
    supported_colors_for_extraction = list(COLOR_RANGES_HSV.keys())
    print(f"Supported colors for extraction: {', '.join(supported_colors_for_extraction)}")

    for i in range(1, 3317):  # L0001 to L3316
        folder_name = f"L{i:04d}"
        folder_path = os.path.join(base_path, folder_name)

        if not os.path.isdir(folder_path):
            continue

        ys_folders = glob.glob(os.path.join(folder_path, "YS*"))
        if not ys_folders:
            continue

        for ys_folder in ys_folders:
            if not os.path.isdir(ys_folder):
                continue

            data_folders = glob.glob(os.path.join(ys_folder, "data*"))
            if not data_folders:
                continue

            for data_folder in data_folders:
                if not os.path.isdir(data_folder):
                    continue

                json_file_path = os.path.join(data_folder, "graph_out_fixed.json")
                if not os.path.isfile(json_file_path):
                    # print(f"graph_out_fixed.json not found in {data_folder}, skipping") # Can be verbose
                    continue

                target_color_name = None
                try:
                    with open(json_file_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    y_axis_desc = json_data.get("y_axis_type_description", "")
                    prefix = "double Y axis ("
                    if y_axis_desc.startswith(prefix) and y_axis_desc.endswith(")"):
                        color_from_json = y_axis_desc[len(prefix):-1].lower()
                        
                        if color_from_json != "black":
                            if color_from_json in supported_colors_for_extraction:
                                target_color_name = color_from_json
                            else:
                                print(f"Notice: Color '{color_from_json}' from '{y_axis_desc}' in {json_file_path} is valid but not in current COLOR_RANGES_HSV, skipping.")
                        # else: y_axis_desc is "double Y axis (black)", skipped by if color_from_json != "black"
                    # else: y_axis_desc does not match "double Y axis (color)" or is not "double Y axis"
                
                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON in {json_file_path}, skipping")
                    continue
                except Exception as e:
                    print(f"Error reading/parsing {json_file_path}: {e}, skipping")
                    continue

                if target_color_name:
                    input_file = os.path.join(data_folder, "in.png")
                    output_image_file = os.path.join(data_folder, "out_axis2.png")
                    output_csv_file = os.path.join(data_folder, "axis2.csv")

                    if not os.path.isfile(input_file):
                        print(f"in.png not found in {data_folder} (for color '{target_color_name}'), skipping")
                        continue

                    print(f"Processing: '{input_file}' for '{target_color_name}' numbers...")
                    process_image(input_file, output_image_file, output_csv_file, target_color_name)

# --- Main execution ---
if __name__ == "__main__":
    base_path = r".\HEA_finetune\Yield_Strength" 
    
    if not os.path.isdir(base_path):
        print(f"Error: Base path '{base_path}' does not exist. Please check the path.")
    else:
        print(f"Starting processing from base path: {base_path}")
        process_folders(base_path)
        print("Processing complete.")