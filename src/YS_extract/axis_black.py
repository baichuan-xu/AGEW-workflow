import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import os
import csv
import json
import glob

# --- Configuration ---
# IMPORTANT: Set this to your Tesseract installation path if it's not in your system's PATH
pytesseract.pytesseract.tesseract_cmd = r'tesseract'

MIN_CONFIDENCE = 10
CROSS_COLOR = (0, 255, 0)  # Green
TEXT_COLOR = (255, 0, 0)   # Blue
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
CROSS_SIZE = 8
UPSCALE_FACTOR = 2

# --- Helper function to check if a string is purely numeric ---
def is_numeric_string(s):
    """Checks if the string contains only digits."""
    return bool(s) and s.isdigit()

# --- Main processing function ---
def process_image(input_path, output_path, csv_path):
    """
    Loads an image, preprocesses it, detects numbers using OCR, marks them,
    saves the result image, and writes detected numbers and coordinates to a CSV file.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input image '{input_path}' not found.")
        print(f"Please ensure your graph image is named 'in.png' and is in the same directory.")
        print("Creating a basic sample 'in.png' for demonstration purposes.")
        sample_img = np.full((200, 400, 3), 220, dtype=np.uint8) # Light gray background
        cv2.putText(sample_img, "Sample 123", (50, 50), FONT, 1, (0,0,0), 2)
        cv2.putText(sample_img, "Test 456", (50, 150), FONT, 1, (0,0,0), 2)
        cv2.imwrite(input_path, sample_img)
        print(f"Sample '{input_path}' created. Please use your actual graph image and run the script again.")
        return

    try:
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error: OpenCV could not load image at '{input_path}'. Check file integrity and path.")
            return

        output_image = image.copy()

        # --- Preprocessing for better OCR ---
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(f"Original dimensions: {gray_image.shape}")
        # Upscaling can help OCR detect smaller characters
        upscaled_gray = cv2.resize(gray_image, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=cv2.INTER_CUBIC)
        print(f"Upscaled dimensions: {upscaled_gray.shape}")

        # Apply Gaussian blur to reduce noise
        blurred_image = cv2.GaussianBlur(upscaled_gray, (5, 5), 0)
        # Adaptive thresholding can be effective for varying lighting conditions
        preprocessed_image = cv2.adaptiveThreshold(
            blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        # Invert the image back for Tesseract (black text on white background)
        preprocessed_image = cv2.bitwise_not(preprocessed_image)

        # --- OCR Configuration ---
        custom_config = r'--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789'
        print(f"Performing OCR on preprocessed '{input_path}' (using PSM 11 and digit whitelist)...")
        ocr_data = pytesseract.image_to_data(preprocessed_image, config=custom_config, output_type=Output.DICT)

        n_boxes = len(ocr_data['level'])
        detected_numbers_info = []

        # --- Prepare CSV file ---
        with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(["Identified Number", "Pixel X", "Pixel Y"])

            for i in range(n_boxes):
                if ocr_data['level'][i] == 5: # Word level
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
                            "text": text,
                            "x": x, "y": y, "w": w, "h": h,
                            "mid_x": mid_x,
                            "mid_y": mid_y,
                            "conf": conf
                        })
                        print(f"  Detected: '{text}' at original coords (x:{x}, y:{y}, w:{w}, h:{h}), Midpoint: ({mid_x},{mid_y}), Conf: {conf}%")
                        
                        csv_writer.writerow([text, mid_x, mid_y])

        if not detected_numbers_info:
            print("No numbers detected with sufficient confidence after preprocessing.")
        else:
            print(f"Detected {len(detected_numbers_info)} number(s). Drawing markings and data saved to '{csv_path}'.")

        # --- Draw markings on the output image ---
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
            if text_y < text_h: text_y = mid_y - CROSS_SIZE - 4
            if text_y > img_h: text_y = img_h - 5

            cv2.putText(output_image, coords_text, (text_x, text_y), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        cv2.imwrite(output_path, output_image)
        print(f"Output image with markings saved to '{output_path}'")

    except pytesseract.TesseractNotFoundError:
        print("Error: Tesseract is not installed or not found in your PATH.")
        print("Please install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("Ensure the Tesseract installation directory is in your system's PATH.")
    except ImportError as e:
        print(f"Error: A required library is not installed. {e}")
        print("Please install them using: pip install opencv-python pytesseract numpy")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

def process_folders(base_path):
    """Iterates through folders and processes in.png files based on graph_out_fixed.json."""
    for i in range(1, 3317):
        folder_name = f"L{i:04d}"
        folder_path = os.path.join(base_path, folder_name)

        if not os.path.isdir(folder_path):
            print(f"Folder {folder_path} does not exist, skipping")
            continue

        print(f"Checking folder: {folder_path}")

        ys_folders = glob.glob(os.path.join(folder_path, "YS*"))
        if not ys_folders:
            print(f"No YS* folders found in {folder_path}, skipping")
            continue

        for ys_folder in ys_folders:
            if not os.path.isdir(ys_folder):
                print(f"{ys_folder} is not a directory, skipping")
                continue

            print(f"Found YS folder: {ys_folder}")

            data_folders = glob.glob(os.path.join(ys_folder, "data*"))
            if not data_folders:
                print(f"No data* folders found in {ys_folder}, skipping")
                continue

            for data_folder in data_folders:
                if not os.path.isdir(data_folder):
                    print(f"{data_folder} is not a directory, skipping")
                    continue

                print(f"Found data folder: {data_folder}")

                json_file = os.path.join(data_folder, "graph_out_fixed.json")
                if not os.path.isfile(json_file):
                    print(f"graph_out_fixed.json not found in {data_folder}, skipping")
                    continue

                # Read and check y_axis_type_description
                try:
                    with open(json_file, 'r') as f:
                        json_data = json.load(f)
                    y_axis_type = json_data.get("y_axis_type_description", "")
                    if y_axis_type != "single Y axis" and not y_axis_type.startswith("double Y axis"):
                        print(f"y_axis_type_description '{y_axis_type}' in {json_file} does not match requirements, skipping")
                        continue
                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON in {json_file}, skipping")
                    continue
                except Exception as e:
                    print(f"Error reading {json_file}: {e}, skipping")
                    continue

                # Process in.png if conditions are met
                input_file = os.path.join(data_folder, "in.png")
                output_image = os.path.join(data_folder, "out_axis.png")
                output_csv = os.path.join(data_folder, "axis.csv")

                if not os.path.isfile(input_file):
                    print(f"in.png not found in {data_folder}, skipping")
                    continue

                print(f"Processing {input_file}...")
                process_image(input_file, output_image, output_csv)

# Usage
if __name__ == "__main__":
    base_path = r".\HEA_finetune\Yield_Strength"
    process_folders(base_path)