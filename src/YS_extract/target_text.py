import cv2
import pytesseract
from pytesseract import Output
import os
import csv
import numpy as np

# --- Configuration ---
BASE_PATH = r".\HEA_finetune\Yield_Strength"
SUBFOLDER_PREFIX = "L"
NUM_SUBFOLDERS = 3316
YS_PREFIX = "YS"
INPUT_IMAGE_NAME = "in.png"
OUTPUT_CSV_NAME = "target_text.csv"
pytesseract.pytesseract.tesseract_cmd = r'tesseract'
MIN_CONFIDENCE_WORD = 3
TARGET_WORD_1 = "yield"
TARGET_WORD_2 = "strength"
OCR_PSM_CONFIG = r'--oem 3 --psm 11'

def transform_coordinates(x_in_rotated_frame, y_in_rotated_frame, original_img_height, original_img_width, rotation_type):
    """
    Transforms coordinates from a rotated image frame back to the original image frame.
    """
    if rotation_type is None:
        return int(x_in_rotated_frame), int(y_in_rotated_frame)
    elif rotation_type == "cw":
        return int(y_in_rotated_frame), int(original_img_height - 1 - x_in_rotated_frame)
    elif rotation_type == "ccw":
        return int(original_img_width - 1 - y_in_rotated_frame), int(x_in_rotated_frame)
    return int(x_in_rotated_frame), int(y_in_rotated_frame)

def find_target_phrase_in_ocr_data(ocr_data, original_img_h, original_img_w, rotation_type=None):
    """
    Finds the target phrase (TARGET_WORD_1 + TARGET_WORD_2) in OCR data.
    """
    found_phrases = []
    n_boxes = len(ocr_data['level'])

    yield_detections = []
    strength_detections = []

    for i in range(n_boxes):
        try:
            conf = int(float(ocr_data['conf'][i]))
        except ValueError:
            conf = -1

        if conf > MIN_CONFIDENCE_WORD and ocr_data['text'][i].strip():
            text_lower = ocr_data['text'][i].strip().lower()
            box_data = {
                'left': ocr_data['left'][i],
                'top': ocr_data['top'][i],
                'width': ocr_data['width'][i],
                'height': ocr_data['height'][i],
                'text_orig': ocr_data['text'][i].strip(),
            }
            if box_data['width'] <= 0 or box_data['height'] <= 0:
                continue
            if text_lower == TARGET_WORD_1:
                yield_detections.append(box_data)
            elif text_lower == TARGET_WORD_2:
                strength_detections.append(box_data)

    for y_box in yield_detections:
        for s_box in strength_detections:
            avg_word_height = (y_box['height'] + s_box['height']) / 2.0
            if avg_word_height <= 1:
                avg_word_height = 10

            y_box_right = y_box['left'] + y_box['width']
            y_box_bottom = y_box['top'] + y_box['height']
            y_box_center_x = y_box['left'] + y_box['width'] / 2.0
            y_box_center_y = y_box['top'] + y_box['height'] / 2.0
            s_box_center_x = s_box['left'] + s_box['width'] / 2.0
            s_box_center_y = s_box['top'] + s_box['height'] / 2.0

            is_horizontally_plausible = (
                y_box['left'] < s_box['left'] and
                abs(y_box_center_y - s_box_center_y) < avg_word_height * 0.75 and
                (s_box['left'] - y_box_right) < (avg_word_height * 1.5) and
                (s_box['left'] - y_box_right) > (-0.5 * y_box['width'])
            )
            if is_horizontally_plausible:
                combined_left = y_box['left']
                combined_top = min(y_box['top'], s_box['top'])
                combined_right = s_box['left'] + s_box['width']
                combined_bottom = max(y_box_bottom, s_box['top'] + s_box['height'])
                center_x_current_frame = combined_left + (combined_right - combined_left) / 2.0
                center_y_current_frame = combined_top + (combined_bottom - combined_top) / 2.0
                orig_x, orig_y = transform_coordinates(
                    center_x_current_frame, center_y_current_frame,
                    original_img_h, original_img_w, rotation_type
                )
                orientation = "Horizontal" if rotation_type is None else "Clockwise" if rotation_type == "cw" else "Counterclockwise"
                found_phrases.append({
                    'x': orig_x, 'y': orig_y,
                    'orientation': orientation,
                    'text': f"{y_box['text_orig']} {s_box['text_orig']}"
                })
                continue

            max_word_width = max(y_box['width'], s_box['width'])
            if max_word_width <= 1:
                max_word_width = avg_word_height
            is_vertically_plausible = (
                y_box['top'] < s_box['top'] and
                abs(y_box_center_x - s_box_center_x) < max_word_width * 1.0 and
                (s_box['top'] - y_box_bottom) < avg_word_height and
                (s_box['top'] - y_box_bottom) > (-0.5 * y_box['height'])
            )
            if is_vertically_plausible:
                combined_left = min(y_box['left'], s_box['left'])
                combined_top = y_box['top']
                combined_right = max(y_box_right, s_box['left'] + s_box['width'])
                combined_bottom = s_box['top'] + s_box['height']
                center_x_current_frame = combined_left + (combined_right - combined_left) / 2.0
                center_y_current_frame = combined_top + (combined_bottom - combined_top) / 2.0
                orig_x, orig_y = transform_coordinates(
                    center_x_current_frame, center_y_current_frame,
                    original_img_h, original_img_w, rotation_type
                )
                orientation = "Horizontal" if rotation_type is None else "Clockwise" if rotation_type == "cw" else "Counterclockwise"
                found_phrases.append({
                    'x': orig_x, 'y': orig_y,
                    'orientation': orientation,
                    'text': f"{y_box['text_orig']} {s_box['text_orig']}"
                })
    return found_phrases

def process_image(image_path, output_csv_path):
    """
    Process a single image to extract target phrase coordinates and save to CSV with orientation.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image '{image_path}'.")
        return False

    original_h, original_w = image.shape[:2]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    all_found_coordinates_with_text = []

    orientations_to_process = [
        (None, "Original"),
        ("cw", "90 degrees Clockwise"),
        ("ccw", "90 degrees Counter-Clockwise")
    ]

    for rotation_code, description in orientations_to_process:
        print(f"Processing: {description} orientation for {image_path}...")
        current_image_for_ocr = gray_image
        if rotation_code == "cw":
            current_image_for_ocr = cv2.rotate(gray_image, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_code == "ccw":
            current_image_for_ocr = cv2.rotate(gray_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        try:
            ocr_data = pytesseract.image_to_data(
                current_image_for_ocr,
                config=OCR_PSM_CONFIG,
                output_type=Output.DICT
            )
        except pytesseract.TesseractNotFoundError:
            print("CRITICAL ERROR: Tesseract not found. Please install or update 'pytesseract.pytesseract.tesseract_cmd'.")
            return False
        except Exception as e:
            print(f"Error during OCR on {description} orientation: {e}")
            continue

        coords_found_in_orientation = find_target_phrase_in_ocr_data(
            ocr_data, original_h, original_w, rotation_type=rotation_code
        )
        all_found_coordinates_with_text.extend(coords_found_in_orientation)

    unique_coords_to_write = []
    seen_rounded_coords = set()

    if all_found_coordinates_with_text:
        print("\nPotential Detections (before deduplication):")
        for item in all_found_coordinates_with_text:
            print(f"  Text: '{item['text']}', Orientation: {item['orientation']}, Coords: ({item['x']}, {item['y']})")

    for item in all_found_coordinates_with_text:
        rounding_precision = 5
        rounded_x = round(item['x'] / rounding_precision) * rounding_precision
        rounded_y = round(item['y'] / rounding_precision) * rounding_precision
        if (rounded_x, rounded_y) not in seen_rounded_coords:
            unique_coords_to_write.append({
                'orientation': item['orientation'],
                'x': item['x'],
                'y': item['y']
            })
            seen_rounded_coords.add((rounded_x, rounded_y))

    if unique_coords_to_write:
        print(f"\nFound {len(unique_coords_to_write)} unique instance(s) in {image_path}.")
        try:
            with open(output_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Orientation", "X", "Y"])
                for item in unique_coords_to_write:
                    writer.writerow([item['orientation'], item['x'], item['y']])
            print(f"Coordinates saved to '{output_csv_path}'.")
            return True
        except IOError:
            print(f"Error: Could not write to '{output_csv_path}'.")
            return False
    else:
        print(f"\nTarget phrase not found in '{image_path}'.")
        return False

def main():
    """
    Iterate through L0001 to L3316, process YS subfolders, and extract text coordinates.
    """
    if not os.path.exists(BASE_PATH):
        print(f"Error: Base path '{BASE_PATH}' does not exist.")
        return

    print(f"Processing subfolders in {BASE_PATH}...")
    for i in range(1, NUM_SUBFOLDERS + 1):
        subfolder_name = f"{SUBFOLDER_PREFIX}{i:04d}"
        subfolder_path = os.path.join(BASE_PATH, subfolder_name)

        if not os.path.exists(subfolder_path):
            print(f"Subfolder {subfolder_path} does not exist. Skipping...")
            continue

        # Check for subfolders starting with YS
        for item in os.listdir(subfolder_path):
            if item.startswith(YS_PREFIX) and os.path.isdir(os.path.join(subfolder_path, item)):
                ys_subfolder_path = os.path.join(subfolder_path, item)
                image_path = os.path.join(ys_subfolder_path, INPUT_IMAGE_NAME)
                output_csv_path = os.path.join(ys_subfolder_path, OUTPUT_CSV_NAME)

                if not os.path.exists(image_path):
                    print(f"Image {image_path} not found. Skipping...")
                    continue

                print(f"\nProcessing image: {image_path}")
                process_image(image_path, output_csv_path)

    print("\nBatch processing complete.")

if __name__ == "__main__":
    main()