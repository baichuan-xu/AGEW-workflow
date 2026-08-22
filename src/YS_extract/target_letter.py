import cv2
import pytesseract
from pytesseract import Output
import os
import csv
import numpy as np

# --- Configuration ---
BASE_PATH = r".\HEA_finetune\Yield_Strength" # Base directory for processing
SUBFOLDER_PREFIX = "L" # Prefix for the numbered subfolders
NUM_SUBFOLDERS = 3316 # Number of Lxxxx subfolders to iterate through
YS_PREFIX = "YS" # Prefix for subfolders within Lxxxx folders
INPUT_IMAGE_NAME = "in.png" # Name of the image file to process in each YS subfolder
TARGET_LETTER_CSV_NAME = "target_letter.csv" # Name of the output CSV file for letters

# IMPORTANT: Set the path to your Tesseract executable if it's not in your system's PATH
pytesseract.pytesseract.tesseract_cmd = r'tesseract' 

MIN_CONFIDENCE_WORD = 3  # User-defined minimum confidence for OCR word detection
TARGET_LETTERS = ["a", "b", "c", "d", "e", "f"] # Letters to search for
OCR_PSM_CONFIG = r'--oem 3 --psm 11' # Tesseract Page Segmentation Mode configuration
PADDING_SIZE = 15 # Pixels of white padding to add on all sides. Experiment with 10, 15, or 20.


def transform_coordinates(x_in_frame, y_in_frame, original_img_height, original_img_width, rotation_type):
    """
    Transforms coordinates. In this version, since rotation_type will always be None,
    it effectively just ensures integer coordinates.
    """
    if rotation_type is None: # No rotation
        return int(x_in_frame), int(y_in_frame)
    # The following rotation logic will not be triggered with the current process_image implementation
    elif rotation_type == "cw": # Rotated 90 degrees clockwise
        return int(y_in_frame), int(original_img_height - 1 - x_in_frame)
    elif rotation_type == "ccw": # Rotated 90 degrees counter-clockwise
        return int(original_img_width - 1 - y_in_frame), int(x_in_frame)
    return int(x_in_frame), int(y_in_frame) # Fallback

def find_standalone_letters_in_ocr_data(ocr_data, original_img_h, original_img_w, rotation_type=None, padding_applied=0):
    """
    Finds standalone target letters (e.g., "a" or "(a)") in OCR data.
    Adjusts coordinates if padding was applied to the image given to OCR.
    rotation_type will be None in the current simplified workflow.
    Returns a list of dictionaries, each including the normalized letter and the raw text detected.
    """
    found_letters_info = []
    n_boxes = len(ocr_data['level'])

    for i in range(n_boxes):
        if ocr_data['level'][i] != 5: # Process only word-level detections
            continue

        try:
            conf = int(float(ocr_data['conf'][i]))
        except ValueError:
            conf = -1 

        if conf > MIN_CONFIDENCE_WORD and ocr_data['text'][i].strip():
            raw_text_from_ocr = ocr_data['text'][i].strip() # This is the source text
            text_cleaned_lower = raw_text_from_ocr.lower()

            detected_char_to_store = None

            if len(text_cleaned_lower) == 1 and text_cleaned_lower in TARGET_LETTERS:
                detected_char_to_store = text_cleaned_lower
            elif len(text_cleaned_lower) == 3 and \
                 text_cleaned_lower.startswith("(") and \
                 text_cleaned_lower.endswith(")") and \
                 text_cleaned_lower[1] in TARGET_LETTERS:
                detected_char_to_store = text_cleaned_lower[1]

            if detected_char_to_store:
                # Coordinates from OCR are relative to the image Tesseract processed (padded image)
                x_coord_padded = ocr_data['left'][i]
                y_coord_padded = ocr_data['top'][i]
                w_coord = ocr_data['width'][i]
                h_coord = ocr_data['height'][i]

                if w_coord <= 0 or h_coord <= 0: 
                    continue
                
                # Center in the padded image's coordinate system
                center_x_padded_frame = x_coord_padded + w_coord / 2.0
                center_y_padded_frame = y_coord_padded + h_coord / 2.0

                # Adjust center to be relative to the original image content (before padding)
                center_x_unpadded_frame = center_x_padded_frame - padding_applied
                center_y_unpadded_frame = center_y_padded_frame - padding_applied

                # Transform coordinates (rotation_type will be None here)
                # original_img_h and original_img_w are for the UNPADDED original image
                orig_x, orig_y = transform_coordinates(
                    center_x_unpadded_frame, center_y_unpadded_frame,
                    original_img_h, original_img_w, rotation_type 
                )
                
                orientation_info = "Original" # Will always be "Original" in this workflow
                
                found_letters_info.append({
                    'letter': detected_char_to_store,      # Normalized letter (e.g., 'a')
                    'x': orig_x, 
                    'y': orig_y,
                    'orientation_info': orientation_info, 
                    'raw_detected_text': raw_text_from_ocr # The source text (e.g., 'A', '(a)', 'a')
                })
                
    return found_letters_info

def process_image(image_path, output_csv_path):
    """
    Processes a single image (original orientation only, with padding) to extract coordinates 
    and source of target standalone letters, then saves them to a CSV file.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image '{image_path}'. Skipping this image.")
        return False

    original_h, original_w = image.shape[:2] # Dimensions of the original, unpadded image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # --- Add Padding ---
    # Add a white border around the grayscale image
    padded_gray_image = cv2.copyMakeBorder(
        gray_image,
        PADDING_SIZE,  # top
        PADDING_SIZE,  # bottom
        PADDING_SIZE,  # left
        PADDING_SIZE,  # right
        cv2.BORDER_CONSTANT,
        value=[255]    # White color for padding (for grayscale)
    )
    # --- End Padding ---
    
    all_found_letters_details = []

    print(f"  Processing: Original orientation for {os.path.basename(image_path)} (with {PADDING_SIZE}px padding)...")
    
    try:
        # Perform OCR on the PADDED image
        ocr_data = pytesseract.image_to_data(
            padded_gray_image, # Use the padded image here
            config=OCR_PSM_CONFIG,
            output_type=Output.DICT
        )
    except pytesseract.TesseractNotFoundError:
        print("CRITICAL ERROR: Tesseract is not installed or not found in your PATH.")
        print(f"Please check Tesseract installation and the path: {pytesseract.pytesseract.tesseract_cmd}")
        return False 
    except Exception as e:
        print(f"    Error during Tesseract OCR on Original orientation: {e}")
        return False 

    # Pass PADDING_SIZE for coordinate adjustment.
    # original_h and original_w are for the original UNPADDED image.
    coords_found = find_standalone_letters_in_ocr_data(
        ocr_data, original_h, original_w, rotation_type=None, padding_applied=PADDING_SIZE
    )
    all_found_letters_details.extend(coords_found)

    unique_letters_to_write = []
    seen_letter_at_rounded_coords = set() 

    if all_found_letters_details:
        print(f"\n  Potential Detections for {os.path.basename(image_path)} (before deduplication, coords are for original image):")
        for item in all_found_letters_details:
            print(f"    Letter: '{item['letter']}' (from Source: '{item['raw_detected_text']}'), Coords: ({item['x']}, {item['y']})")

    for item in all_found_letters_details:
        rounding_precision = 5 
        rounded_x = round(item['x'] / rounding_precision) * rounding_precision
        rounded_y = round(item['y'] / rounding_precision) * rounding_precision
        
        deduplication_key = (item['letter'], rounded_x, rounded_y) 

        if deduplication_key not in seen_letter_at_rounded_coords:
            unique_letters_to_write.append({
                'letter': item['letter'],
                'x': item['x'], 
                'y': item['y'],
                'source': item['raw_detected_text'] 
            })
            seen_letter_at_rounded_coords.add(deduplication_key)

    if unique_letters_to_write:
        unique_letters_to_write.sort(key=lambda d: (d['letter'], d['y'], d['x']))
        
        print(f"\n  Found {len(unique_letters_to_write)} unique letter instance(s) in {os.path.basename(image_path)}.")
        try:
            with open(output_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Letter", "X", "Y", "Source"]) 
                for item in unique_letters_to_write:
                    writer.writerow([item['letter'], item['x'], item['y'], item['source']]) 
            print(f"  Letter coordinates and source saved to '{output_csv_path}'.")
            return True 
        except IOError as e:
            print(f"  Error: Could not write to CSV file '{output_csv_path}'. Reason: {e}")
            return False
    else:
        print(f"\n  Target letters ({', '.join(TARGET_LETTERS)}) not found in '{os.path.basename(image_path)}'.")
        return False

def main():
    """
    Iterates through specified directory structure, processes 'in.png' files 
    (original orientation only, with padding), extracts target letter coordinates and source, 
    and saves them to 'target_letter.csv'.
    """
    if not os.path.exists(BASE_PATH):
        print(f"Error: Base path '{BASE_PATH}' does not exist. Please check the path.")
        return

    print(f"Starting batch processing in base directory: {BASE_PATH}")
    print(f"Processing only original image orientation (no rotations).")
    print(f"Applying {PADDING_SIZE}px padding to images before OCR.")
    processed_images_count = 0
    images_with_letters_found_count = 0

    for i in range(1, NUM_SUBFOLDERS + 1):
        subfolder_name = f"{SUBFOLDER_PREFIX}{i:04d}"
        subfolder_path = os.path.join(BASE_PATH, subfolder_name)

        if not os.path.isdir(subfolder_path):
            continue 

        for item_in_l_folder in os.listdir(subfolder_path):
            if item_in_l_folder.startswith(YS_PREFIX) and \
               os.path.isdir(os.path.join(subfolder_path, item_in_l_folder)):
                
                ys_subfolder_path = os.path.join(subfolder_path, item_in_l_folder)
                image_path_to_process = os.path.join(ys_subfolder_path, INPUT_IMAGE_NAME)
                output_csv_for_letters = os.path.join(ys_subfolder_path, TARGET_LETTER_CSV_NAME)

                if not os.path.exists(image_path_to_process):
                    continue 
                
                processed_images_count += 1
                print(f"\nProcessing image: {image_path_to_process}")
                if process_image(image_path_to_process, output_csv_for_letters):
                    images_with_letters_found_count +=1
    
    print("\n-----------------------------------------")
    print("Batch processing summary:")
    print(f"  Total 'L' subfolders scanned (up to L{NUM_SUBFOLDERS:04d}).")
    print(f"  Total images targeted for processing: {processed_images_count}")
    print(f"  Images where target letters were found and CSV created: {images_with_letters_found_count}")
    print("Batch processing complete.")

if __name__ == "__main__":
    main()