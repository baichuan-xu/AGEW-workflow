from PIL import Image
import os
import csv
import re

# --- Configuration ---
BASE_PATH = r".\HEA_finetune\Yield_Strength"
SUBFOLDER_PREFIX = "L"
NUM_SUBFOLDERS = 3316
YS_PREFIX = "YS"
INPUT_IMAGE_NAME = "in.png"
CSV_PREFIX = "split_range"
CSV_EXTENSION = ".csv"
OUTPUT_IMAGE_PREFIX = "in_split"

def load_crop_coordinates_from_csv(csv_path):
    """
    Loads crop coordinates from a specified CSV file.

    The CSV file is expected to have the coordinates in the second row:
    - Column 1: x_min (X_start)
    - Column 2: x_max (X_end)
    - Column 3: y_min (Y_start)
    - Column 4: y_max (Y_end)

    Args:
        csv_path (str): The full path to the CSV file.

    Returns:
        tuple: A tuple (X_start, Y_start, X_end, Y_end) if successful, None otherwise.
    """
    try:
        with open(csv_path, mode='r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            rows = list(reader)
            
            if len(rows) < 2:
                print(f"Error: CSV file '{csv_path}' does not have enough rows (expected at least 2).")
                return None
            
            target_row = rows[1]  # Second row (data row)
            
            if len(target_row) < 4:
                print(f"Error: Data row in CSV file '{csv_path}' does not have enough columns (expected 4).")
                return None
            
            x_start = int(float(target_row[0]))  # x_min
            x_end = int(float(target_row[1]))    # x_max
            y_start = int(float(target_row[2]))  # y_min
            y_end = int(float(target_row[3]))    # y_max
            
            return (x_start, y_start, x_end, y_end)

    except FileNotFoundError:
        print(f"Error: The CSV file '{csv_path}' could not be opened.")
        return None
    except ValueError:
        print(f"Error: Could not convert CSV data to integers in '{csv_path}'.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading '{csv_path}': {e}")
        return None

def crop_and_save_as_image(input_image_path, output_image_path, crop_box):
    """
    Crops an image and saves the cropped portion as a new image file.

    Args:
        input_image_path (str): Full path to the input image file.
        output_image_path (str): Full path to save the output cropped image file.
        crop_box (tuple): A tuple (left, upper, right, lower) defining the pixel coordinates for cropping.
    """
    if crop_box is None:
        print("Error: Crop box coordinates were not provided.")
        return False

    try:
        if not os.path.exists(input_image_path):
            print(f"Error: Input image '{input_image_path}' not found.")
            return False

        img = Image.open(input_image_path)
        img_width, img_height = img.size
        left, upper, right, lower = crop_box

        if not (0 <= left < right <= img_width and 0 <= upper < lower <= img_height):
            print(f"Error: Crop box {crop_box} is invalid or outside image dimensions ({img_width}x{img_height}).")
            return False

        cropped_img = img.crop(crop_box)

        if cropped_img.size[0] == 0 or cropped_img.size[1] == 0:
            print(f"Error: Cropping resulted in an empty image (size: {cropped_img.size}).")
            return False

        cropped_img.save(output_image_path)
        print(f"Successfully cropped '{input_image_path}' (Original size: {img_width}x{img_height}).")
        print(f"Saved to '{output_image_path}' (Size: {cropped_img.size[0]}x{cropped_img.size[1]}).")
        return True

    except FileNotFoundError:
        print(f"Error: The image file '{input_image_path}' could not be opened.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def process_subfolders():
    """
    Iterate through L0001 to L3316, process YS subfolders with split_range*.csv files.
    """
    for i in range(1, NUM_SUBFOLDERS + 1):
        subfolder_name = f"{SUBFOLDER_PREFIX}{i:04d}"
        subfolder_path = os.path.join(BASE_PATH, subfolder_name)

        if not os.path.exists(subfolder_path):
            print(f"Subfolder {subfolder_path} does not exist. Skipping...")
            continue

        # Check for YS subfolders
        for item in os.listdir(subfolder_path):
            if item.startswith(YS_PREFIX) and os.path.isdir(os.path.join(subfolder_path, item)):
                ys_subfolder_path = os.path.join(subfolder_path, item)
                input_image_path = os.path.join(ys_subfolder_path, INPUT_IMAGE_NAME)

                if not os.path.exists(input_image_path):
                    print(f"Image {input_image_path} not found. Skipping...")
                    continue

                # Find all split_range*.csv files
                csv_files = [f for f in os.listdir(ys_subfolder_path) if f.startswith(CSV_PREFIX) and f.endswith(CSV_EXTENSION)]
                if not csv_files:
                    print(f"No {CSV_PREFIX}*.csv files found in {ys_subfolder_path}. Skipping...")
                    continue

                print(f"\nProcessing subfolder: {ys_subfolder_path}")
                for csv_file in csv_files:
                    # Extract the number from the CSV filename (e.g., 1 from split_range1.csv)
                    match = re.search(r'split_range(\d+)\.csv', csv_file)
                    if not match:
                        print(f"Skipping {csv_file}: Invalid filename format.")
                        continue
                    csv_number = match.group(1)

                    csv_path = os.path.join(ys_subfolder_path, csv_file)
                    output_image = f"{OUTPUT_IMAGE_PREFIX}{csv_number}.png"
                    output_image_path = os.path.join(ys_subfolder_path, output_image)

                    # Load crop coordinates
                    crop_coordinates = load_crop_coordinates_from_csv(csv_path)
                    if crop_coordinates:
                        crop_and_save_as_image(input_image_path, output_image_path, crop_coordinates)
                    else:
                        print(f"Failed to load coordinates from {csv_path}. Skipping...")

def main():
    """
    Main function to verify base path and start processing.
    """
    if not os.path.exists(BASE_PATH):
        print(f"Error: Base path '{BASE_PATH}' does not exist.")
        return

    print(f"Processing subfolders in {BASE_PATH}...")
    process_subfolders()
    print("\nBatch processing complete.")

if __name__ == "__main__":
    main()