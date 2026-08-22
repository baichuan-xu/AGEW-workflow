import cv2
import numpy as np
import os
import csv

# --- Configuration ---
BASE_PATH = r".\HEA_finetune\Yield_Strength"
SUBFOLDER_PREFIX = "L"
NUM_SUBFOLDERS = 3316
YS_PREFIX = "YS"
INPUT_IMAGE_NAME = "in.png"
OUTPUT_IMAGE_NAME = "in_grid.png"
OUTPUT_CSV_NAME = "grid.csv"

# Grid configuration
GRID_SPACING = 50  # Draw a grid line every 50 pixels
GRID_COLOR = (128, 128, 128)  # Gray color for grid lines
GRID_THICKNESS = 1

def save_grid_dimensions_to_csv(image_width, image_height, csv_path):
    """
    Saves the minimum and maximum pixel coordinates of the image to a CSV file.
    """
    x_min = 0
    x_max = image_width - 1
    y_min = 0
    y_max = image_height - 1

    header = ['X_min', 'X_max', 'Y_min', 'Y_max']
    data = [x_min, x_max, y_min, y_max]

    try:
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerow(data)
        print(f"Grid dimensions saved to '{csv_path}'")
        return True
    except IOError as e:
        print(f"Error writing to '{csv_path}': {e}")
        return False
    except Exception as e:
        print(f"Unexpected error writing to '{csv_path}': {e}")
        return False

def add_grid_to_image(input_path, output_image_path, output_csv_path):
    """
    Add a grid to the input image and save grid dimensions to CSV.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input image '{input_path}' not found.")
        return False

    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Could not load image '{input_path}'.")
        return False

    output_image = image.copy()
    height, width, _ = output_image.shape
    print(f"Processing image: {input_path} ({width}x{height} pixels)")

    # Draw vertical lines
    for x in range(0, width, GRID_SPACING):
        cv2.line(output_image, (x, 0), (x, height), GRID_COLOR, GRID_THICKNESS)
        cv2.putText(output_image, str(x), (x + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRID_COLOR, 2)

    # Draw horizontal lines
    for y in range(0, height, GRID_SPACING):
        cv2.line(output_image, (0, y), (width, y), GRID_COLOR, GRID_THICKNESS)
        cv2.putText(output_image, str(y), (2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRID_COLOR, 2)

    # Save the output image
    try:
        cv2.imwrite(output_image_path, output_image)
        print(f"Output image saved to '{output_image_path}'")
    except Exception as e:
        print(f"Error saving '{output_image_path}': {e}")
        return False

    # Save grid dimensions to CSV
    return save_grid_dimensions_to_csv(width, height, output_csv_path)

def process_subfolders():
    """
    Iterate through L0001 to L3316, process YS subfolders, and add grids to in.png.
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
                input_path = os.path.join(ys_subfolder_path, INPUT_IMAGE_NAME)
                output_image_path = os.path.join(ys_subfolder_path, OUTPUT_IMAGE_NAME)
                output_csv_path = os.path.join(ys_subfolder_path, OUTPUT_CSV_NAME)

                if not os.path.exists(input_path):
                    print(f"Image {input_path} not found. Skipping...")
                    continue

                print(f"\nProcessing: {input_path}")
                add_grid_to_image(input_path, output_image_path, output_csv_path)

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