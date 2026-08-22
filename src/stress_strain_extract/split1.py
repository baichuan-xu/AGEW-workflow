import os
import csv
import sys
from PIL import Image

# --- Configuration ---
# The base directory where the L0001, L0002, etc. folders are located.
# "." means the current directory where the script is run.
BASE_DIRECTORY = "." 
# The full path for the output file that logs folders that were skipped.
# The script will create the directory if it doesn't exist.
OTHER_CSV_OUTPUT_PATH = r".\stress-strain2\other.csv"


def get_crop_definitions(width, height):
    """
    Returns a dictionary mapping figure modes to their cropping coordinates.
    Coordinates are calculated based on the actual image width and height.
    """
    return {
        "2 figures": {
            "left": (0, 0, width // 2, height),
            "right": (width // 2, 0, width, height),
        },
        "3 figures": {
            "left": (0, 0, width // 3, height),
            "mid": (width // 3, 0, 2 * width // 3, height),
            "right": (2 * width // 3, 0, width, height),
        },
        "4 figures": {
            "leftupper": (0, 0, width // 2, height // 2),
            "leftdown": (0, height // 2, width // 2, height),
            "rightupper": (width // 2, 0, width, height // 2),
            "rightdown": (width // 2, height // 2, width, height),
        },
        "6 figures": {
            "leftupper": (0, 0, width // 3, height // 2),
            "leftdown": (0, height // 2, width // 3, height),
            "midupper": (width // 3, 0, 2 * width // 3, height // 2),
            "middown": (width // 3, height // 2, 2 * width // 3, height),
            "rightupper": (2 * width // 3, 0, width, height // 2),
            "rightdown": (2 * width // 3, height // 2, width, height),
        },
    }

def process_folders_and_crop_images():
    """
    Main function to loop through specified folders, read CSV files,
    and crop images based on the defined rules.
    """
    print(f"Starting image processing in directory: {os.path.abspath(BASE_DIRECTORY)}")
    skipped_paths = []
    
    # 1. Loop through the subfolders L0001 to L2594.
    for i in range(1, 3356):
        folder_name = f"L{i:04d}"
        l_folder_path = os.path.join(BASE_DIRECTORY, folder_name)

        if not os.path.isdir(l_folder_path):
            continue

        print(f"\n--- Processing folder: {folder_name} ---")
            
        # 1. Loop through YS subfolders.
        found_ys_folder = False
        for item_name in os.listdir(l_folder_path):
            ys_folder_path = os.path.join(l_folder_path, item_name)
            if os.path.isdir(ys_folder_path) and item_name.startswith("YS"):
                found_ys_folder = True
                print(f"  -> Found YS subfolder: {item_name}")

                # Corrected Logic: Check for CSV and PNG inside the YS subfolder.
                csv_path = os.path.join(ys_folder_path, 'figure_split.csv')
                image_path = os.path.join(ys_folder_path, 'in.png')

                if not os.path.exists(csv_path):
                    print(f"     [!] 'figure_split.csv' not found in {item_name}. Skipping this subfolder.")
                    continue
                
                if not os.path.exists(image_path):
                    print(f"     [!] 'in.png' not found in {item_name}. Skipping this subfolder.")
                    continue
                
                try:
                    with open(csv_path, mode='r', newline='', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        header = next(reader)
                except (StopIteration, IOError) as e:
                    print(f"     [!] Could not read '{csv_path}': {e}. Skipping this subfolder.")
                    continue
                
                if not header:
                    print(f"     [!] 'figure_split.csv' in {item_name} is empty. Skipping this subfolder.")
                    continue

                figure_mode = header[0].strip()
                parts_to_crop = [part.strip() for part in header[1:]]

                # 3. Skip if "1 figure".
                if figure_mode == "1 figure":
                    print("     - Mode is '1 figure'. No operations needed.")
                    continue
                
                # 8. Check if the mode is one of the recognized types.
                valid_modes = ["2 figures", "3 figures", "4 figures", "6 figures"]
                if figure_mode not in valid_modes:
                    print(f"     - Mode '{figure_mode}' is not recognized. Logging and skipping.")
                    skipped_paths.append(ys_folder_path)
                    continue

                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                        crop_definitions = get_crop_definitions(width, height)
                        
                        current_mode_definitions = crop_definitions.get(figure_mode, {})
                        
                        if not parts_to_crop:
                            print(f"     [!] No crop parts specified in CSV (e.g., 'left', 'right'). Cannot proceed.")
                            continue

                        # 4, 5, 6, 7. Perform cropping based on the parts listed in the CSV.
                        split_count = 0
                        for part_name in parts_to_crop:
                            if part_name in current_mode_definitions:
                                split_count += 1
                                coords = current_mode_definitions[part_name]
                                
                                cropped_img = img.crop(coords)
                                
                                output_folder_name = f"split{split_count}"
                                output_folder_path = os.path.join(ys_folder_path, output_folder_name)
                                os.makedirs(output_folder_path, exist_ok=True)
                                
                                output_image_path = os.path.join(output_folder_path, 'in.png')
                                cropped_img.save(output_image_path)
                                print(f"     - Created '{output_folder_name}' and saved crop for '{part_name}'.")
                            else:
                                print(f"     [!] Warning: Unknown part '{part_name}' for mode '{figure_mode}'.")

                except Exception as e:
                    print(f"     [!] ERROR processing image '{image_path}': {e}")
        
        if not found_ys_folder:
            print("  - No subfolders starting with 'YS' were found.")

    # 8. After all loops, write the skipped folder paths to the output file.
    if skipped_paths:
        print(f"\nWriting {len(skipped_paths)} skipped folder paths to '{OTHER_CSV_OUTPUT_PATH}'...")
        try:
            # Ensure the output directory exists.
            output_dir = os.path.dirname(OTHER_CSV_OUTPUT_PATH)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            with open(OTHER_CSV_OUTPUT_PATH, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Skipped_Folder_Paths"])
                for path in skipped_paths:
                    writer.writerow([path])
            print("  - Done writing skipped paths.")
        except IOError as e:
            print(f"  [!] ERROR: Could not write to '{OTHER_CSV_OUTPUT_PATH}': {e}")


if __name__ == "__main__":
    # A small check to ensure the Pillow library is installed.
    try:
        from PIL import Image
    except ImportError:
        print("ERROR: The 'Pillow' library is required to run this script.")
        print("Please install it using the command: pip install Pillow")
        sys.exit(1)
        
    process_folders_and_crop_images()
    
    print("\n==========================")
    print("  Processing complete.")
    print("==========================")

