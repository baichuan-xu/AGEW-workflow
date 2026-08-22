import os
import shutil
import csv
import re

# --- Configuration ---
BASE_PATH = r".\HEA_finetune\Yield_Strength"
SUBFOLDER_PREFIX = "L"
NUM_SUBFOLDERS = 3316
YS_PREFIX = "YS"
IN_SPLIT_PREFIX = "in_split"
IN_SPLIT_EXTENSION = ".png"
TARGET_LETTER_CSV = "target_letter.csv"
INPUT_IMAGE_NAME = "in.png"
OUTPUT_IMAGE_NAME = "in.png"
DATA_FOLDER_PREFIX = "data"

def read_csv_file(file_path):
    """
    Read a CSV file and return its rows as a list of lists.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            return list(reader)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error reading CSV {file_path}: {e}")
        return None

def check_target_letter_conditions(rows):
    """
    Check if fourth column does not contain any of '(a)', '(b)', '(c)', '(C)', '(d)', '(e)', '(f)'.
    Return True if condition is met, else False.
    """
    if not rows:
        return False
    fourth_column = [row[3].strip() for row in rows[1:] if len(row) > 3]
    valid_labels = ["(a)", "(b)", "(c)", "(C)", "(d)", "(e)", "(f)"]
    return not any(label in fourth_column for label in valid_labels)

def process_subfolders():
    """
    Iterate through L0001 to L3316, process YS subfolders according to conditions.
    """
    for i in range(1, NUM_SUBFOLDERS + 1):
        subfolder_name = f"{SUBFOLDER_PREFIX}{i:04d}"
        subfolder_path = os.path.join(BASE_PATH, subfolder_name)

        if not os.path.exists(subfolder_path):
            continue

        # Check for YS subfolders
        for item in os.listdir(subfolder_path):
            if item.startswith(YS_PREFIX) and os.path.isdir(os.path.join(subfolder_path, item)):
                ys_subfolder_path = os.path.join(subfolder_path, item)

                # Check for in_split*.png files
                in_split_files = [
                    f for f in os.listdir(ys_subfolder_path)
                    if f.startswith(IN_SPLIT_PREFIX) and f.endswith(IN_SPLIT_EXTENSION)
                ]

                if in_split_files:
                    # Process each in_split*.png file
                    for in_split_file in in_split_files:
                        # Extract the number from the filename (e.g., 1 from in_split1.png)
                        match = re.search(r'in_split(\d+)\.png', in_split_file)
                        if not match:
                            continue
                        number = match.group(1)
                        data_folder = f"{DATA_FOLDER_PREFIX}{number}"
                        data_folder_path = os.path.join(ys_subfolder_path, data_folder)
                        src_path = os.path.join(ys_subfolder_path, in_split_file)
                        dst_path = os.path.join(data_folder_path, OUTPUT_IMAGE_NAME)

                        try:
                            os.makedirs(data_folder_path, exist_ok=True)
                            shutil.copy2(src_path, dst_path)
                        except Exception as e:
                            print(f"Error copying {src_path} to {dst_path}: {e}")
                else:
                    # No in_split*.png files, check target_letter.csv
                    target_letter_path = os.path.join(ys_subfolder_path, TARGET_LETTER_CSV)
                    if not os.path.exists(target_letter_path):
                        continue

                    letter_rows = read_csv_file(target_letter_path)
                    if not check_target_letter_conditions(letter_rows):
                        continue

                    # Copy in.png to data folder
                    src_path = os.path.join(ys_subfolder_path, INPUT_IMAGE_NAME)
                    if not os.path.exists(src_path):
                        continue

                    data_folder_path = os.path.join(ys_subfolder_path, DATA_FOLDER_PREFIX)
                    dst_path = os.path.join(data_folder_path, OUTPUT_IMAGE_NAME)

                    try:
                        os.makedirs(data_folder_path, exist_ok=True)
                        shutil.copy2(src_path, dst_path)
                    except Exception as e:
                        print(f"Error copying {src_path} to {dst_path}: {e}")

def main():
    """
    Main function to verify base path and start processing.
    """
    if not os.path.exists(BASE_PATH):
        print(f"Error: Base path '{BASE_PATH}' does not exist.")
        return

    process_subfolders()

if __name__ == "__main__":
    main()