import os
import csv

BASE_PATH = r".\HEA_finetune\Yield_Strength"
SUBFOLDER_PREFIX = "L"
NUM_SUBFOLDERS = 3316
YS_PREFIX = "YS"
TARGET_LETTER_CSV = "target_letter.csv"
IN_SPLIT_PREFIX = "in_split"
IN_SPLIT_EXTENSION = ".png"

def read_csv_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            return list(reader)
    except:
        return None

def check_target_letter_conditions(rows):
    if not rows:
        return False
    fourth_column = [row[3].strip() for row in rows[1:] if len(row) > 3]
    valid_labels = ["(a)", "(b)", "(c)", "(C)", "(d)", "(e)", "(f)"]
    return any(label in fourth_column for label in valid_labels)

def process_subfolders():
    for i in range(1, NUM_SUBFOLDERS + 1):
        subfolder_name = f"{SUBFOLDER_PREFIX}{i:04d}"
        subfolder_path = os.path.join(BASE_PATH, subfolder_name)
        if not os.path.exists(subfolder_path):
            continue
        for item in os.listdir(subfolder_path):
            if item.startswith(YS_PREFIX) and os.path.isdir(os.path.join(subfolder_path, item)):
                ys_subfolder_path = os.path.join(subfolder_path, item)
                target_letter_path = os.path.join(ys_subfolder_path, TARGET_LETTER_CSV)
                if not os.path.exists(target_letter_path):
                    continue
                letter_rows = read_csv_file(target_letter_path)
                if not check_target_letter_conditions(letter_rows):
                    continue
                has_in_split = any(
                    f.startswith(IN_SPLIT_PREFIX) and f.endswith(IN_SPLIT_EXTENSION)
                    for f in os.listdir(ys_subfolder_path)
                )
                if not has_in_split:
                    print(ys_subfolder_path)

if __name__ == "__main__":
    process_subfolders()