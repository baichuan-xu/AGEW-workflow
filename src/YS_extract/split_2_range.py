import os
import csv

# --- Configuration ---
BASE_PATH = r".\HEA_finetune\Yield_Strength"
SUBFOLDER_PREFIX = "L"
NUM_SUBFOLDERS = 3316
YS_PREFIX = "YS"
TARGET_LETTER_CSV = "target_letter.csv"
GRID_CSV = "grid.csv"
TARGET_TEXT_CSV = "target_text.csv"
OUTPUT_CSV_PREFIX = "split_range"

def read_csv_file(file_path):
    """
    Read a CSV file and return its rows as a list of lists.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            return list(reader)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV {file_path}: {e}")
        return None

def write_split_range_csv(file_path, x_min, x_max, y_min, y_max):
    """
    Write x_min, x_max, y_min, y_max to a CSV file with headers.
    """
    try:
        with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["x_min", "x_max", "y_min", "y_max"])
            writer.writerow([x_min, x_max, y_min, y_max])
        print(f"Created {file_path}")
        return True
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return False

def check_target_letter_conditions(rows):
    """
    Check if fourth column contains '(b)' but none of '(c)', '(d)', '(e)', '(f)', '(C)', '(D)', '(E)', '(F)'.
    Return X_b, Y_b if conditions are met, else None.
    """
    fourth_column = [row[3].strip() for row in rows[1:] if len(row) > 3]
    if "(b)" not in fourth_column:
        return None
    if any(label in fourth_column for label in ["(c)", "(d)", "(e)", "(f)", "(C)", "(D)", "(E)", "(F)"]):
        return None
    
    # Find X_b, Y_b for the row with '(b)'
    for row in rows[1:]:
        if len(row) > 3 and row[3].strip() == "(b)":
            try:
                x_b = float(row[1])  # Second column is X_b
                y_b = float(row[2])  # Third column is Y_b
                return x_b, y_b
            except (IndexError, ValueError):
                print(f"Error: Invalid X_b or Y_b value in row {row}")
                return None
    return None

def calculate_x_mid(rows, x_b):
    """
    Calculate X_mid for X_b > 500 based on presence of '(a)' and its X value.
    """
    has_a = False
    x_a = None
    for row in rows[1:]:
        if len(row) > 3 and row[3].strip() == "(a)":
            has_a = True
            try:
                x_a = float(row[1])  # Second column is X for '(a)'
            except (IndexError, ValueError):
                print(f"Error: Invalid X value for '(a)' in row {row}")
                return None
            break
    
    if not has_a:
        return x_b - 50  # Case (1): No '(a)'
    elif x_a < 200:
        return x_b - 50  # Case (2): '(a)' with X < 200
    else:
        return x_b - 300  # Case (3): '(a)' with X > 200

def get_grid_dimensions(grid_path):
    """
    Extract X_min, X_max, Y_min, Y_max from grid.csv.
    """
    rows = read_csv_file(grid_path)
    if not rows or len(rows) < 2 or len(rows[1]) < 4:
        print(f"Error: Invalid grid.csv format at {grid_path}")
        return None
    try:
        return {
            "X_min": float(rows[1][0]),
            "X_max": float(rows[1][1]),
            "Y_min": float(rows[1][2]),
            "Y_max": float(rows[1][3])
        }
    except (IndexError, ValueError):
        print(f"Error: Invalid numeric values in {grid_path}")
        return None

def process_subfolders():
    """
    Iterate through L0001 to L3316, process YS subfolders meeting conditions.
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
                target_letter_path = os.path.join(ys_subfolder_path, TARGET_LETTER_CSV)
                grid_path = os.path.join(ys_subfolder_path, GRID_CSV)
                target_text_path = os.path.join(ys_subfolder_path, TARGET_TEXT_CSV)

                # Check if target_letter.csv exists
                if not os.path.exists(target_letter_path):
                    print(f"No {TARGET_LETTER_CSV} in {ys_subfolder_path}. Skipping...")
                    continue

                # Read and check target_letter.csv
                letter_rows = read_csv_file(target_letter_path)
                if not letter_rows:
                    continue

                letter_data = check_target_letter_conditions(letter_rows)
                if letter_data is None:
                    print(f"Conditions not met in {target_letter_path}. Skipping...")
                    continue

                x_b, y_b = letter_data

                # Read grid.csv
                grid_dims = get_grid_dimensions(grid_path)
                if not grid_dims:
                    print(f"Skipping {ys_subfolder_path} due to invalid grid.csv")
                    continue

                # Read target_text.csv and process all rows except header
                text_rows = read_csv_file(target_text_path)
                if not text_rows or len(text_rows) < 2:
                    print(f"No data rows in {target_text_path}. Skipping...")
                    continue

                # Process all data rows (skip header)
                for idx, row in enumerate(text_rows[1:], 1):
                    if len(row) < 3:
                        print(f"Invalid row format in {target_text_path}: {row}")
                        continue
                    try:
                        x = float(row[1])  # Second column is X
                        y = float(row[2])  # Third column is Y
                    except (IndexError, ValueError):
                        print(f"Invalid X or Y value in row {row} of {target_text_path}")
                        continue

                    # Determine x_min, x_max, y_min, y_max
                    if x_b < 500:
                        y_mid = y_b - 50
                        if y < y_mid:
                            x_min = grid_dims["X_min"]
                            x_max = grid_dims["X_max"]
                            y_min = grid_dims["Y_min"]
                            y_max = y_mid
                        else:
                            x_min = grid_dims["X_min"]
                            x_max = grid_dims["X_max"]
                            y_min = y_mid
                            y_max = grid_dims["Y_max"]
                    else:
                        x_mid = calculate_x_mid(letter_rows, x_b)
                        if x_mid is None:
                            print(f"Failed to calculate X_mid in {ys_subfolder_path}. Skipping row...")
                            continue
                        if x < x_mid:
                            x_min = grid_dims["X_min"]
                            x_max = x_mid
                            y_min = grid_dims["Y_min"]
                            y_max = grid_dims["Y_max"]
                        else:
                            x_min = x_mid
                            x_max = grid_dims["X_max"]
                            y_min = grid_dims["Y_min"]
                            y_max = grid_dims["Y_max"]

                    # Write to split_rangeN.csv
                    output_csv = f"{OUTPUT_CSV_PREFIX}{idx}.csv"
                    output_path = os.path.join(ys_subfolder_path, output_csv)
                    write_split_range_csv(output_path, x_min, x_max, y_min, y_max)

def main():
    """
    Main function to verify base path and start processing.
    """
    if not os.path.exists(BASE_PATH):
        print(f"Error: Base path '{BASE_PATH}' does not exist.")
        return

    print(f"Processing subfolders in {BASE_PATH}...")
    process_subfolders()
    print("\nProcessing complete.")

if __name__ == "__main__":
    main()