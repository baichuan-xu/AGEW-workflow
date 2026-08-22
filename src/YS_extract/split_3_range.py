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
    Check if fourth column contains '(c)' or '(C)' but none of '(d)', '(e)', '(f)', '(D)', '(E)', '(F)'.
    Return True if conditions are met, else False.
    """
    fourth_column = [row[3].strip() for row in rows[1:] if len(row) > 3]
    if "(c)" not in fourth_column and "(C)" not in fourth_column:
        return False
    if any(label in fourth_column for label in ["(d)", "(e)", "(f)", "(D)", "(E)", "(F)"]):
        return False
    return True

def extract_letter_coordinates(rows):
    """
    Extract X_c, Y_c for '(c)' or '(C)', X_b, Y_b for '(b)', X_a, Y_a for '(a)'.
    Return a dictionary with available coordinates.
    """
    coords = {"X_c": None, "Y_c": None, "X_b": None, "Y_b": None, "X_a": None, "Y_a": None}
    for row in rows[1:]:
        if len(row) > 3:
            label = row[3].strip()
            try:
                x = float(row[1])
                y = float(row[2])
                if label in ["(c)", "(C)"]:
                    coords["X_c"] = x
                    coords["Y_c"] = y
                elif label == "(b)":
                    coords["X_b"] = x
                    coords["Y_b"] = y
                elif label == "(a)":
                    coords["X_a"] = x
                    coords["Y_a"] = y
            except (IndexError, ValueError):
                print(f"Error: Invalid X or Y value in row {row}")
    return coords

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

                if not check_target_letter_conditions(letter_rows):
                    print(f"Conditions not met in {target_letter_path}. Skipping...")
                    continue

                # Extract coordinates
                coords = extract_letter_coordinates(letter_rows)
                x_c, y_c = coords["X_c"], coords["Y_c"]
                x_b, y_b = coords["X_b"], coords["Y_b"]
                x_a, y_a = coords["X_a"], coords["Y_a"]

                # Read grid.csv
                grid_dims = get_grid_dimensions(grid_path)
                if not grid_dims:
                    print(f"Skipping {ys_subfolder_path} due to invalid grid.csv")
                    continue

                # Read target_text.csv and process all data rows
                text_rows = read_csv_file(target_text_path)
                if not text_rows or len(text_rows) < 2:
                    print(f"No data rows in {target_text_path}. Skipping...")
                    continue

                # Determine which condition applies
                y_values = [y for y in [y_c, y_b, y_a] if y is not None]
                x_values = [x for x in [x_c, x_b, x_a] if x is not None]
                condition = None

                if all(y < 500 for y in y_values):
                    condition = 1
                elif all(x < 500 for x in x_values):
                    condition = 2
                elif y_b is not None and y_c is not None and y_b > 500 and y_c > 500 and x_b is not None and x_c is not None and x_b < 500 and x_c > 500:
                    condition = 3
                elif y_b is not None and y_c is not None and y_b < 500 and y_c > 500:
                    condition = 4
                else:
                    print(f"No applicable condition met in {ys_subfolder_path}. Skipping...")
                    continue

                # Process each data row in target_text.csv
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

                    x_min, x_max, y_min, y_max = None, None, None, None

                    if condition == 1:
                        if x_b is None or x_c is None:
                            continue
                        if x_a is None:
                            x_mid1 = x_b - 50
                            x_mid2 = x_c - 50
                        elif x_a < 200:
                            x_mid1 = x_b - 50
                            x_mid2 = x_c - 50
                        else:
                            x_mid1 = x_b - 300
                            x_mid2 = x_c - 300
                        if x < x_mid1:
                            x_min = grid_dims["X_min"]
                            x_max = x_mid1
                            y_min = grid_dims["Y_min"]
                            y_max = grid_dims["Y_max"]
                        elif x_mid1 < x < x_mid2:
                            x_min = x_mid1
                            x_max = x_mid2
                            y_min = grid_dims["Y_min"]
                            y_max = grid_dims["Y_max"]
                        else:  # x > x_mid2
                            x_min = x_mid2
                            x_max = grid_dims["X_max"]
                            y_min = grid_dims["Y_min"]
                            y_max = grid_dims["Y_max"]

                    elif condition == 2:
                        if y_b is None or y_c is None:
                            continue
                        y_mid1 = y_b - 50
                        y_mid2 = y_c - 50
                        if y < y_mid1:
                            x_min = grid_dims["X_min"]
                            x_max = grid_dims["X_max"]
                            y_min = grid_dims["Y_min"]
                            y_max = y_mid1
                        elif y_mid1 < y < y_mid2:
                            x_min = grid_dims["X_min"]
                            x_max = grid_dims["X_max"]
                            y_min = y_mid1
                            y_max = y_mid2
                        else:  # y > y_mid2
                            x_min = grid_dims["X_min"]
                            x_max = grid_dims["X_max"]
                            y_min = y_mid2
                            y_max = grid_dims["Y_max"]

                    elif condition == 3:
                        if x_b is None or y_c is None:
                            continue
                        if x_b < 200:
                            x_mid = x_c - 50
                            y_mid = y_c - 50
                        else:
                            x_mid = x_c - 300
                            y_mid = y_c - 50
                        if y < y_mid:
                            x_min = grid_dims["X_min"]
                            x_max = grid_dims["X_max"]
                            y_min = grid_dims["Y_min"]
                            y_max = y_mid
                        elif y > y_mid and x < x_mid:
                            x_min = grid_dims["X_min"]
                            x_max = x_mid
                            y_min = y_mid
                            y_max = grid_dims["Y_max"]
                        else:  # y > y_mid and x > x_mid
                            x_min = x_mid
                            x_max = grid_dims["X_max"]
                            y_min = y_mid
                            y_max = grid_dims["Y_max"]

                    elif condition == 4:
                        if x_b is None or y_c is None:
                            continue
                        if x_a is None:
                            x_mid = x_b - 50
                            y_mid = y_c - 50
                        elif x_a < 200:
                            x_mid = x_b - 50
                            y_mid = y_c - 50
                        else:
                            x_mid = x_b - 300
                            y_mid = y_c - 50
                        if y > y_mid:
                            x_min = grid_dims["X_min"]
                            x_max = grid_dims["X_max"]
                            y_min = y_mid
                            y_max = grid_dims["Y_max"]
                        elif y < y_mid and x < x_mid:
                            x_min = grid_dims["X_min"]
                            x_max = x_mid
                            y_min = grid_dims["Y_min"]
                            y_max = y_mid
                        else:  # y < y_mid and x > x_mid
                            x_min = x_mid
                            x_max = grid_dims["X_max"]
                            y_min = grid_dims["Y_min"]
                            y_max = y_mid

                    if x_min is not None:
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