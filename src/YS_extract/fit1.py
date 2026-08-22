import os
import csv
import random
import numpy as np

# Base directory containing L0001 to L3316 folders
base_dir = r".\HEA_finetune\Yield_Strength"

# Function to check first line of check_data.txt
def check_data_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as text_file:
            first_line = text_file.readline().strip()
            return first_line == "No Numerical annotation"
    except FileNotFoundError:
        print(f"Error: check_data.txt not found at {file_path}")
        return False
    except Exception as e:
        print(f"Error reading check_data.txt {file_path}: {e}")
        return False

# Function to perform linear fitting (y = ax + b) given two points
def linear_fit(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    if x1 == x2:
        return None, None  # Avoid division by zero
    a = (y2 - y1) / (x2 - x1)  # Slope
    b = y1 - a * x1  # Intercept
    return a, b

# Function to process CSV file and perform linear fitting
def process_csv_file(file_path, x_col_idx, y_col_idx):
    try:
        with open(file_path, "r", encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file)
            header = next(csv_reader)  # Skip header
            data_rows = [row for row in csv_reader if row and len(row) > max(x_col_idx, y_col_idx)]
            if len(data_rows) < 2:
                print(f"Error: {file_path} has fewer than 2 valid data rows")
                return None, None
            # Convert to float and select x, y values
            points = [(float(row[x_col_idx]), float(row[y_col_idx])) for row in data_rows]
            # Randomly select two rows
            selected_points = random.sample(points, 2)
            return linear_fit(selected_points[0], selected_points[1])
    except FileNotFoundError:
        print(f"Error: {file_path} not found")
        return None, None
    except (ValueError, IndexError) as e:
        print(f"Error reading or parsing {file_path}: {e}")
        return None, None

# Function to process a single folder
def process_folder(folder_path, folder_name):
    print(f"\nProcessing folder: {folder_name}")
    
    # Check for YS* subfolders
    ys_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f)) and f.startswith("YS")]
    
    for ys_folder in ys_folders:
        ys_folder_path = os.path.join(folder_path, ys_folder)
        print(f"Found YS folder: {ys_folder}")
        
        # Check for subfolders starting with "data"
        data_folders = [f for f in os.listdir(ys_folder_path) if os.path.isdir(os.path.join(ys_folder_path, f)) and f.startswith("data")]
        if not data_folders:
            print(f"No folder starting with 'data' found in {ys_folder}. Skipping.")
            continue
        
        for data_folder in data_folders:
            data_folder_path = os.path.join(ys_folder_path, data_folder)
            print(f"Found data folder: {data_folder}")
            
            # Check for check_data.txt and its first line
            check_data_path = os.path.join(data_folder_path, "check_data.txt")
            if not check_data_file(check_data_path):
                print(f"check_data.txt in {data_folder_path} does not start with 'No Numerical annotation' or is missing. Skipping.")
                continue
            
            # Check for y_group.csv
            y_group_path = os.path.join(data_folder_path, "y_group.csv")
            if not os.path.exists(y_group_path):
                print(f"y_group.csv not found in {data_folder_path}. Skipping.")
                continue
            
            # Process y_group.csv (x: 3rd column, y: 1st column)
            a1, b1 = process_csv_file(y_group_path, x_col_idx=2, y_col_idx=0)
            if a1 is None and b1 is None:
                print(f"Skipping {data_folder_path} due to errors in y_group.csv processing.")
                continue
            
            # Process x_group.csv (x: 2nd column, y: 1st column)
            x_group_path = os.path.join(data_folder_path, "x_group.csv")
            if os.path.exists(x_group_path):
                a2, b2 = process_csv_file(x_group_path, x_col_idx=1, y_col_idx=0)
            else:
                print(f"x_group.csv not found in {data_folder_path}. Setting a2 and b2 to 'N/A'.")
                a2, b2 = "N/A", "N/A"
            
            # Create axis_fit.csv
            output_path = os.path.join(data_folder_path, "axis_fit.csv")
            try:
                with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(["Label", "Value"])
                    writer.writerow(["a1", a1 if a1 is not None else "N/A"])
                    writer.writerow(["b1", b1 if b1 is not None else "N/A"])
                    writer.writerow(["a2", a2])
                    writer.writerow(["b2", b2])
                print(f"Successfully created {output_path}")
            except Exception as e:
                print(f"Error writing to {output_path}: {e}")

# Main loop to process L0001 to L3316
for i in range(1, 3317):
    folder_name = f"L{i:04d}"
    folder_path = os.path.join(base_dir, folder_name)
    if os.path.exists(folder_path):
        process_folder(folder_path, folder_name)
    else:
        print(f"Folder {folder_name} does not exist, skipping")