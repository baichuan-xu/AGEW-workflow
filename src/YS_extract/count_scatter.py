import os
import csv

# Base directory containing L0001 to L3316 folders
base_dir = r".\HEA_finetune\Yield_Strength"

# Function to count rows in a CSV file
def count_csv_rows(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file)
            row_count = sum(1 for row in csv_reader)
        return row_count
    except FileNotFoundError:
        print(f"Error: CSV file not found at {file_path}")
        return 0
    except Exception as e:
        print(f"Error reading CSV file {file_path}: {e}")
        return 0

# Function to process a single folder
def process_folder(folder_path, folder_name):
    global total_n_true
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
            
            # Check for scatter_data.csv in data* folder
            csv_file_path = os.path.join(data_folder_path, "scatter_data.csv")
            if not os.path.exists(csv_file_path):
                print(f"scatter_data.csv not found in {data_folder_path}. Skipping.")
                continue
            
            # Count rows in scatter_data.csv
            n_original = count_csv_rows(csv_file_path)
            if n_original == 0:
                print(f"No valid rows in {csv_file_path}. Skipping.")
                continue
            
            # Calculate n_true
            n_true = n_original - 2
            if n_true < 0:
                print(f"Warning: n_original ({n_original}) in {csv_file_path} results in negative n_true. Setting n_true to 0.")
                n_true = 0
            
            print(f"File {csv_file_path}: n_original = {n_original}, n_true = {n_true}")
            total_n_true += n_true

# Main loop to process L0001 to L3316
total_n_true = 0
for i in range(1, 3317):
    folder_name = f"L{i:04d}"
    folder_path = os.path.join(base_dir, folder_name)
    if os.path.exists(folder_path):
        process_folder(folder_path, folder_name)
    else:
        print(f"Folder {folder_name} does not exist. Skipping.")

# Output the total sum of n_true
print(f"\nTotal sum of n_true across all scatter_data.csv files: {total_n_true}")