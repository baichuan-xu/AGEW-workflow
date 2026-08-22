import os
import csv

# Base directory containing L0001 to L3316 folders
base_dir = r".\HEA_finetune\Yield_Strength"

# Function to check for negative values in the 5th column of image_data_final.csv
def check_negative_values(file_path, folder_path):
    try:
        with open(file_path, "r", encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file)
            # Skip the header row
            next(csv_reader, None)
            # Check the 5th column (index 4) for negative values
            for row in csv_reader:
                if len(row) >= 5:  # Ensure row has at least 5 columns
                    try:
                        value = float(row[4])  # Convert to float
                        if value < 0:
                            print(folder_path)
                            return  # Exit after finding the first negative value
                    except ValueError:
                        continue  # Skip non-numeric values
    except FileNotFoundError:
        pass  # File not found is handled by the existence check
    except Exception as e:
        pass  # Silently skip errors to avoid output

# Function to process a single folder
def process_folder(folder_path, folder_name):
    # Check for YS* subfolders
    ys_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f)) and f.startswith("YS")]
    
    for ys_folder in ys_folders:
        ys_folder_path = os.path.join(folder_path, ys_folder)
        
        # Check for subfolders starting with "data"
        data_folders = [f for f in os.listdir(ys_folder_path) if os.path.isdir(os.path.join(ys_folder_path, f)) and f.startswith("data")]
        
        for data_folder in data_folders:
            data_folder_path = os.path.join(ys_folder_path, data_folder)
            
            # Check for image_data_final.csv in data* folder
            csv_file_path = os.path.join(data_folder_path, "image_data_final.csv")
            if not os.path.exists(csv_file_path):
                continue  # Skip silently if file doesn't exist
            
            # Check for negative values
            check_negative_values(csv_file_path, data_folder_path)

# Main loop to process L0001 to L3316
for i in range(1, 3317):
    folder_name = f"L{i:04d}"
    folder_path = os.path.join(base_dir, folder_name)
    if os.path.exists(folder_path):
        process_folder(folder_path, folder_name)