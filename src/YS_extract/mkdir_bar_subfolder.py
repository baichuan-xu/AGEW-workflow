import os
import glob
import re
import shutil

def process_json_files(data_folder):
    """
    Processes description_*.json files in the data folder, creates subfolders based on identifiers,
    copies the JSON files as description.json, and copies in.png to each subfolder.
    """
    # Pattern to match files starting with 'description_' and ending with '.json'
    pattern = os.path.join(data_folder, 'description_*.json')
    json_files = glob.glob(pattern)

    if not json_files:
        print(f"No files found matching 'description_*.json' in {data_folder}.")
        return

    # Check for in.png
    in_png_path = os.path.join(data_folder, 'in.png')
    in_png_exists = os.path.isfile(in_png_path)

    # Regular expression to extract the identifier (e.g., 'FeCrNi' from 'description_FeCrNi.json')
    filename_pattern = re.compile(r'description_(.+)\.json$')

    for file_path in json_files:
        try:
            # Get the filename from the full path
            filename = os.path.basename(file_path)
            
            # Extract the identifier using regex
            match = filename_pattern.match(filename)
            if not match:
                print(f"Warning: File '{filename}' in {data_folder} does not match expected pattern. Skipping.")
                continue
            
            identifier = match.group(1)  # e.g., 'FeCrNi'
            
            # Create subfolder path
            subfolder_path = os.path.join(data_folder, identifier)
            
            # Create subfolder if it doesn't exist
            os.makedirs(subfolder_path, exist_ok=True)
            
            # Destination path for copying the JSON file, renamed to 'description.json'
            dest_json_path = os.path.join(subfolder_path, 'description.json')
            
            # Copy the JSON file
            shutil.copy2(file_path, dest_json_path)
            print(f"Successfully copied '{filename}' to subfolder '{identifier}' as 'description.json' in {data_folder}")
            
            # Copy in.png if it exists
            if in_png_exists:
                dest_png_path = os.path.join(subfolder_path, 'in.png')
                shutil.copy2(in_png_path, dest_png_path)
                print(f"Successfully copied 'in.png' to subfolder '{identifier}' in {data_folder}")
            else:
                print(f"Warning: 'in.png' not found in {data_folder}, skipping copy to '{identifier}'")
                
        except OSError as e:
            print(f"Error processing '{filename}' in {data_folder}: {e}")
        except Exception as e:
            print(f"Unexpected error processing '{filename}' in {data_folder}: {e}")

def process_folders(base_path):
    """
    Iterates through folders and processes description.json files.
    """
    for i in range(1, 3317):
        folder_name = f"L{i:04d}"
        folder_path = os.path.join(base_path, folder_name)

        if not os.path.isdir(folder_path):
            print(f"Folder {folder_path} does not exist, skipping")
            continue

        print(f"Checking folder: {folder_path}")

        ys_folders = glob.glob(os.path.join(folder_path, "YS*"))
        if not ys_folders:
            print(f"No YS* folders found in {folder_path}, skipping")
            continue

        for ys_folder in ys_folders:
            if not os.path.isdir(ys_folder):
                print(f"{ys_folder} is not a directory, skipping")
                continue

            print(f"Found YS folder: {ys_folder}")

            data_folders = glob.glob(os.path.join(ys_folder, "data*"))
            if not data_folders:
                print(f"No data* folders found in {ys_folder}, skipping")
                continue

            for data_folder in data_folders:
                if not os.path.isdir(data_folder):
                    print(f"{data_folder} is not a directory, skipping")
                    continue

                print(f"Found data folder: {data_folder}")

                # Check for description.json
                desc_json_file = os.path.join(data_folder, "description.json")
                if not os.path.isfile(desc_json_file):
                    print(f"description.json not found in {data_folder}, skipping")
                    continue

                print(f"Processing files in {data_folder}...")
                process_json_files(data_folder)

# Usage
if __name__ == "__main__":
    base_path = r".\HEA_finetune\Yield_Strength"
    process_folders(base_path)
    print("Processing complete.")