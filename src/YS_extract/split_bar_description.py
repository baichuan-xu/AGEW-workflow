import json
import os
import glob
import re

def split_json_file(input_file, output_dir):
    """
    Splits a description.json file into multiple JSON files, one per series_detail,
    saving them in the specified output directory.
    """
    try:
        # Read the input JSON file with UTF-8 encoding
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get the series_details list
        series_details = data.get('series_details', [])
        if not series_details:
            print(f"No series_details found in {input_file}, skipping")
            return
        
        # Process each series detail
        for item in series_details:
            # Create a new dictionary with graph_type and single series detail
            new_data = {
                'graph_type': data['graph_type'],
                'series_details': [item]
            }
            
            # Generate output filename based on label
            label = item['label']
            # Replace spaces, slashes, and other invalid filename characters
            safe_label = re.sub(r'[^\w\-]', '_', label)
            output_filename = f'description_{safe_label}.json'
            output_path = os.path.join(output_dir, output_filename)
            
            # Write to new JSON file with UTF-8 encoding
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, indent=4, ensure_ascii=False)
            
            print(f'Created file: {output_path}')

    except FileNotFoundError:
        print(f'Error: {input_file} not found')
    except json.JSONDecodeError:
        print(f'Error: {input_file} contains invalid JSON')
    except Exception as e:
        print(f'Error processing {input_file}: {str(e)}')

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
                json_file = os.path.join(data_folder, "description.json")
                if not os.path.isfile(json_file):
                    print(f"description.json not found in {data_folder}, skipping")
                    continue

                print(f"Processing {json_file}...")
                split_json_file(json_file, data_folder)

# Usage
if __name__ == "__main__":
    base_path = r".\HEA_finetune\Yield_Strength"
    process_folders(base_path)