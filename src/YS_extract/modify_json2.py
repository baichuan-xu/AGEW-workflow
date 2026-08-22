import os
import glob
import re
import json

def clean_description_json(base_path):
    """
    Iterates through folders, finds description.json, and keeps only the JSON content between ```json and ```.
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

                # Read and process description.json
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Extract JSON between ```json and ```
                    match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                    if not match:
                        print(f"No JSON block found in {json_file}, skipping")
                        continue

                    json_content = match.group(1).strip()

                    # Validate JSON
                    try:
                        json_data = json.loads(json_content)
                    except json.JSONDecodeError as e:
                        print(f"Invalid JSON in {json_file}: {e}, skipping")
                        continue

                    # Write cleaned JSON back to description.json
                    try:
                        with open(json_file, 'w', encoding='utf-8') as f:
                            json.dump(json_data, f, ensure_ascii=False, indent=2)
                        print(f"Successfully cleaned and saved {json_file}")
                    except Exception as e:
                        print(f"Error writing to {json_file}: {e}")

                except Exception as e:
                    print(f"Error reading {json_file}: {e}, skipping")

# Usage
if __name__ == "__main__":
    base_path = r".\HEA_finetune\Yield_Strength"
    clean_description_json(base_path)