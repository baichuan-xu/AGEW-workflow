import os
import glob
import re

def clean_extract_py(subfolder_path):
    """
    Reads extract.py in the subfolder, extracts Python code between ```python and ```,
    and overwrites the file with the cleaned code.
    """
    extract_py_path = os.path.join(subfolder_path, "extract.py")
    if not os.path.isfile(extract_py_path):
        print(f"extract.py not found in {subfolder_path}, skipping")
        return

    try:
        # Read extract.py
        with open(extract_py_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract Python code between ```python and ```
        match = re.search(r'```python\s*(.*?)\s*```', content, re.DOTALL)
        if not match:
            print(f"No Python code block found in {extract_py_path}, skipping")
            return

        python_code = match.group(1).strip()

        # Write cleaned Python code back to extract.py
        try:
            with open(extract_py_path, 'w', encoding='utf-8') as f:
                f.write(python_code)
            print(f"Successfully cleaned and saved {extract_py_path}")
        except Exception as e:
            print(f"Error writing to {extract_py_path}: {e}")

    except Exception as e:
        print(f"Error reading {extract_py_path}: {e}")

def process_folders(base_path):
    """
    Iterates through folders, checks for description.json, and processes extract.py in subfolders.
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
                desc_json_path = os.path.join(data_folder, "description.json")
                if not os.path.isfile(desc_json_path):
                    print(f"description.json not found in {data_folder}, skipping")
                    continue

                # Find all subfolders in the data* folder
                subfolders = [f for f in glob.glob(os.path.join(data_folder, "*")) if os.path.isdir(f)]
                for subfolder in subfolders:
                    print(f"Processing subfolder: {subfolder}")
                    clean_extract_py(subfolder)

# Usage
if __name__ == "__main__":
    base_path = r".\HEA_finetune\Yield_Strength"
    process_folders(base_path)