import os
import glob
import subprocess
import sys

def run_extract_py(subfolder_path):
    """
    Runs extract.py in the specified subfolder, capturing output and errors.
    """
    extract_py_path = os.path.join(subfolder_path, "extract.py")
    if not os.path.isfile(extract_py_path):
        print(f"extract.py not found in {subfolder_path}, skipping")
        return

    print(f"Running {extract_py_path}...")
    try:
        # Run extract.py in its directory
        result = subprocess.run(
            [sys.executable, "extract.py"],
            cwd=subfolder_path,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        # Log output
        if result.stdout:
            print(f"Output from {extract_py_path}:\n{result.stdout.strip()}")
        if result.stderr:
            print(f"Errors from {extract_py_path}:\n{result.stderr.strip()}")
        if result.returncode == 0:
            print(f"Successfully executed {extract_py_path}")
        else:
            print(f"Execution of {extract_py_path} failed with return code {result.returncode}")
    except Exception as e:
        print(f"Error running {extract_py_path}: {e}")

def process_folders(base_path):
    """
    Iterates through folders, checks for description.json, and runs extract.py in subfolders.
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
                    run_extract_py(subfolder)

# Usage
if __name__ == "__main__":
    base_path = r".\HEA_finetune\Yield_Strength"
    process_folders(base_path)