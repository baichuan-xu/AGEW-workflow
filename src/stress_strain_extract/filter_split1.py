import argparse
import csv
import os
from pathlib import Path

def process_split_check_csv(folder_path, execute=False):
    """
    Checks for 'split_check.csv' in the given folder. If the first cell
    contains 'no', the file is deleted.

    Args:
        folder_path (str): The full path to the directory to check.
    """
    csv_file_path = os.path.join(folder_path, 'split_check.csv')
    should_delete = False
    
    try:
        # First, open the file to read its content.
        with open(csv_file_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            first_row = next(reader)
            if first_row:
                first_cell_value = first_row[0].strip().lower()
                # If the content is 'no', mark the file for deletion.
                if first_cell_value == 'no':
                    should_delete = True
    except (IOError, StopIteration, IndexError):
        # Silently ignore files that are empty, unreadable, or have empty rows.
        pass

    # After the file is closed, delete it if it was marked.
    if should_delete:
        if not execute:
            print(f"Would delete: {csv_file_path} (dry-run)")
            return
        try:
            os.remove(csv_file_path)
            print(f"Deleted: {csv_file_path}")
        except OSError as e:
            print(f"Error deleting file {csv_file_path}: {e}")

def find_and_process_folders(base_directory, execute=False):
    """
    Scans through the Lxxxx/YSxxxx directory structure and processes
    the 'split_check.csv' files within.
    """
    # 1. Loop through the subfolders L0001 to L2594.
    for i in range(1, 3356):
        l_folder_name = f"L{i:04d}"
        l_folder_path = os.path.join(base_directory, l_folder_name)

        if not os.path.isdir(l_folder_path):
            continue

        # Find all subdirectories within the Lxxxx folder.
        for item_name in os.listdir(l_folder_path):
            # Check if a subfolder starts with 'YS'.
            if item_name.startswith("YS"):
                ys_folder_path = os.path.join(l_folder_path, item_name)
                if os.path.isdir(ys_folder_path):
                    
                    # 2. Find all subfolders starting with 'split' inside the YS folder.
                    split_folders = []
                    for sub_item in os.listdir(ys_folder_path):
                        if sub_item.startswith("split"):
                            split_folder_path = os.path.join(ys_folder_path, sub_item)
                            if os.path.isdir(split_folder_path):
                                split_folders.append(split_folder_path)
                    
                    # 3. If no 'split' subfolders exist, check the 'YS' folder directly.
                    if not split_folders:
                        csv_path_in_ys = os.path.join(ys_folder_path, 'split_check.csv')
                        if os.path.isfile(csv_path_in_ys):
                            process_split_check_csv(ys_folder_path, execute=execute)
                    # Otherwise, loop through each 'split' subfolder.
                    else:
                        for split_path in split_folders:
                            csv_path_in_split = os.path.join(split_path, 'split_check.csv')
                            if os.path.isfile(csv_path_in_split):
                                process_split_check_csv(split_path, execute=execute)
    
    print("\nScan and deletion process complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Legacy cleanup utility (dry-run by default).")
    parser.add_argument("--target", required=True, help="Explicit directory containing Lxxxx folders")
    parser.add_argument("--execute", action="store_true", help="Actually delete matched files")
    args = parser.parse_args()
    target = Path(args.target).expanduser().resolve()
    if not target.is_dir() or target == Path(target.anchor):
        parser.error("--target must resolve to an existing non-root directory")
    find_and_process_folders(str(target), execute=args.execute)

