import os
import glob

def check_multi_images(base_path):
    """
    Iterates through folders, checks check_out.txt for 'multi images' content,
    and prints the folder path if found.
    """
    # Iterate through folders L0001 to L3316
    for i in range(1, 3317):
        folder_name = f"L{i:04d}"
        folder_path = os.path.join(base_path, folder_name)

        if not os.path.isdir(folder_path):
            print(f"Folder {folder_path} does not exist, skipping")
            continue

        # Check for subfolders starting with YS
        ys_folders = glob.glob(os.path.join(folder_path, "YS*"))
        if not ys_folders:
            continue  # Silently skip folders without YS* subfolders

        for ys_folder in ys_folders:
            if not os.path.isdir(ys_folder):
                continue  # Silently skip non-directory items

            # Check for subfolders starting with data
            data_folders = glob.glob(os.path.join(ys_folder, "data*"))
            if not data_folders:
                continue  # Silently skip folders without data* subfolders

            for data_folder in data_folders:
                if not os.path.isdir(data_folder):
                    continue  # Silently skip non-directory items

                # Look for check_out.txt
                txt_file = os.path.join(data_folder, "check_out.txt")
                if not os.path.isfile(txt_file):
                    continue  # Silently skip folders without check_out.txt

                # Read the content of check_out.txt
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    # Check if content is exactly "multi images"
                    if content == "multi images":
                        print(data_folder)
                except Exception as e:
                    print(f"Error reading {txt_file}: {e}")

# Usage
if __name__ == "__main__":
    base_path = r".\HEA_finetune\Yield_Strength"
    check_multi_images(base_path)