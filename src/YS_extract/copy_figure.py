import os
import csv
import shutil
import re

def process_and_copy_files(base_dir):
    """
    Traverses the directory structure, checks for non-empty data files,
    and copies them to the parent 'L' directory with a new name.
    """
    # Loop through folders L0001 to L3316
    for i in range(1, 3317):
        l_folder_name = f"L{i:04d}"
        l_folder_path = os.path.join(base_dir, l_folder_name)

        if not os.path.isdir(l_folder_path):
            continue

        print(f"\nProcessing folder: {l_folder_path}")

        # Find subfolders starting with 'YS'
        try:
            ys_folders = [f for f in os.listdir(l_folder_path) if os.path.isdir(os.path.join(l_folder_path, f)) and f.startswith("YS")]
        except FileNotFoundError:
            print(f"  - Could not access {l_folder_path}, skipping.")
            continue

        for ys_folder in ys_folders:
            ys_folder_path = os.path.join(l_folder_path, ys_folder)

            # Find subfolders starting with 'data'
            try:
                data_folders = [f for f in os.listdir(ys_folder_path) if os.path.isdir(os.path.join(ys_folder_path, f)) and f.startswith("data")]
            except FileNotFoundError:
                print(f"  - Could not access {ys_folder_path}, skipping.")
                continue

            for data_folder in data_folders:
                data_folder_path = os.path.join(ys_folder_path, data_folder)
                
                # Path for the source file
                source_file_path = os.path.join(data_folder_path, "image_data_filter.csv")

                if not os.path.exists(source_file_path):
                    print(f"  - SKIPPING: 'image_data_filter.csv' not found in {data_folder_path}")
                    continue

                # Check if the file has more than just a header line
                try:
                    with open(source_file_path, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        # Count the number of rows. More than 1 means there's data.
                        row_count = sum(1 for row in reader)
                    
                    if row_count <= 1:
                        print(f"  - SKIPPING: 'image_data_filter.csv' in {data_folder_path} is empty or has only a header.")
                        continue
                        
                except Exception as e:
                    print(f"  - SKIPPING: Could not read {source_file_path}. Error: {e}")
                    continue

                # Construct the new filename
                try:
                    # Extract numbers from folder names using regular expressions
                    ys_num = re.findall(r'\d+', ys_folder)[0]
                    data_num = re.findall(r'\d+', data_folder)[0]
                    new_filename = f"figure_out{ys_num}_data{data_num}.csv"
                except IndexError:
                    print(f"  - SKIPPING: Could not parse numbers from folder names '{ys_folder}' and '{data_folder}'.")
                    continue
                
                # Define the destination path in the parent 'L' folder
                destination_path = os.path.join(l_folder_path, new_filename)

                # Copy and rename the file
                try:
                    shutil.copy2(source_file_path, destination_path)
                    print(f"  - SUCCESS: Copied and renamed to {destination_path}")
                except Exception as e:
                    print(f"  - FAILED: Could not copy file to {destination_path}. Error: {e}")


def main():
    """
    Main function to run the batch processing.
    """
    base_directory = r".\HEA_finetune\Yield_Strength"
    process_and_copy_files(base_directory)
    print("\nBatch processing complete.")

if __name__ == '__main__':
    main()
