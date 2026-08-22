import os
import csv
import re
import glob
from collections import defaultdict
import shutil

def merge_figure_csvs(base_dir):
    """
    Traverses the directory structure, finds all 'figure_out*_data*.csv' or 'figure_out*_data.csv'
    files, groups them by the primary figure number, and merges them into a single
    'figure_out{number}.csv' file in each 'L' folder.
    """
    # Loop through folders L0001 to L3316
    for i in range(1, 3317):
        l_folder_name = f"L{i:04d}"
        l_folder_path = os.path.join(base_dir, l_folder_name)

        if not os.path.isdir(l_folder_path):
            continue

        # --- Find all possible data files for figures ---
        # This pattern matches figure_out1_data1.csv, figure_out1_data.csv etc.
        figure_csv_pattern = os.path.join(l_folder_path, "figure_out*_data*.csv")
        figure_csv_files = glob.glob(figure_csv_pattern)

        if not figure_csv_files:
            continue  # Skip this L folder if no data files are found

        print(f"\nProcessing folder: {l_folder_path}")

        # --- Group files by the main figure number ---
        # The key is the number after 'out', the value is a list of file paths
        grouped_files = defaultdict(list)
        for f_path in figure_csv_files:
            # This regex captures the number immediately after 'figure_out'
            match = re.search(r'figure_out(\d+)_data', os.path.basename(f_path))
            if match:
                fig_num = int(match.group(1))
                grouped_files[fig_num].append(f_path)
        
        if not grouped_files:
            continue

        # --- Process each group of files ---
        for fig_num, file_list in grouped_files.items():
            output_filename = f"figure_out{fig_num}.csv"
            output_path = os.path.join(l_folder_path, output_filename)
            
            print(f"  - Merging {len(file_list)} file(s) for figure {fig_num} -> {output_filename}")

            # Case 1: Only one file exists for this figure number.
            if len(file_list) == 1:
                try:
                    shutil.copy2(file_list[0], output_path)
                    print(f"    - Copied single file to {output_filename}")
                except Exception as e:
                    print(f"    - FAILED to copy {file_list[0]}. Error: {e}")
                continue # Move to the next figure group

            # Case 2: Multiple files need to be merged.
            try:
                header = None
                all_data_rows = []

                # Read all data rows from all files in the group
                for file_path in sorted(file_list): # Sort to maintain a consistent order
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            reader = csv.reader(infile)
                            # Read the header from the first file we successfully open
                            if header is None:
                                header = next(reader)
                            else:
                                next(reader) # Skip headers of subsequent files
                            
                            # Add all data rows from this file
                            all_data_rows.extend(list(reader))
                    except Exception as e:
                        print(f"    - Warning: Could not read {file_path}. Error: {e}")

                # Write the merged data to the new file
                if header is not None:
                    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
                        writer = csv.writer(outfile)
                        writer.writerow(header)
                        writer.writerows(all_data_rows)
                    print(f"    - Successfully merged {len(all_data_rows)} data rows into {output_filename}")
                else:
                    print(f"    - FAILED: Could not read a header from any of the files for figure {fig_num}.")

            except Exception as e:
                print(f"    - FAILED to merge files for figure {fig_num}. Error: {e}")


def main():
    """
    Main function to run the batch merging process.
    """
    base_directory = r".\HEA_finetune\Yield_Strength"
    merge_figure_csvs(base_directory)
    print("\nBatch processing complete.")

if __name__ == '__main__':
    main()
