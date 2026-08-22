import os
import csv
import glob
import sys

def main():
    """
    Main function to find all 'Fig*_out_2.csv' files in L* subfolders
    and combine them into a single 'Fig_out.csv' file.
    """
    base_path = os.getcwd()
    output_filename = os.path.join(base_path, 'Fig_out.csv')
    
    print(f"Starting to combine files into: {output_filename}")
    
    is_header_written = False
    total_files_processed = 0
    total_rows_written = 0

    try:
        # Open the single output file in write mode
        with open(output_filename, mode='w', encoding='utf-8', newline='') as outfile:
            writer = csv.writer(outfile)
            
            # 1. Loop through folders L0001 to L2594
            for i in range(1, 3356):
                folder_name = f"L{i:04d}"
                folder_path = os.path.join(base_path, folder_name)
                
                if not os.path.isdir(folder_path):
                    continue

                # 2. Find all target files (Fig*_out_2.csv)
                search_pattern = os.path.join(folder_path, 'Fig*_out_2.csv')
                target_files = glob.glob(search_pattern)

                if not target_files:
                    continue
                
                print(f"\nChecking folder: {folder_name}")

                for filepath in target_files:
                    print(f"  Processing file: {os.path.basename(filepath)}")
                    try:
                        # Open each target file in read mode
                        with open(filepath, mode='r', encoding='utf-8', newline='') as infile:
                            reader = csv.reader(infile)
                            
                            try:
                                # Read the header row from this file
                                header = next(reader)
                            except StopIteration:
                                # File is empty, skip it
                                print(f"    Warning: File is empty, skipping. {filepath}")
                                continue
                                
                            # 3. Write header ONLY if it's the first file
                            if not is_header_written:
                                writer.writerow(header)
                                is_header_written = True
                            
                            # Write all data rows from this file
                            rows_in_this_file = 0
                            for row in reader:
                                writer.writerow(row)
                                rows_in_this_file += 1
                                
                            total_rows_written += rows_in_this_file
                            total_files_processed += 1

                    except Exception as e:
                        print(f"  Error reading {filepath}: {e}")

    except Exception as e:
        print(f"CRITICAL ERROR writing to output file {output_filename}: {e}")
        return

    print(f"\nBatch combination complete.")
    print(f"  Total files processed: {total_files_processed}")
    print(f"  Total data rows written: {total_rows_written}")
    print(f"  Combined file created: {output_filename}")

if __name__ == "__main__":
    main()