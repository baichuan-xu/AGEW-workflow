import os
import csv
import sys

def search_in_target_files():
    """
    Traverses subfolders (L0001-L2954), finds ALL files matching Fig*_out_2.csv,
    and checks if user input (text) is contained within the 5th column.
    It checks every target file in every folder, never skipping files.
    """
    
    # 1. Prompt user for text input
    try:
        user_input = input("Please enter the text to search for: ").strip()
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        return

    if not user_input:
        print("Input cannot be empty.")
        return

    print(f"\nSearching for text '{user_input}' in the 5th column of ALL target files...\n")

    cwd = os.getcwd()
    total_files_checked = 0
    found_count = 0

    # 2. Walk through the directory tree
    for root, dirs, files in os.walk(cwd):
        for filename in files:
            # 3. Check file naming convention (Fig*_out_2.csv)
            if filename.startswith("Fig") and filename.endswith("_out_2.csv"):
                file_path = os.path.join(root, filename)
                total_files_checked += 1
                
                try:
                    # Open the file
                    with open(file_path, mode='r', encoding='utf-8', errors='replace') as csvfile:
                        reader = csv.reader(csvfile)
                        
                        # Skip the header row
                        next(reader, None)

                        # Browse rows within this specific file
                        for row_idx, row in enumerate(reader):
                            # Ensure the row has at least 5 columns (Index 4 is the 5th column)
                            if len(row) >= 5:
                                # Extract text from 5th column
                                target_column_text = row[4]
                                
                                # 4. Match logic: Text substring check
                                # Checks if user_input is inside the target column text
                                if user_input in target_column_text:
                                    print(f"Match found: {file_path}")
                                    found_count += 1
                                    
                                    # Stop searching THIS file once a match is found to avoid duplicates
                                    break 
                                    
                except Exception as e:
                    print(f"Error reading {filename}: {e}")

    if found_count == 0:
        print(f"No matches found. (Checked {total_files_checked} target files)")
    else:
        print(f"\nSearch complete. Found matches in {found_count} files (out of {total_files_checked} checked).")

if __name__ == "__main__":
    search_in_target_files()