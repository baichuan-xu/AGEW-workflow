import os
import shutil

def process_and_organize_files(base_directory="."):
    """
    Loops through subfolders L0001 to L2594 to find and process specific JPG files.

    In each 'Lxxxx' subfolder, the script searches for files that start with 'YS'
    and end with '.jpg'. For each matching file found, it creates a new subfolder
    named after the file's base name (e.g., 'YS8' for 'YS8.jpg'), and then
    copies the original file into this new subfolder, renaming it to 'in.png'.

    Args:
        base_directory (str): The path to the directory containing the Lxxxx folders.
                              Defaults to the current directory where the script is run.
    """
    print(f"Starting file processing in: {os.path.abspath(base_directory)}")
    
    # 1. Loop through the specified range of subfolders (L0001 to L2594).
    # The f-string formatting L{i:04d} ensures numbers are padded with leading zeros (e.g., 1 -> 0001).
    for i in range(1, 3356):
        folder_name = f"L{i:04d}"
        folder_path = os.path.join(base_directory, folder_name)

        # First, check if the subfolder actually exists before proceeding.
        if not os.path.isdir(folder_path):
            continue  # If the folder doesn't exist, skip to the next number in the loop.

        print(f"\n--- Scanning folder: {folder_name} ---")
        found_a_match_in_folder = False

        try:
            # 2. Check each item in the current subfolder.
            for filename in os.listdir(folder_path):
                # 3. Check if the file name matches the required pattern (starts with YS, ends with .jpg).
                # We use .lower().endswith() to catch variations like .JPG or .Jpg.
                if filename.startswith("YS") and filename.lower().endswith(".jpg"):
                    found_a_match_in_folder = True
                    print(f"  [+] Found matching file: {filename}")

                    try:
                        # Extract the base name of the file without its extension (e.g., "YS8.jpg" -> "YS8").
                        file_base_name = os.path.splitext(filename)[0]
                        
                        # 4. Create the new subfolder path (e.g., ./L0001/YS8).
                        new_subfolder_path = os.path.join(folder_path, file_base_name)
                        
                        # Create the new directory. exist_ok=True prevents an error if it already exists.
                        os.makedirs(new_subfolder_path, exist_ok=True)
                        print(f"      - Created/verified subfolder: {file_base_name}")

                        # Define the full path for the source and destination files.
                        source_file_path = os.path.join(folder_path, filename)
                        destination_file_path = os.path.join(new_subfolder_path, "in.png")

                        # 5. Copy the source file to the new destination, renaming it in the process.
                        # shutil.copy2 preserves file metadata like creation time.
                        shutil.copy2(source_file_path, destination_file_path)
                        print(f"      - Copied and renamed to: {destination_file_path}")

                    except Exception as e:
                        print(f"  [!] ERROR processing file '{filename}': {e}")
            
            if not found_a_match_in_folder:
                print("  No matching 'YS*.jpg' files found in this folder.")

        except OSError as e:
            print(f"  [!] ERROR accessing folder '{folder_name}': {e}")
            continue

if __name__ == "__main__":
    # To run this script:
    # 1. Save it as a Python file (e.g., process_image_folders.py).
    # 2. Place it in the parent directory that contains all the 'Lxxxx' subfolders.
    # 3. Open a terminal or command prompt, navigate to that directory, and run:
    #    python process_image_folders.py
    
    process_and_organize_files()
    
    print("\n==========================")
    print("  Processing complete.")
    print("==========================")
