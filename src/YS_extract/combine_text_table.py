import os

def append_table_to_text_in_subfolders(base_path):
    # Ensure the base path exists
    if not os.path.exists(base_path):
        print(f"Error: Base path {base_path} does not exist!")
        return
    
    # Loop through subfolders from L0001 to L1281
    for i in range(1, 3317):
        folder_name = f"L{i:04d}"  # Format folder name as L0001, L0002, ..., L1281
        folder_path = os.path.join(base_path, folder_name)
        
        # Check if the folder exists
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist. Skipping...")
            continue
        
        # Define file paths inside the folder
        table_file = os.path.join(folder_path, "table.txt")
        target_file = os.path.join(folder_path, "text.txt")
        
        # Check if both files exist
        if not os.path.exists(table_file):
            print(f"Warning: {table_file} not found. Skipping folder {folder_name}...")
            continue
        
        if not os.path.exists(target_file):
            print(f"Warning: {target_file} not found. Skipping folder {folder_name}...")
            continue
        
        # Read table content
        try:
            with open(table_file, 'r', encoding='utf-8') as f:
                table_content = f.read()
        except Exception as e:
            print(f"Error reading {table_file}: {str(e)}")
            continue
        
        # Append to target file
        try:
            with open(target_file, 'a', encoding='utf-8') as f:
                f.write('\n\n')  # Add spacing before table
                f.write(table_content)
                f.write('\n')    # Add spacing after table
            print(f"Successfully appended {table_file} to {target_file}")
        except Exception as e:
            print(f"Error writing to {target_file}: {str(e)}")

if __name__ == "__main__":
    base_path = r".\HEA_finetune\Yield_Strength"
    append_table_to_text_in_subfolders(base_path)