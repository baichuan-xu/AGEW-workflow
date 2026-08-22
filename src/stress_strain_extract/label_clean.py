import os
from pathlib import Path


# ============================================================================
TARGET_FILENAME = "estimate_data.csv"  # Change this to process different files
# ============================================================================


def process_file(file_path: Path) -> bool:
    """
    Process a single CSV file:
    - Read as text
    - Remove first and last lines
    - Save back to the original file
    
    Returns True if successful, False otherwise.
    """
    try:
        # Read the file as text
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Check if file has at least 3 lines (need to keep at least 1 line after removal)
        if len(lines) < 3:
            print(f"    ⚠ File has fewer than 3 lines, skipping: {file_path}")
            return False
        
        # Remove first and last lines
        modified_lines = lines[1:-1]
        
        # Write back to the original file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(modified_lines)
        
        print(f"    ✓ Processed: {file_path.name} ({len(lines)} → {len(modified_lines)} lines)")
        return True
    
    except Exception as e:
        print(f"    ✗ Error processing {file_path}: {str(e)}")
        return False


def main():
    """
    Main function to process all target files in L#### folders.
    """
    current_path = Path.cwd()
    
    print("=" * 80)
    print(f"Starting batch processing of {TARGET_FILENAME} files")
    print("=" * 80)
    
    processed_count = 0
    success_count = 0
    
    # Loop through L0001 to L2594
    for i in range(1, 3356):
        folder_name = f"L{i:04d}"
        folder_path = current_path / folder_name
        
        # Check if folder exists
        if not folder_path.exists() or not folder_path.is_dir():
            continue
        
        print(f"\n[{folder_name}]")
        
        # Find all YS* subfolders
        ys_folders = sorted([f for f in folder_path.iterdir() 
                            if f.is_dir() and f.name.startswith('YS')])
        
        if not ys_folders:
            print(f"  ⊘ No YS* folders found, skipping")
            continue
        
        print(f"  Found {len(ys_folders)} YS* folder(s)")
        
        # Process each YS* folder
        for ys_folder in ys_folders:
            print(f"\n  Processing: {ys_folder.name}")
            
            # Check for split* subfolders
            split_folders = sorted([f for f in ys_folder.iterdir() 
                                   if f.is_dir() and f.name.startswith('split')])
            
            if split_folders:
                # Process each split* folder
                print(f"    Found {len(split_folders)} split* folder(s)")
                
                for split_folder in split_folders:
                    target_file = split_folder / TARGET_FILENAME
                    
                    if not target_file.exists():
                        print(f"    ⊘ No {TARGET_FILENAME} in {split_folder.name}")
                        continue
                    
                    print(f"    Processing: {split_folder.name}/{TARGET_FILENAME}")
                    processed_count += 1
                    
                    if process_file(target_file):
                        success_count += 1
            else:
                # No split folders, check for target file directly in YS folder
                target_file = ys_folder / TARGET_FILENAME
                
                if target_file.exists():
                    print(f"    Processing: {TARGET_FILENAME} (direct)")
                    processed_count += 1
                    
                    if process_file(target_file):
                        success_count += 1
                else:
                    print(f"    ⊘ No split* folders or {TARGET_FILENAME} found")
    
    print("\n" + "=" * 80)
    print(f"Processing complete!")
    print(f"  Total {TARGET_FILENAME} files processed: {processed_count}")
    print(f"  Successfully modified: {success_count}")
    print(f"  Failed: {processed_count - success_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()