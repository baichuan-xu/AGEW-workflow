import os
from pathlib import Path
import re
from typing import List, Optional, Tuple


def find_fig_composition_files(folder: Path) -> List[Path]:
    """
    Find all Fig*_composition.json files in the folder.
    Returns list of file paths.
    """
    pattern = re.compile(r'^Fig\d+_composition\.json$', re.IGNORECASE)
    
    fig_files = []
    for file in folder.iterdir():
        if file.is_file() and pattern.match(file.name):
            fig_files.append(file)
    
    return sorted(fig_files)


def find_json_block(lines: List[str]) -> Optional[Tuple[int, int]]:
    """
    Find the JSON block markers in the text.
    Returns (N1, N2) where:
    - N1: index of the last line containing "```json" (searching from bottom to top)
    - N2: index of the first line containing "```" after N1 (searching downwards)
    Returns None if markers not found.
    """
    # Find N1: last occurrence of "```json" (search from bottom to top)
    n1 = None
    for i in range(len(lines) - 1, -1, -1):
        if '```json' in lines[i]:
            n1 = i
            break
    
    if n1 is None:
        return None
    
    # Find N2: first occurrence of "```" after N1 (search downwards)
    n2 = None
    for i in range(n1 + 1, len(lines)):
        if '```' in lines[i]:
            n2 = i
            break
    
    if n2 is None:
        return None
    
    return (n1, n2)


def extract_json_content(file_path: Path) -> bool:
    """
    Extract JSON content from text file:
    1. Read file as text
    2. Find last "```json" marker (N1)
    3. Find first "```" marker after N1 (N2)
    4. Keep only content between N1 and N2 (excluding N1 and N2)
    5. Save back to the file
    Returns True if successful, False otherwise.
    """
    try:
        # Read file as text
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) == 0:
            print(f"    ⚠ File is empty: {file_path.name}")
            return False
        
        # Find JSON block markers
        markers = find_json_block(lines)
        
        if markers is None:
            print(f"    ⚠ JSON markers not found in: {file_path.name}")
            return False
        
        n1, n2 = markers
        
        # Extract content between N1 and N2 (excluding N1 and N2)
        json_content = lines[n1 + 1:n2]
        
        if not json_content:
            print(f"    ⚠ No content between markers in: {file_path.name}")
            return False
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(json_content)
        
        print(f"    ✓ Processed: {file_path.name} (kept {len(json_content)} lines)")
        return True
    
    except Exception as e:
        print(f"    ✗ Error processing {file_path.name}: {str(e)}")
        return False


def process_l_folder(l_folder: Path) -> bool:
    """
    Process a single L#### folder:
    - Find all Fig*_composition.json files
    - Extract JSON content from each file
    """
    # Find all Fig*_composition.json files
    fig_files = find_fig_composition_files(l_folder)
    
    if not fig_files:
        return False
    
    print(f"  Found {len(fig_files)} Fig*_composition.json file(s)")
    
    success_count = 0
    
    # Process each file
    for fig_file in fig_files:
        if extract_json_content(fig_file):
            success_count += 1
    
    return success_count > 0


def main():
    """
    Main function to process all L#### folders.
    """
    current_path = Path.cwd()
    
    print("=" * 80)
    print("Starting extraction of JSON content from Fig*_composition.json files")
    print("=" * 80)
    
    processed_count = 0
    success_count = 0
    
    # Loop through L0001 to L2594
    for i in range(1, 3356):
        folder_name = f"L{i:04d}"
        folder_path = current_path / folder_name
        
        if not folder_path.exists() or not folder_path.is_dir():
            continue
        
        # Check for Fig*_composition.json files
        if not find_fig_composition_files(folder_path):
            continue
        
        print(f"\n[{folder_name}]")
        
        processed_count += 1
        
        if process_l_folder(folder_path):
            success_count += 1
    
    print("\n" + "=" * 80)
    print(f"Processing complete!")
    print(f"  Total L#### folders with Fig*_composition.json: {processed_count}")
    print(f"  Successfully processed: {success_count}")
    print(f"  Failed: {processed_count - success_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()