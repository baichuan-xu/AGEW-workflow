import os
import pandas as pd
from pathlib import Path
from typing import List, Dict
import re
import warnings
warnings.filterwarnings('ignore')


def find_fig_process_files(folder: Path) -> List[int]:
    """
    Find all Fig*_process.csv files and extract figure numbers.
    Returns list of figure numbers (e.g., [2, 10]).
    """
    pattern = re.compile(r'^Fig(\d+)_process\.csv$', re.IGNORECASE)
    
    fig_numbers = []
    for file in folder.iterdir():
        if file.is_file():
            match = pattern.match(file.name)
            if match:
                fig_num = int(match.group(1))
                fig_numbers.append(fig_num)
    
    return sorted(fig_numbers)


def find_all_out_filter_files(ys_folder: Path) -> List[Path]:
    """
    Recursively find all out_filter.csv files in YS folder and all its subfolders.
    Returns list of paths to out_filter.csv files.
    """
    out_filter_files = []
    
    # Walk through all subdirectories recursively
    for root, dirs, files in os.walk(ys_folder):
        if 'out_filter.csv' in files:
            out_filter_path = Path(root) / 'out_filter.csv'
            out_filter_files.append(out_filter_path)
    
    return out_filter_files


def merge_out_filter_files(out_filter_paths: List[Path]) -> pd.DataFrame:
    """
    Merge multiple out_filter.csv files, skipping header rows.
    Returns a single DataFrame with all data (excluding headers).
    """
    all_data = []
    
    for file_path in out_filter_paths:
        try:
            # Read CSV, skip header
            df = pd.read_csv(file_path)
            all_data.append(df)
            print(f"      ✓ Read: {file_path.relative_to(file_path.parents[2])} ({len(df)} rows)")
        except Exception as e:
            print(f"      ✗ Error reading {file_path}: {str(e)}")
    
    if not all_data:
        return None
    
    # Concatenate all dataframes
    merged_df = pd.concat(all_data, ignore_index=True)
    return merged_df


def process_figure(l_folder: Path, fig_num: int) -> bool:
    """
    Process a single figure:
    - Find corresponding YS folder
    - Collect all out_filter.csv files recursively
    - Merge and save to FigX_out.csv
    """
    ys_folder_name = f"YS{fig_num}"
    ys_folder = l_folder / ys_folder_name
    
    if not ys_folder.exists() or not ys_folder.is_dir():
        print(f"    ⚠ Folder {ys_folder_name} not found")
        return False
    
    print(f"    Processing: {ys_folder_name}")
    
    # Find all out_filter.csv files recursively
    out_filter_files = find_all_out_filter_files(ys_folder)
    
    if not out_filter_files:
        print(f"      ⊘ No out_filter.csv files found in {ys_folder_name}")
        return False
    
    print(f"      Found {len(out_filter_files)} out_filter.csv file(s)")
    
    # Merge all out_filter.csv files
    merged_df = merge_out_filter_files(out_filter_files)
    
    if merged_df is None or len(merged_df) == 0:
        print(f"      ⚠ No data to merge")
        return False
    
    # Save to FigX_out.csv in L folder
    output_file = l_folder / f"Fig{fig_num}_out.csv"
    
    try:
        merged_df.to_csv(output_file, index=False)
        print(f"    ✓ Saved: {output_file.name} (Total: {len(merged_df)} rows)")
        return True
    except Exception as e:
        print(f"    ✗ Error saving {output_file}: {str(e)}")
        return False


def process_l_folder(l_folder: Path) -> bool:
    """
    Process a single L#### folder:
    - Find all Fig*_process.csv files
    - For each figure, merge corresponding YS folder's out_filter.csv files
    """
    # Find all Fig*_process.csv files
    fig_numbers = find_fig_process_files(l_folder)
    
    if not fig_numbers:
        return False
    
    print(f"  Found Fig*_process.csv files for figures: {fig_numbers}")
    
    success_count = 0
    
    # Process each figure
    for fig_num in fig_numbers:
        if process_figure(l_folder, fig_num):
            success_count += 1
    
    return success_count > 0


def main():
    """
    Main function to process all L#### folders.
    """
    current_path = Path.cwd()
    
    print("=" * 80)
    print("Starting merging of out_filter.csv files by figure")
    print("=" * 80)
    
    processed_count = 0
    success_count = 0
    
    # Loop through L0001 to L2594
    for i in range(1, 3356):
        folder_name = f"L{i:04d}"
        folder_path = current_path / folder_name
        
        if not folder_path.exists() or not folder_path.is_dir():
            continue
        
        # Check for Fig*_process.csv files
        if not find_fig_process_files(folder_path):
            continue
        
        print(f"\n[{folder_name}]")
        
        processed_count += 1
        
        if process_l_folder(folder_path):
            success_count += 1
    
    print("\n" + "=" * 80)
    print(f"Processing complete!")
    print(f"  Total L#### folders with Fig*_process.csv: {processed_count}")
    print(f"  Successfully processed: {success_count}")
    print(f"  Failed: {processed_count - success_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()