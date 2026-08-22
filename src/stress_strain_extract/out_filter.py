import os
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


def normalize_label(label: str) -> str:
    """
    Normalize label by removing surrounding quotes and whitespace.
    Examples:
        'Fe' -> 'Fe'
        '"Fe"' -> 'Fe'
        ' Fe ' -> 'Fe'
        '"Cu"' -> 'Cu'
    """
    # Strip whitespace
    label = str(label).strip()
    
    # Remove surrounding double quotes
    if label.startswith('"') and label.endswith('"'):
        label = label[1:-1]
    
    # Remove surrounding single quotes
    if label.startswith("'") and label.endswith("'"):
        label = label[1:-1]
    
    return label.strip()


def read_estimate_data(estimate_path: Path) -> Optional[Dict[str, Tuple[float, float, float, float]]]:
    """
    Read estimate_data.csv and create a mapping from label to (x1, y1, x2, y2).
    Returns a dictionary: {normalized_label: (x1, y1, x2, y2)}
    """
    try:
        df = pd.read_csv(estimate_path)
        
        # Verify required columns
        required_columns = ['label', 'x1', 'y1', 'x2', 'y2']
        if not all(col in df.columns for col in required_columns):
            print(f"    ⚠ estimate_data.csv missing required columns: {estimate_path}")
            return None
        
        # Create mapping with normalized labels
        estimate_dict = {}
        for _, row in df.iterrows():
            normalized_label = normalize_label(row['label'])
            estimate_dict[normalized_label] = (
                float(row['x1']),
                float(row['y1']),
                float(row['x2']),
                float(row['y2'])
            )
        
        return estimate_dict
    
    except Exception as e:
        print(f"    ✗ Error reading estimate_data.csv: {str(e)}")
        return None


def filter_out_csv(out_path: Path, estimate_path: Path, output_path: Path) -> bool:
    """
    Filter out.csv based on estimate_data.csv conditions.
    Keep rows where:
        - 0 <= x_new <= x1
        - 0 <= y_new <= y2
    Save filtered result to out_filter.csv
    """
    try:
        # Read out.csv
        out_df = pd.read_csv(out_path)
        
        # Verify required columns in out.csv
        if 'x_new' not in out_df.columns or 'y_new' not in out_df.columns or 'label' not in out_df.columns:
            print(f"    ⚠ out.csv missing required columns: {out_path}")
            return False
        
        # Read estimate_data.csv
        estimate_dict = read_estimate_data(estimate_path)
        
        if estimate_dict is None:
            return False
        
        print(f"    Estimate data loaded: {len(estimate_dict)} labels")
        
        # Filter rows
        filtered_rows = []
        total_rows = len(out_df)
        
        for idx, row in out_df.iterrows():
            # Normalize the label from out.csv
            normalized_label = normalize_label(row['label'])
            
            # Check if label exists in estimate_data
            if normalized_label not in estimate_dict:
                # Label not found in estimate_data, skip this row
                continue
            
            # Get estimate boundaries
            x1, y1, x2, y2 = estimate_dict[normalized_label]
            
            x_new = row['x_new']
            y_new = row['y_new']
            
            # Check conditions: 0 <= x_new <= x1 AND 0 <= y_new <= y2
            if 0 <= x_new <= x1 and 0 <= y_new <= y2:
                filtered_rows.append(row)
        
        # Create filtered dataframe
        if filtered_rows:
            filtered_df = pd.DataFrame(filtered_rows)
            filtered_df.to_csv(output_path, index=False)
            
            kept_rows = len(filtered_df)
            removed_rows = total_rows - kept_rows
            
            print(f"    ✓ Saved: {output_path.name}")
            print(f"      Total rows: {total_rows}")
            print(f"      Kept: {kept_rows} ({kept_rows/total_rows*100:.1f}%)")
            print(f"      Removed: {removed_rows} ({removed_rows/total_rows*100:.1f}%)")
            
            return True
        else:
            print(f"    ⚠ No rows passed the filter criteria")
            # Still create an empty file with headers
            empty_df = pd.DataFrame(columns=['x_new', 'y_new', 'label'])
            empty_df.to_csv(output_path, index=False)
            print(f"    ✓ Saved empty file: {output_path.name}")
            return True
    
    except Exception as e:
        print(f"    ✗ Error filtering out.csv: {str(e)}")
        return False


def process_directory(working_dir: Path) -> bool:
    """
    Process a single directory containing out.csv and estimate_data.csv.
    """
    # Check for out.csv
    out_path = working_dir / 'out.csv'
    if not out_path.exists():
        print(f"    ⊘ No out.csv found")
        return False
    
    # Check for estimate_data.csv
    estimate_path = working_dir / 'estimate_data.csv'
    if not estimate_path.exists():
        print(f"    ⊘ No estimate_data.csv found")
        return False
    
    # Output path
    output_path = working_dir / 'out_filter.csv'
    
    # Filter and save
    return filter_out_csv(out_path, estimate_path, output_path)


def main():
    """
    Main function to process all L#### folders with YS* subfolders.
    """
    current_path = Path.cwd()
    
    print("=" * 80)
    print("Starting batch filtering of out.csv files")
    print("=" * 80)
    
    processed_count = 0
    success_count = 0
    
    # Loop through L0001 to L2594
    for i in range(1, 3356):
        folder_name = f"L{i:04d}"
        folder_path = current_path / folder_name
        
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
                for split_folder in split_folders:
                    in_png = split_folder / 'in.png'
                    
                    if not in_png.exists():
                        print(f"    ⊘ No in.png in {split_folder.name}")
                        continue
                    
                    print(f"    Processing: {split_folder.name}/")
                    processed_count += 1
                    
                    if process_directory(split_folder):
                        success_count += 1
            else:
                # No split folders, check for in.png directly in YS folder
                in_png = ys_folder / 'in.png'
                
                if in_png.exists():
                    print(f"    Processing: (direct)")
                    processed_count += 1
                    
                    if process_directory(ys_folder):
                        success_count += 1
                else:
                    print(f"    ⊘ No split* folders or in.png found")
    
    print("\n" + "=" * 80)
    print(f"Processing complete!")
    print(f"  Total directories processed: {processed_count}")
    print(f"  Successfully created out_filter.csv: {success_count}")
    print(f"  Failed: {processed_count - success_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()