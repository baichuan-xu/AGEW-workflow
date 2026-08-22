import os
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def read_fit_parameters(fit_path: Path) -> Optional[Tuple[float, float, float, float]]:
    """
    Read fit.csv and extract a1, b1, a2, b2 from the second row.
    """
    try:
        df = pd.read_csv(fit_path)
        if len(df) < 1:
            print(f"    ⚠ fit.csv is empty: {fit_path}")
            return None
        
        a1 = df.iloc[0]['a1']
        b1 = df.iloc[0]['b1']
        a2 = df.iloc[0]['a2']
        b2 = df.iloc[0]['b2']
        
        return (a1, b1, a2, b2)
    
    except Exception as e:
        print(f"    ✗ Error reading fit.csv: {str(e)}")
        return None


def read_label_mapping(label_path: Path) -> Optional[Dict[str, str]]:
    """
    Read label.csv and create a mapping from color to label.
    Returns a dictionary: {color: label}
    """
    try:
        df = pd.read_csv(label_path)
        
        if 'label' not in df.columns or 'color' not in df.columns:
            print(f"    ⚠ label.csv missing required columns: {label_path}")
            return None
        
        # Create color -> label mapping
        color_to_label = dict(zip(df['color'], df['label']))
        
        return color_to_label
    
    except Exception as e:
        print(f"    ✗ Error reading label.csv: {str(e)}")
        return None


def extract_color_from_filename(filename: str) -> Optional[str]:
    """
    Extract color from filename like 'out_black.csv' -> 'black'
    """
    if not filename.startswith('out_') or not filename.endswith('.csv'):
        return None
    
    # Remove 'out_' prefix and '.csv' suffix
    color = filename[4:-4]
    return color


def process_out_files(working_dir: Path, a1: float, b1: float, a2: float, b2: float, 
                      color_to_label: Dict[str, str]) -> bool:
    """
    Process all out_*.csv files and create a combined out.csv file.
    """
    # Find all out_*.csv files
    out_files = sorted([f for f in working_dir.iterdir() 
                       if f.is_file() and f.name.startswith('out_') and f.name.endswith('.csv')])
    
    if not out_files:
        print(f"    ⊘ No out_*.csv files found")
        return False
    
    print(f"    Found {len(out_files)} out_*.csv file(s)")
    
    all_data = []
    
    for out_file in out_files:
        color = extract_color_from_filename(out_file.name)
        
        if color is None:
            print(f"    ⚠ Could not extract color from: {out_file.name}")
            continue
        
        # Get corresponding label
        label = color_to_label.get(color)
        
        if label is None:
            print(f"    ⚠ No label found for color '{color}', skipping {out_file.name}")
            continue
        
        try:
            # Read the out_*.csv file
            df = pd.read_csv(out_file)
            
            # Verify required columns
            if 'X' not in df.columns or 'Y' not in df.columns:
                print(f"    ⚠ Missing X or Y columns in {out_file.name}")
                continue
            
            # Transform coordinates
            # x_new = X * a2 + b2
            # y_new = Y * a1 + b1
            x_new = df['X'] * a2 + b2
            y_new = df['Y'] * a1 + b1
            
            # Create dataframe for this file
            temp_df = pd.DataFrame({
                'x_new': x_new,
                'y_new': y_new,
                'label': label
            })
            
            all_data.append(temp_df)
            print(f"      ✓ Processed {out_file.name} -> label: {label} ({len(temp_df)} points)")
        
        except Exception as e:
            print(f"    ✗ Error processing {out_file.name}: {str(e)}")
            continue
    
    if not all_data:
        print(f"    ⚠ No data to save")
        return False
    
    # Concatenate all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save to out.csv
    output_path = working_dir / 'out.csv'
    combined_df.to_csv(output_path, index=False)
    
    print(f"    ✓ Saved: {output_path} (Total: {len(combined_df)} points)")
    
    return True


def process_directory(working_dir: Path) -> bool:
    """
    Process a single directory containing fit.csv, label.csv, and out_*.csv files.
    """
    # Check for fit.csv
    fit_path = working_dir / 'fit2.csv'
    if not fit_path.exists():
        print(f"    ⊘ No fit2.csv found")
        return False
    
    # Check for label.csv
    label_path = working_dir / 'label.csv'
    if not label_path.exists():
        print(f"    ⊘ No label.csv found")
        return False
    
    # Read fit parameters
    fit_params = read_fit_parameters(fit_path)
    if fit_params is None:
        return False
    
    a1, b1, a2, b2 = fit_params
    print(f"    Fit parameters: a1={a1:.6f}, b1={b1:.6f}, a2={a2:.6f}, b2={b2:.6f}")
    
    # Read label mapping
    color_to_label = read_label_mapping(label_path)
    if color_to_label is None:
        return False
    
    print(f"    Label mapping: {color_to_label}")
    
    # Process all out_*.csv files
    return process_out_files(working_dir, a1, b1, a2, b2, color_to_label)


def main():
    """
    Main function to process all L#### folders with YS* subfolders.
    """
    current_path = Path.cwd()
    
    print("=" * 80)
    print("Starting batch processing of out_*.csv files")
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
    print(f"  Successfully created out.csv: {success_count}")
    print(f"  Failed: {processed_count - success_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()