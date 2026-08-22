import os
import pandas as pd
import numpy as np

# Define the root directory
root_dir = r".\HEA_finetune\Yield_Strength"

def process_csv_file(subfolder_path, csv_path):
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Check if 108th column exists
        if len(df.columns) < 108:
            return
        
        # Extract columns 6 to 108 (indices 5 to 107)
        cols = df.iloc[:, 5:108]
        
        # Convert to numeric, treating non-numeric as 0
        cols = cols.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Find rows where all values in columns 6 to 108 are 0
        zero_rows = (cols == 0).all(axis=1)
        
        if zero_rows.any():
            # Get indices of zero rows (0-based)
            zero_indices = df.index[zero_rows].tolist()
            # Convert to 1-based row numbers (including header)
            row_numbers = [idx + 2 for idx in zero_indices]
            print(f"{subfolder_path}: {row_numbers}")
    
    except Exception:
        pass  # Silent on errors

def main():
    # Iterate through subfolders L0001 to L3316
    for i in range(1, 3317):
        subfolder = f"L{i:04d}"
        subfolder_path = os.path.join(root_dir, subfolder)
        
        if not os.path.isdir(subfolder_path):
            continue
        
        # Check for figure_data_composition.csv
        csv_path = os.path.join(subfolder_path, 'figure_data_composition.csv')
        if not os.path.isfile(csv_path):
            continue
        
        process_csv_file(subfolder_path, csv_path)

if __name__ == "__main__":
    main()