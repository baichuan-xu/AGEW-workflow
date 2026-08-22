import os
import pandas as pd

# Define the root directory
root_dir = r".\HEA_finetune\Yield_Strength"

def process_csv_file(subfolder_path, csv_path):
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Check if 108th column exists
        if len(df.columns) < 108:
            return
        
        # Extract columns 4 to 108 (indices 3 to 107)
        cols = df.iloc[:, 3:108].astype(str)  # Convert to string to handle NaN and mixed types
        
        # Group rows by identical values in columns 4 to 108
        grouped = cols.groupby(cols.columns.tolist()).groups
        
        # Filter groups with more than one row (duplicates)
        duplicate_groups = {k: v.tolist() for k, v in grouped.items() if len(v) > 1}
        
        if duplicate_groups:
            # Print subfolder path and row numbers for each group of duplicates
            print(f"{subfolder_path}:")
            for group_key, indices in duplicate_groups.items():
                # Convert 0-based indices to 1-based row numbers (including header)
                row_numbers = [idx + 2 for idx in indices]
                print(f"  {row_numbers}")
    
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
        csv_path = os.path.join(subfolder_path, 'table_data_composition.csv')
        if not os.path.isfile(csv_path):
            continue
        
        process_csv_file(subfolder_path, csv_path)

if __name__ == "__main__":
    main()