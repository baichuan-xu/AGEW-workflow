import os
import pandas as pd

# Define the root directory and output file
root_dir = r".\HEA_finetune\Yield_Strength"
output_csv_path = os.path.join(root_dir, 'figure_data.csv')

def process_csv_files():
    try:
        # Flag to track if header has been written
        header_written = False
        
        # Iterate through subfolders L0001 to L3316
        for i in range(1, 3317):
            subfolder = f"L{i:04d}"
            subfolder_path = os.path.join(root_dir, subfolder)
            
            if not os.path.isdir(subfolder_path):
                continue
            
            # Check for table_data_composition.csv
            csv_path = os.path.join(subfolder_path, 'figure_data_composition.csv')
            if not os.path.isfile(csv_path):
                continue
            
            try:
                # Read CSV, skipping header for all but the first file
                df = pd.read_csv(csv_path)
                
                if df.empty:
                    continue
                
                # Write to output CSV
                if header_written:
                    # Append without header
                    df.to_csv(output_csv_path, mode='a', header=False, index=False)
                else:
                    # Write with header for the first file
                    df.to_csv(output_csv_path, mode='w', header=True, index=False)
                    header_written = True
            
            except Exception:
                print(csv_path)
                continue
        
        return True
    
    except Exception:
        print(output_csv_path)
        return False

def main():
    if process_csv_files():
        if os.path.isfile(output_csv_path):
            print(f"Successfully created {output_csv_path}")
        else:
            print("No table_data_composition.csv files found to process")

if __name__ == "__main__":
    main()