import os
import json
import pandas as pd

# Define the root directory
root_dir = r".\HEA_finetune\Yield_Strength"

def process_csv_file(subfolder_path, csv_path):
    try:
        # Check for figure JSON files (figure_mod.json, figure.json)
        figure_json_files = [
            os.path.join(subfolder_path, 'figure_mod.json'),
            os.path.join(subfolder_path, 'figure.json')
        ]
        
        figure_json_path = None
        for path in figure_json_files:
            if os.path.isfile(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get('yield_strength_data') and isinstance(data['yield_strength_data'], list) and len(data['yield_strength_data']) > 0:
                        figure_json_path = path
                        break
        
        # Check for table JSON files (table_mod.json, table.json)
        table_json_files = [
            os.path.join(subfolder_path, 'table_mod.json'),
            os.path.join(subfolder_path, 'table.json')
        ]
        
        table_json_path = None
        for path in table_json_files:
            if os.path.isfile(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get('yield_strength_data') and isinstance(data['yield_strength_data'], list) and len(data['yield_strength_data']) > 0:
                        table_json_path = path
                        break
        
        # Skip if neither figure nor table JSON files exist with non-empty yield_strength_data
        if not figure_json_path and not table_json_path:
            return False
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Define output paths
        figure_csv_path = os.path.join(subfolder_path, 'figure_data_composition.csv')
        table_csv_path = os.path.join(subfolder_path, 'table_data_composition.csv')
        
        # Case 1: Only figure JSON exists
        if figure_json_path and not table_json_path:
            df.to_csv(figure_csv_path, index=False)
            return True
        
        # Case 2: Only table JSON exists
        if table_json_path and not figure_json_path:
            df.to_csv(table_csv_path, index=False)
            return True
        
        # Case 3: Both figure and table JSON exist
        # Read yield_strength_value from figure JSON
        figure_values = set()
        with open(figure_json_path, 'r', encoding='utf-8') as f:
            figure_data = json.load(f)
            for item in figure_data.get('yield_strength_data', []):
                if 'yield_strength_value' in item:
                    figure_values.add(str(item['yield_strength_value']))
        
        # Read yield_strength_value from table JSON
        table_values = set()
        with open(table_json_path, 'r', encoding='utf-8') as f:
            table_data = json.load(f)
            for item in table_data.get('yield_strength_data', []):
                if 'yield_strength_value' in item:
                    table_values.add(str(item['yield_strength_value']))
        
        # Split rows based on yield_strength_value matches
        figure_df = df[df['yield_strength_value'].astype(str).isin(figure_values)]
        table_df = df[df['yield_strength_value'].astype(str).isin(table_values)]
        
        # Write to respective CSV files if there are matching rows
        if not figure_df.empty:
            figure_df.to_csv(figure_csv_path, index=False)
        if not table_df.empty:
            table_df.to_csv(table_csv_path, index=False)
        
        return True
    
    except Exception:
        print(csv_path)
        return False

def main():
    # Iterate through subfolders L0001 to L3316
    for i in range(1, 3317):
        subfolder = f"L{i:04d}"
        subfolder_path = os.path.join(root_dir, subfolder)
        
        if not os.path.isdir(subfolder_path):
            continue
        
        # Check for data+composition.csv
        csv_path = os.path.join(subfolder_path, 'data+composition.csv')
        if not os.path.isfile(csv_path):
            continue
        
        process_csv_file(subfolder_path, csv_path)

if __name__ == "__main__":
    main()