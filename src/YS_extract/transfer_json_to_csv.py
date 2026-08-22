import os
import json
import pandas as pd

# Define the root directory
root_dir = r".\HEA_finetune\Yield_Strength"

# List of all elements in the periodic table
elements = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"
]

def process_json_file(json_file_path, output_csv_path):
    try:
        # Read JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if yield_strength_data exists and is non-empty
        if 'yield_strength_data' not in data or not data['yield_strength_data']:
            return False
        
        # Prepare CSV data
        csv_data = []
        for item in data['yield_strength_data']:
            row = {
                'material_name': item.get('material_name', ''),
                'yield_strength_value': item.get('yield_strength_value', ''),
                'yield_strength_unit': item.get('yield_strength_unit', ''),
                'processing_text': item.get('processing_text', ''),
                'processing_text_supply': item.get('processing_text_supply', '')
            }
            # Initialize all elements to 0.0
            for element in elements:
                row[element] = 0.0
            # Fill in composition values if available
            if 'composition' in item and isinstance(item['composition'], dict):
                for element, value in item['composition'].items():
                    if element in elements:
                        row[element] = float(value)
            csv_data.append(row)
        
        # Write to CSV using pandas
        df = pd.DataFrame(csv_data)
        # Define column order
        columns = ['material_name', 'yield_strength_value', 'yield_strength_unit', 
                  'processing_text', 'processing_text_supply'] + elements
        df = df[columns]
        df.to_csv(output_csv_path, index=False)
        return True
    
    except Exception:
        print(json_file_path)
        return False

def main():
    # Iterate through subfolders L0001 to L3316
    for i in range(1, 3317):
        subfolder = f"L{i:04d}"
        subfolder_path = os.path.join(root_dir, subfolder)
        
        if not os.path.isdir(subfolder_path):
            continue
        
        # Check for JSON files in order of priority
        json_files = [
            'data+composition_mod.json',
            'data+composition.json'
        ]
        
        json_file_path = None
        for json_file in json_files:
            potential_path = os.path.join(subfolder_path, json_file)
            if os.path.isfile(potential_path):
                json_file_path = potential_path
                break
        
        if not json_file_path:
            continue
        
        output_csv_path = os.path.join(subfolder_path, 'data+composition.csv')
        process_json_file(json_file_path, output_csv_path)

if __name__ == "__main__":
    main()