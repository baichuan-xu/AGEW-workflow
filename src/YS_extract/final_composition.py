import os
import json
import csv
import pandas as pd
from pathlib import Path

# Define the root directory
root_dir = r".\HEA_finetune\Yield_Strength"

# List of all elements in the periodic table (simplified, can be expanded)
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
        
        # Check if materials exist
        if 'materials' not in data:
            print(f"Error: No 'materials' key in {json_file_path}")
            return
        
        # Prepare CSV data
        csv_data = []
        for material in data['materials']:
            row = {'name': material.get('name', '')}
            # Initialize all elements to 0.0
            for element in elements:
                row[element] = 0.0
            # Fill in composition values if available and valid
            if 'composition' in material and isinstance(material['composition'], dict):
                for element, value in material['composition'].items():
                    if element in elements:
                        row[element] = float(value)
            else:
                print(f"Error: Invalid or missing composition in material '{material.get('name', 'unknown')}' in {json_file_path}")
                continue
            csv_data.append(row)
        
        # Only write CSV if there is valid data
        if csv_data:
            # Write to CSV using pandas for better handling
            df = pd.DataFrame(csv_data)
            # Reorder columns: name first, then elements
            columns = ['name'] + elements
            df = df[columns]
            df.to_csv(output_csv_path, index=False)
        else:
            print(f"Error: No valid materials to process in {json_file_path}")

    except Exception as e:
        print(f"Error processing {json_file_path}: {str(e)}")

def main():
    # Iterate through subfolders L0001 to L3316
    for i in range(1, 3317):
        subfolder = f"L{i:04d}"
        subfolder_path = os.path.join(root_dir, subfolder)
        
        if not os.path.isdir(subfolder_path):
            continue
        
        # Check for JSON files in order of priority
        json_files = [
            'composition3.json',
            'composition2.json',
            'composition_mod.json'
        ]
        
        json_file_path = None
        for json_file in json_files:
            potential_path = os.path.join(subfolder_path, json_file)
            if os.path.isfile(potential_path):
                json_file_path = potential_path
                break
        
        if json_file_path:
            output_csv_path = os.path.join(subfolder_path, 'composition.csv')
            process_json_file(json_file_path, output_csv_path)
        # No need to print for folders with no JSON files

if __name__ == "__main__":
    main()