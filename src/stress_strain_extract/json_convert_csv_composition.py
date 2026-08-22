"""
Batch Composition File Processor
Converts JSON composition files to CSV format with complete periodic table columns
"""

import json
import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


# Complete list of elements from H to Lr
ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr"
]


def find_composition_files(folder_path: str) -> List[Tuple[str, str]]:
    """
    Find all FigX_composition.json files in a folder.
    
    Args:
        folder_path: Path to the folder to search
        
    Returns:
        List of tuples (filename, full_path) sorted by figure number
    """
    composition_files = []
    pattern = re.compile(r'^Fig(\d+)_composition\.json$')
    
    try:
        for file in os.listdir(folder_path):
            match = pattern.match(file)
            if match:
                fig_number = int(match.group(1))
                full_path = os.path.join(folder_path, file)
                composition_files.append((fig_number, file, full_path))
    except FileNotFoundError:
        return []
    
    # Sort by figure number
    composition_files.sort(key=lambda x: x[0])
    
    return [(file, path) for _, file, path in composition_files]


def convert_json_to_csv(json_path: str, output_path: str) -> bool:
    """
    Convert a composition JSON file to CSV format with all elements as columns.
    
    Args:
        json_path: Path to the input JSON file
        output_path: Path to the output CSV file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Prepare CSV data
        csv_rows = []
        
        for material_name, composition in data.items():
            # Initialize row with zeros for all elements
            row = {'Material': material_name}
            for element in ELEMENTS:
                row[element] = 0.0
            
            # Fill in the actual composition values
            for element, value in composition.items():
                if element in ELEMENTS:
                    row[element] = value
                else:
                    print(f"Warning: Element '{element}' not in standard periodic table")
            
            csv_rows.append(row)
        
        # Write CSV file
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            # Create header: Material + all elements
            fieldnames = ['Material'] + ELEMENTS
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(csv_rows)
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {json_path}: {e}")
        return False
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return False


def process_batch_folders(base_path: str = '.', start_folder: int = 1, end_folder: int = 2594):
    """
    Process all L#### folders and convert composition JSON files to CSV.
    
    Args:
        base_path: Base directory containing the L#### folders
        start_folder: Starting folder number (default: 1)
        end_folder: Ending folder number (default: 2594)
    """
    total_folders = 0
    processed_files = 0
    failed_files = 0
    
    print("=" * 70)
    print("Batch Composition File Processor")
    print("=" * 70)
    print(f"Base path: {os.path.abspath(base_path)}")
    print(f"Processing folders: L{start_folder:04d} to L{end_folder:04d}")
    print("=" * 70)
    
    for folder_num in range(start_folder, end_folder + 1):
        folder_name = f"L{folder_num:04d}"
        folder_path = os.path.join(base_path, folder_name)
        
        # Check if folder exists
        if not os.path.isdir(folder_path):
            continue
        
        total_folders += 1
        
        # Find all composition files in this folder
        composition_files = find_composition_files(folder_path)
        
        if not composition_files:
            continue
        
        print(f"\n[{folder_name}] Found {len(composition_files)} composition file(s)")
        
        # Process each composition file
        for filename, json_path in composition_files:
            # Generate output CSV filename
            csv_filename = filename.replace('.json', '.csv')
            csv_path = os.path.join(folder_path, csv_filename)
            
            print(f"  Processing: {filename} -> {csv_filename}...", end=' ')
            
            # Convert JSON to CSV
            if convert_json_to_csv(json_path, csv_path):
                processed_files += 1
                print("✓ Success")
            else:
                failed_files += 1
                print("✗ Failed")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total folders checked: {end_folder - start_folder + 1}")
    print(f"Folders with composition files: {total_folders}")
    print(f"Files successfully processed: {processed_files}")
    print(f"Files failed: {failed_files}")
    print("=" * 70)


def main():
    """
    Main entry point for the script.
    """
    # You can modify these parameters as needed
    base_path = '.'  # Current directory
    start_folder = 1
    end_folder = 3356
    
    # Process all folders
    process_batch_folders(base_path, start_folder, end_folder)


if __name__ == "__main__":
    main()