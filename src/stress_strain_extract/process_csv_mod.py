import os
import csv
import glob
from collections import defaultdict
import sys

def process_csv_file(input_filepath):
    """
    Reads a Fig*_process.csv file, inverts its structure,
    and writes a new Fig*_process_2.csv file.

    The input format is expected to be:
    serial_number,materials
    1,"MaterialA,MaterialB"
    2,"MaterialA"
    3,"MaterialB,MaterialC"

    The output format will be:
    material,serial_number
    MaterialA,"1,2"
    MaterialB,"1,3"
    MaterialC,"3"
    """
    
    # Define the output filepath
    output_filepath = input_filepath.replace('_process.csv', '_process_2.csv')
    
    # Use defaultdict to automatically handle list initialization
    materials_map = defaultdict(list)
    
    print(f"  Processing file: {input_filepath}")

    try:
        with open(input_filepath, mode='r', encoding='utf-8', newline='') as infile:
            # Use csv.reader to handle quoted fields correctly
            reader = csv.reader(infile)
            
            try:
                # Skip the header row
                header = next(reader)
                if header != ['serial_number', 'materials']:
                    print(f"    Warning: Unexpected header in {input_filepath}: {header}")
            except StopIteration:
                print(f"    Warning: File is empty {input_filepath}")
                return # Skip empty file

            # Process each data row
            for row in reader:
                if not row or len(row) < 2:
                    # Skip empty or malformed rows
                    continue
                
                serial_number = row[0].strip()
                materials_string = row[1].strip()
                
                # Split the materials string by comma.
                # This works for "MaterialA,MaterialB" and "MaterialA"
                materials_list = materials_string.split(',')
                
                for material in materials_list:
                    # Clean up each material name (remove extra spaces or quotes)
                    cleaned_material = material.strip().strip('"').strip()
                    
                    if cleaned_material: # Avoid adding empty strings
                        materials_map[cleaned_material].append(serial_number)

    except FileNotFoundError:
        print(f"  Error: File not found {input_filepath}")
        return
    except Exception as e:
        print(f"  Error reading {input_filepath}: {e}")
        return

    # Check if we actually processed any data
    if not materials_map:
        print(f"    No data found or processed in {input_filepath}.")
        return

    # Write the aggregated data to the new CSV file
    try:
        with open(output_filepath, mode='w', encoding='utf-8', newline='') as outfile:
            # The csv.writer will automatically add quotes around fields
            # that contain a comma (like "1,88,90"), which matches
            # your requirement perfectly.
            writer = csv.writer(outfile)
            
            # Write the new header
            writer.writerow(['material', 'serial_number'])
            
            # Write the data rows
            for material, serials in materials_map.items():
                serial_string = ",".join(serials)
                writer.writerow([material, serial_string])
                
        print(f"  Successfully created: {output_filepath}")

    except Exception as e:
        print(f"  Error writing {output_filepath}: {e}")

def main():
    """
    Main function to loop through specified folders and process files.
    """
    base_path = os.getcwd() # Get the current working directory
    print(f"Starting batch processing in: {base_path}")
    
    processed_file_count = 0
    
    # 1. Loop through folders L0001 to L2594
    for i in range(1, 3356):
        folder_name = f"L{i:04d}" # Formats the number as 0001, 0002, etc.
        folder_path = os.path.join(base_path, folder_name)
        
        # Check if the folder actually exists
        if not os.path.isdir(folder_path):
            continue # Skip if folder doesn't exist

        print(f"\nChecking folder: {folder_name}")
        
        # 2. Find all target files in the folder
        # Use glob to find files matching the pattern
        search_pattern = os.path.join(folder_path, 'Fig*_process.csv')
        target_files = glob.glob(search_pattern)
        
        if not target_files:
            print(f"  No 'Fig*_process.csv' files found.")
            continue
            
        # Process each target file found
        for filepath in target_files:
            # Avoid re-processing an output file if it matches the pattern
            if filepath.endswith('_process_2.csv'):
                continue
                
            process_csv_file(filepath)
            processed_file_count += 1
            
    print(f"\nBatch processing complete. Total files processed: {processed_file_count}")

if __name__ == "__main__":
    main()