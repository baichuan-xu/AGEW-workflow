import os
import csv
import glob
from collections import defaultdict
import sys
from difflib import SequenceMatcher

def clean_cell(cell_data):
    """
    Helper function to clean up CSV cell data, removing extra
    spaces and the double quotes that sometimes surround the string.
    """
    return cell_data.strip().strip('"').strip()

def load_sentences(filepath):
    """
    Loads the global sentence_split.csv file into a dictionary
    for fast lookups.
    Returns: {serial_number: sentence}
    """
    sentence_data = {}
    if not os.path.exists(filepath):
        print(f"  Warning: Global 'sentence_split.csv' not found at {filepath}")
        return sentence_data
        
    try:
        with open(filepath, mode='r', encoding='utf-8', newline='') as infile:
            reader = csv.reader(infile)
            try:
                header = next(reader) # Skip header
            except StopIteration:
                return sentence_data # File is empty

            for row in reader:
                if len(row) >= 2:
                    serial_number, sentence = clean_cell(row[0]), clean_cell(row[1])
                    if serial_number:
                        sentence_data[serial_number] = sentence
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        
    print(f"  Loaded {len(sentence_data)} sentences from 'sentence_split.csv'")
    return sentence_data

def load_process2_data(filepath):
    """
    Loads a Fig*_process_2.csv file into a dictionary.
    Returns: {material_name: "serial,numbers,string"}
    """
    process_data = {}
    try:
        with open(filepath, mode='r', encoding='utf-8', newline='') as infile:
            reader = csv.reader(infile)
            try:
                header = next(reader) # Skip header
            except StopIteration:
                return process_data

            for row in reader:
                if len(row) >= 2:
                    material, serials = clean_cell(row[0]), clean_cell(row[1])
                    if material:
                        process_data[material] = serials
    except FileNotFoundError:
        print(f"    Error: Required file not found: {filepath}")
        return None
    except Exception as e:
        print(f"    Error reading {filepath}: {e}")
        return None
        
    return process_data

def load_composition_data(filepath):
    """
    Loads a Fig*_composition.csv file.
    Returns:
        - list: The header row (elements)
        - dict: {material_name: [list_of_element_values]}
    """
    composition_data = {}
    element_header = []
    
    try:
        with open(filepath, mode='r', encoding='utf-8', newline='') as infile:
            reader = csv.reader(infile)
            try:
                # Read the header row
                header = next(reader)
                if len(header) > 1:
                    element_header = [clean_cell(h) for h in header[1:]]
                else:
                    print(f"    Warning: Composition file has no element columns: {filepath}")
                    return [], {}
                    
            except StopIteration:
                return [], {} # File is empty

            # Read the data rows
            for row in reader:
                if len(row) > 1:
                    material = clean_cell(row[0])
                    elements = [clean_cell(e) for e in row[1:]]
                    if material:
                        composition_data[material] = elements
                        
    except FileNotFoundError:
        print(f"    Error: Required file not found: {filepath}")
        return None, None
    except Exception as e:
        print(f"    Error reading {filepath}: {e}")
        return None, None
        
    return element_header, composition_data

def find_best_match(query, target_keys, cache):
    """
    Finds the key in target_keys that best matches the query.
    Uses a cache to speed up repeated lookups.
    """
    # Use tuple(target_keys) as a hashable cache key
    cache_key = (query, tuple(target_keys))
    if cache_key in cache:
        return cache[cache_key]

    best_ratio = 0.0
    best_match_key = None
    
    for key in target_keys:
        ratio = SequenceMatcher(None, query, key).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match_key = key
    
    # Only return a match if it's reasonably close (e.g., > 60%)
    if best_ratio < 0.6:
        best_match_key = None

    cache[cache_key] = best_match_key
    return best_match_key

def process_figure_files(step2_filepath, process2_filepath, composition_filepath, output_filepath, sentence_data):
    """
    The core function to process one set of Figure files.
    """
    print(f"  Processing: {os.path.basename(step2_filepath)}")
    
    # 1. Load the lookup files for this figure
    element_header, composition_data = load_composition_data(composition_filepath)
    process2_data = load_process2_data(process2_filepath)

    if element_header is None or process2_data is None:
        print(f"    Skipping {os.path.basename(step2_filepath)} due to missing lookup files.")
        return

    # Cache for string matching (re-used for each row)
    match_cache = {}
    
    # Get the keys (material names) from our lookups
    process2_keys = process2_data.keys()
    composition_keys = composition_data.keys()

    # --- DEBUGGING ---
    # Let's check if the keys loaded correctly.
    if not process2_keys:
        print(f"    Warning: No data loaded from {os.path.basename(process2_filepath)}. Keys are empty.")
    if not sentence_data:
        print(f"    Warning: 'sentence_data' (from sentence_split.csv) is empty. 'sentence_process' will be blank.")
    # --- END DEBUGGING ---

    try:
        # Open the output file
        with open(output_filepath, mode='w', encoding='utf-8', newline='') as outfile:
            writer = csv.writer(outfile)
            
            # (6) Write the new header
            output_header = ['material_name', 'x_new', 'y_new', 'material_process', 'sentence_process'] + element_header
            writer.writerow(output_header)
            
            # (2) Open and read the main input file (Fig*_out_step2.csv)
            with open(step2_filepath, mode='r', encoding='utf-8', newline='') as infile:
                reader = csv.reader(infile)
                try:
                    next(reader) # Skip header
                except StopIteration:
                    print(f"    Warning: Input file is empty: {step2_filepath}")
                    return

                # Process each row from Fig*_out_step2.csv
                for row in reader:
                    if len(row) < 3: continue # Skip malformed rows
                    
                    x_new = clean_cell(row[0])
                    y_new = clean_cell(row[1])
                    material_name = clean_cell(row[2])
                    material_process = clean_cell(row[3]) if len(row) > 3 else "" # Handle missing col

                    # (3) & (4): Find matching sentences
                    sentence_process = ""
                    best_match_process = find_best_match(material_name, process2_keys, match_cache)
                    
                    # --- DEBUGGING ---
                    if best_match_process is None:
                        # Find out why the match failed
                        if material_name.strip(): # Don't print for empty material names
                            print(f"    DEBUG: Match failed for material: '{material_name}'")
                            # Use a flag on the function to print keys only once per file
                            if not hasattr(process_figure_files, 'debug_printed_keys_' + process2_filepath): 
                                print(f"    DEBUG: Available keys in {os.path.basename(process2_filepath)}: {list(process2_keys)}")
                                setattr(process_figure_files, 'debug_printed_keys_' + process2_filepath, True)
                    # --- END DEBUGGING ---

                    if best_match_process:
                        serial_string = process2_data[best_match_process]
                        serial_numbers = [sn.strip() for sn in serial_string.split(',')]
                        
                        sentences = []
                        for sn in serial_numbers:
                            if sn in sentence_data:
                                sentences.append(sentence_data[sn])
                        
                        # --- DEBUGGING ---
                        if not sentences and serial_numbers: # Only print if we expected sentences
                            print(f"    DEBUG: Found material '{best_match_process}' with serials '{serial_string}', but no matching sentences found in sentence_data.")
                        # --- END DEBUGGING ---
                        
                        sentence_process = " ".join(sentences) # Concatenate sentences

                    # (5): Find matching elements
                    elements = []
                    best_match_comp = find_best_match(material_name, composition_keys, match_cache)
                    
                    if best_match_comp:
                        elements = composition_data[best_match_comp]
                    else:
                        # Ensure we have the right number of empty columns
                        elements = [""] * len(element_header)

                    # (6) Create and write the new aggregated row
                    output_row = [material_name, x_new, y_new, material_process, sentence_process] + elements
                    writer.writerow(output_row)
                    
        print(f"  Successfully created: {os.path.basename(output_filepath)}")

    except Exception as e:
        print(f"  CRITICAL ERROR processing {step2_filepath}: {e}")


def main():
    """
    Main function to loop through specified folders and process files.
    """
    base_path = os.getcwd() # Get the current working directory
    print(f"Starting batch processing in: {base_path}")
    
    # --- REMOVED GLOBAL SENTENCE LOADING ---
    # Load the global sentence_split.csv file ONCE
    # sentence_split_path = os.path.join(base_path, 'sentence_split.csv')
    # sentence_data = load_sentences(sentence_split_path)
    #
    # if not sentence_data:
    #     print("Warning: 'sentence_split.csv' is empty or not found. 'sentence_process' column will be blank.")
    # --- END REMOVAL ---
        
    processed_file_count = 0
    
    # 1. Loop through folders L0001 to L2594
    for i in range(1, 3356):
        folder_name = f"L{i:04d}" # Formats the number as 0001, 0002, etc.
        folder_path = os.path.join(base_path, folder_name)
        
        if not os.path.isdir(folder_path):
            continue

        print(f"\nChecking folder: {folder_name}")
        
        # --- MODIFICATION ---
        # Load the sentence_split.csv for THIS folder.
        # Based on your description, this file is in the L* folder.
        sentence_split_path = os.path.join(folder_path, 'sentence_split.csv')
        sentence_data = load_sentences(sentence_split_path)
        
        if not sentence_data:
            print(f"  Warning: 'sentence_split.csv' not found or empty in {folder_name}.")
        # --- END MODIFICATION ---
        
        # 2. Find all target files (Fig*_out_step2.csv)
        search_pattern = os.path.join(folder_path, 'Fig*_out_step2.csv')
        target_files = glob.glob(search_pattern)
        
        if not target_files:
            print(f"  No 'Fig*_out_step2.csv' files found.")
            continue
            
        # Process each target file found
        for step2_filepath in target_files:
            
            # Extract the 'FigX' prefix
            filename = os.path.basename(step2_filepath)
            fig_prefix = filename.split('_out_step2.csv')[0] # e.g., "Fig8"
            
            # Define the paths for the other required files
            process2_filepath = os.path.join(folder_path, f"{fig_prefix}_process_2.csv")
            composition_filepath = os.path.join(folder_path, f"{fig_prefix}_composition.csv")
            output_filepath = os.path.join(folder_path, f"{fig_prefix}_out_2.csv")

            # Check if all required source files exist before processing
            if not os.path.exists(process2_filepath):
                print(f"    Skipping {filename}: Missing {os.path.basename(process2_filepath)}")
                continue
            if not os.path.exists(composition_filepath):
                print(f"    Skipping {filename}: Missing {os.path.basename(composition_filepath)}")
                continue
                
            process_figure_files(
                step2_filepath, 
                process2_filepath, 
                composition_filepath, 
                output_filepath, 
                sentence_data
            )
            processed_file_count += 1
            
    print(f"\nBatch processing complete. Total files processed: {processed_file_count}")

if __name__ == "__main__":
    main()