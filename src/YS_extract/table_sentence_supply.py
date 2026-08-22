import re
import csv
import os

def read_table_file(file_path):
    """Read the table file and identify table sections."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Find all table sections
    table_sections = {}
    current_table = None
    
    # First, collect all existing table content
    lines = content.split('\n')
    for i, line in enumerate(lines):
        table_match = re.match(r'Table (\d+):', line)
        if table_match:
            current_table = f"Table {table_match.group(1)}"
            table_sections[current_table] = {'start_line': i, 'content': [line], 'end_line': None}
        elif current_table and line.strip():
            table_sections[current_table]['content'].append(line)
    
    # Determine end line for each table
    tables = list(table_sections.keys())
    for i in range(len(tables)):
        if i < len(tables) - 1:
            table_sections[tables[i]]['end_line'] = table_sections[tables[i+1]]['start_line'] - 1
        else:
            table_sections[tables[i]]['end_line'] = len(lines) - 1
    
    return content, table_sections, lines

def read_sentences_from_csv(file_path):
    """Read sentences from the CSV file."""
    sentences = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 2:
                    number = row[0]
                    sentence = row[1]
                    sentences.append((number, sentence))
    except UnicodeDecodeError:
        # Try with different encodings if utf-8 fails
        with open(file_path, 'r', encoding='cp1252') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 2:
                    number = row[0]
                    sentence = row[1]
                    sentences.append((number, sentence))
    
    return sentences

def find_table_references(sentences):
    """Find all sentences that reference tables."""
    table_references = {}
    
    for number, sentence in sentences:
        # Look for "Table X" or "Tables X and Y" or similar patterns
        matches = re.finditer(r'Table\s+(\d+)', sentence, re.IGNORECASE)
        
        for match in matches:
            table_num = match.group(1)
            table_key = f"Table {table_num}"
            
            if table_key not in table_references:
                table_references[table_key] = []
            
            # Add the sentence if it's not already added
            if sentence not in [s for _, s in table_references[table_key]]:
                table_references[table_key].append((number, sentence))
    
    return table_references

def update_table_file(original_content, table_sections, table_references, output_file):
    """Update the table file with referenced sentences."""
    # Convert to list for easier manipulation
    lines = original_content.split('\n')
    
    # For each table, add the referenced sentences
    additions = 0  # Track how many lines we've added
    
    # Sort table sections by their original position in the file
    sorted_tables = sorted(table_sections.items(), key=lambda x: x[1]['start_line'])
    
    for i, (table_key, table_info) in enumerate(sorted_tables):
        if table_key in table_references:
            # Get the ending line number for this table (adjusted for any additions so far)
            end_line = table_info['end_line'] + additions
            
            # Add referenced sentences
            sentences_to_add = [f"\nSentences of {table_key}:"]
            for number, sentence in table_references[table_key]:
                sentences_to_add.append(f"{number}: {sentence}")
            
            # Add extra blank lines if this is not the last table
            if i < len(sorted_tables) - 1:
                sentences_to_add.extend(["", ""])  # Add two blank lines
            
            # Insert at the end of the table section
            lines[end_line+1:end_line+1] = sentences_to_add
            
            # Update the additions count
            additions += len(sentences_to_add)
    
    # Write back to file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write('\n'.join(lines))

def process_folder(folder_path):
    """Process the table.txt and text.csv files in a single folder."""
    table_file_path = os.path.join(folder_path, 'table.txt')
    csv_file_path = os.path.join(folder_path, 'text_split.csv')
    output_file_path = os.path.join(folder_path, 'table_with_references.txt')
    
    # Check if both required files exist
    if not os.path.exists(table_file_path):
        return False, "table.txt missing"
    if not os.path.exists(csv_file_path):
        return False, "text_split.csv missing"
    
    # Skip if output file already exists (uncomment if you want to skip existing)
    # if os.path.exists(output_file_path):
    #     return False, "output already exists"
    
    try:
        # Read the table file and identify table sections
        original_content, table_sections, lines = read_table_file(table_file_path)
        
        # Read sentences from CSV
        sentences = read_sentences_from_csv(csv_file_path)
        
        # Find all sentences that reference tables
        table_references = find_table_references(sentences)
        
        # If no table references found, we still consider it successful but no changes made
        if not table_references:
            return True, "no table references found"
        
        # Update the output file with the references
        update_table_file(original_content, table_sections, table_references, output_file_path)
        
        # Return success with table reference counts
        table_counts = {table: len(refs) for table, refs in table_references.items()}
        return True, table_counts
    
    except Exception as e:
        return False, f"error: {str(e)}"

def process_all_folders(base_path):
    """Process all folders from L0001 to L2621."""
    total_folders = 0
    processed_folders = 0
    skipped_folders = 0
    error_folders = 0
    
    for i in range(1, 3317):
        folder_name = f"L{i:04d}"
        folder_path = os.path.join(base_path, folder_name)
        
        total_folders += 1
        
        # Skip if folder doesn't exist
        if not os.path.isdir(folder_path):
            skipped_folders += 1
            continue
        
        # Process the folder
        success, result = process_folder(folder_path)
        
        if success:
            if isinstance(result, dict):
                # Successful processing with table references
                total_references = sum(result.values())
                processed_folders += 1
                print(f"Processed {folder_name}: Found {total_references} table references")
            else:
                # Successful but nothing changed (e.g., no table references)
                skipped_folders += 1
                print(f"Skipped {folder_name}: {result}")
        else:
            # Error or files missing
            skipped_folders += 1
            print(f"Skipped {folder_name}: {result}")
        
        # Progress update every 100 folders
        if i % 100 == 0:
            print(f"Progress: {i}/2621 folders processed. Success: {processed_folders}, Skipped: {skipped_folders}")
    
    # Print final summary
    print("\nProcessing complete!")
    print(f"Total folders checked: {total_folders}")
    print(f"Folders successfully processed: {processed_folders}")
    print(f"Folders skipped: {skipped_folders}")

def main():
    base_path = r".\HEA_finetune\Yield_Strength"
    print(f"Starting to process folders in {base_path}...")
    process_all_folders(base_path)

if __name__ == "__main__":
    main()