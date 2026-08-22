import os
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Set
import re


def has_fig_process_csv(folder: Path) -> bool:
    """
    Check if folder contains any file matching pattern Fig*_process.csv
    Returns True if found, False otherwise.
    """
    pattern = re.compile(r'^Fig\d+_process\.csv$', re.IGNORECASE)
    
    for file in folder.iterdir():
        if file.is_file() and pattern.match(file.name):
            return True
    
    return False


def parse_tables_from_file(table_file_path: Path) -> List[Tuple[int, List[str]]]:
    """
    Parse table.txt and extract individual tables.
    Returns list of tuples: [(table_number, [lines]), ...]
    """
    try:
        with open(table_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        tables = []
        current_table_num = None
        current_table_start = None
        
        for i, line in enumerate(lines):
            # Check if line contains "Table X:"
            match = re.search(r'Table\s+(\d+)\s*:', line, re.IGNORECASE)
            
            if match:
                # Save previous table if exists
                if current_table_num is not None and current_table_start is not None:
                    table_lines = lines[current_table_start:i]
                    tables.append((current_table_num, table_lines))
                
                # Start new table
                current_table_num = int(match.group(1))
                current_table_start = i
        
        # Add the last table
        if current_table_num is not None and current_table_start is not None:
            table_lines = lines[current_table_start:]
            tables.append((current_table_num, table_lines))
        
        return tables
    
    except Exception as e:
        print(f"    ✗ Error reading table.txt: {str(e)}")
        return []


def is_composition_table(table_lines: List[str]) -> bool:
    """
    Check if table contains composition-related keywords.
    Keywords: element, elemental, wt%, content, contents, composition, compositions
    Case-insensitive.
    """
    keywords = ['element', 'elemental', 'wt%', 'content', 'contents', 'composition', 'compositions']
    
    table_text = ''.join(table_lines).lower()
    
    for keyword in keywords:
        if keyword.lower() in table_text:
            return True
    
    return False


def find_sentences_with_table_reference(sentence_split_path: Path, table_num: int) -> List[int]:
    """
    Find all row indices containing reference to "Table X" (case-insensitive).
    Returns list of row indices.
    """
    try:
        df = pd.read_csv(sentence_split_path)
        
        if 'sentence' not in df.columns:
            print(f"    ⚠ 'sentence' column not found in sentence_split.csv")
            return []
        
        pattern = re.compile(rf'\btable\s+{table_num}\b', re.IGNORECASE)
        
        indices = []
        for idx, row in df.iterrows():
            sentence = str(row['sentence'])
            if pattern.search(sentence):
                indices.append(idx)
        
        return indices
    
    except Exception as e:
        print(f"    ✗ Error reading sentence_split.csv: {str(e)}")
        return []


def extract_context_sentences(sentence_split_path: Path, center_indices: List[int], 
                              window: int = 3) -> List[str]:
    """
    Extract sentences around each center index (3 before and 3 after).
    Remove duplicates while preserving order.
    Returns list of unique sentences.
    """
    try:
        df = pd.read_csv(sentence_split_path)
        
        sentences = []
        
        for center_idx in center_indices:
            # Calculate range (3 before and 3 after)
            start_idx = max(0, center_idx - window)
            end_idx = min(len(df), center_idx + window + 1)
            
            # Extract sentences in range
            for idx in range(start_idx, end_idx):
                sentence = str(df.iloc[idx]['sentence']).strip()
                sentences.append(sentence)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_sentences = []
        for sentence in sentences:
            if sentence not in seen and sentence:
                seen.add(sentence)
                unique_sentences.append(sentence)
        
        return unique_sentences
    
    except Exception as e:
        print(f"    ✗ Error extracting context: {str(e)}")
        return []


def save_composition_tables(l_folder: Path, selected_tables: List[Tuple[int, List[str]]], 
                           sentence_split_path: Path) -> bool:
    """
    Save selected composition tables with descriptions to table_composition.txt.
    """
    try:
        output_path = l_folder / 'table_composition.txt'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, (table_num, table_lines) in enumerate(selected_tables):
                # Write table content
                f.write(''.join(table_lines))
                
                # Add description section
                f.write('\nDescription of Table:\n')
                
                # Find sentences referencing this table
                table_ref_indices = find_sentences_with_table_reference(sentence_split_path, table_num)
                
                if table_ref_indices:
                    # Extract context sentences (3 before and 3 after)
                    context_sentences = extract_context_sentences(sentence_split_path, table_ref_indices, window=3)
                    
                    # Write sentences
                    for sentence in context_sentences:
                        f.write(sentence + '\n')
                    
                    print(f"      Table {table_num}: {len(context_sentences)} context sentences")
                else:
                    f.write("(No description found in sentence_split.csv)\n")
                    print(f"      Table {table_num}: No reference found")
                
                # Add separator between tables (except for last one)
                if i < len(selected_tables) - 1:
                    f.write('\n' + '=' * 80 + '\n\n')
        
        print(f"    ✓ Saved: {output_path.name} ({len(selected_tables)} tables)")
        return True
    
    except Exception as e:
        print(f"    ✗ Error saving table_composition.txt: {str(e)}")
        return False


def process_l_folder(l_folder: Path) -> bool:
    """
    Process a single L#### folder:
    - Parse table.txt
    - Select composition tables
    - Extract context from sentence_split.csv
    - Save to table_composition.txt
    """
    # Check for required files
    table_file = l_folder / 'table.txt'
    sentence_split_file = l_folder / 'sentence_split.csv'
    
    if not table_file.exists():
        print(f"  ⚠ table.txt not found")
        return False
    
    if not sentence_split_file.exists():
        print(f"  ⚠ sentence_split.csv not found")
        return False
    
    print(f"  Processing table.txt")
    
    # Parse tables
    tables = parse_tables_from_file(table_file)
    
    if not tables:
        print(f"  ⚠ No tables found in table.txt")
        return False
    
    print(f"  Found {len(tables)} table(s)")
    
    # Filter composition tables
    selected_tables = []
    for table_num, table_lines in tables:
        if is_composition_table(table_lines):
            selected_tables.append((table_num, table_lines))
            print(f"    ✓ Table {table_num} selected (composition-related)")
        else:
            print(f"    ⊘ Table {table_num} skipped (not composition-related)")
    
    if not selected_tables:
        print(f"  ⚠ No composition tables found")
        return False
    
    print(f"  Selected {len(selected_tables)} composition table(s)")
    
    # Save selected tables with descriptions
    return save_composition_tables(l_folder, selected_tables, sentence_split_file)


def main():
    """
    Main function to process all L#### folders.
    """
    current_path = Path.cwd()
    
    print("=" * 80)
    print("Starting composition table extraction")
    print("=" * 80)
    
    processed_count = 0
    success_count = 0
    
    # Loop through L0001 to L2594
    for i in range(1, 3356):
        folder_name = f"L{i:04d}"
        folder_path = current_path / folder_name
        
        if not folder_path.exists() or not folder_path.is_dir():
            continue
        
        # Check for Fig*_process.csv files
        if not has_fig_process_csv(folder_path):
            continue
        
        print(f"\n[{folder_name}]")
        
        processed_count += 1
        
        if process_l_folder(folder_path):
            success_count += 1
    
    print("\n" + "=" * 80)
    print(f"Processing complete!")
    print(f"  Total L#### folders with Fig*_process.csv: {processed_count}")
    print(f"  Successfully created table_composition.txt: {success_count}")
    print(f"  Failed: {processed_count - success_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()