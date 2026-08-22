import os
import pandas as pd
from pathlib import Path
from typing import List, Optional
import re


def extract_text_content(text_file_path: Path) -> Optional[str]:
    """
    Extract text content between "TEXT CONTENT" and "==== FIGURES ====".
    Returns the extracted content or None if markers not found.
    """
    try:
        with open(text_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Find line indices for markers
        start_idx = None
        end_idx = None
        
        for i, line in enumerate(lines):
            if "TEXT CONTENT" in line:
                start_idx = i
            if "==== FIGURES ====" in line:
                end_idx = i
                break
        
        if start_idx is None or end_idx is None:
            print(f"    ⚠ Could not find markers in {text_file_path}")
            return None
        
        if start_idx >= end_idx:
            print(f"    ⚠ Invalid marker positions in {text_file_path}")
            return None
        
        # Extract content between markers (excluding the marker lines)
        content_lines = lines[start_idx + 1:end_idx]
        content = ''.join(content_lines)
        
        return content
    
    except Exception as e:
        print(f"    ✗ Error reading text.txt: {str(e)}")
        return None


def split_into_sentences(content: str) -> List[str]:
    """
    Split content into sentences by periods with smart rules:
    - Find the first LETTER of the first word after the period (skip numbers, digits, punctuation, whitespace)
    - Split when that first letter is UPPERCASE
    - Do NOT split when that first letter is lowercase
    Returns a list of sentences (cleaned and non-empty).
    """
    sentences = []
    current_sentence = ""
    
    i = 0
    while i < len(content):
        char = content[i]
        current_sentence += char
        
        # Check if current character is a period
        if char == '.':
            # Look ahead to find the first LETTER after the period
            j = i + 1
            
            # Skip whitespace, digits, and punctuation to find the first letter
            while j < len(content) and not content[j].isalpha():
                j += 1
            
            # Check if we found a letter
            if j < len(content):
                # Check if the first letter is uppercase
                if content[j].isupper():
                    # Uppercase letter - split here
                    sentence = current_sentence.strip()
                    if sentence:
                        sentences.append(sentence)
                    current_sentence = ""
                # If lowercase, continue with same sentence (don't split)
            else:
                # No letter found after period (end of text)
                # Treat as end of sentence
                sentence = current_sentence.strip()
                if sentence:
                    sentences.append(sentence)
                current_sentence = ""
        
        i += 1
    
    # Add any remaining content as the last sentence
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    
    return sentences


def save_sentences_to_csv(sentences: List[str], output_path: Path) -> bool:
    """
    Save sentences to sentence_split.csv with serial numbers.
    """
    try:
        # Create dataframe with serial numbers (starting from 1)
        df = pd.DataFrame({
            'serial_number': range(1, len(sentences) + 1),
            'sentence': sentences
        })
        
        df.to_csv(output_path, index=False)
        
        print(f"    ✓ Saved: {output_path.name} ({len(sentences)} sentences)")
        return True
    
    except Exception as e:
        print(f"    ✗ Error saving CSV: {str(e)}")
        return False


def has_out_filter_in_subtree(ys_folder: Path) -> bool:
    """
    Check if out_filter.csv exists anywhere in YS folder or its subfolders.
    Returns True if found, False otherwise.
    """
    # Check in YS folder itself
    if (ys_folder / 'out_filter.csv').exists():
        return True
    
    # Check in all subfolders recursively
    for root, dirs, files in os.walk(ys_folder):
        if 'out_filter.csv' in files:
            return True
    
    return False


def process_l_folder(l_folder: Path) -> bool:
    """
    Process a single L#### folder:
    - Check for YS* folders with out_filter.csv
    - Extract text from text.txt
    - Split into sentences
    - Save to sentence_split.csv
    """
    # Find all YS* subfolders
    ys_folders = sorted([f for f in l_folder.iterdir() 
                        if f.is_dir() and f.name.startswith('YS')])
    
    if not ys_folders:
        return False
    
    # Check if any YS* folder contains out_filter.csv (anywhere in subtree)
    has_out_filter = False
    for ys_folder in ys_folders:
        if has_out_filter_in_subtree(ys_folder):
            has_out_filter = True
            break
    
    if not has_out_filter:
        return False
    
    print(f"  Found YS* folder(s) with out_filter.csv")
    
    # Look for text.txt in L#### folder
    text_file = l_folder / 'text.txt'
    
    if not text_file.exists():
        print(f"  ⚠ No text.txt found in {l_folder.name}")
        return False
    
    print(f"  Processing text.txt")
    
    # Extract content between markers
    content = extract_text_content(text_file)
    
    if content is None:
        return False
    
    # Split into sentences
    sentences = split_into_sentences(content)
    
    if not sentences:
        print(f"    ⚠ No sentences found after splitting")
        return False
    
    # Save to CSV
    output_path = l_folder / 'sentence_split.csv'
    return save_sentences_to_csv(sentences, output_path)


def main():
    """
    Main function to process all L#### folders.
    """
    current_path = Path.cwd()
    
    print("=" * 80)
    print("Starting sentence splitting from text.txt files")
    print("=" * 80)
    
    processed_count = 0
    success_count = 0
    
    # Loop through L0001 to L2594
    for i in range(1, 3356):
        folder_name = f"L{i:04d}"
        folder_path = current_path / folder_name
        
        if not folder_path.exists() or not folder_path.is_dir():
            continue
        
        print(f"\n[{folder_name}]")
        
        processed_count += 1
        
        if process_l_folder(folder_path):
            success_count += 1
    
    print("\n" + "=" * 80)
    print(f"Processing complete!")
    print(f"  Total L#### folders with YS* and out_filter.csv: {processed_count}")
    print(f"  Successfully created sentence_split.csv: {success_count}")
    print(f"  Failed: {processed_count - success_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()