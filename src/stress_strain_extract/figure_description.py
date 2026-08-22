import os
import pandas as pd
from pathlib import Path
from typing import List, Optional, Set
import re


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


def extract_figure_caption_from_text(text_file_path: Path, fig_number: int) -> Optional[str]:
    """
    Extract the sentence in the next row after "Fig. X" from text.txt.
    Searches from "==== FIGURES ====" line to the end.
    Returns the caption sentence or None if not found.
    """
    try:
        with open(text_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Find "==== FIGURES ====" line
        figures_start_idx = None
        for i, line in enumerate(lines):
            if "==== FIGURES ====" in line:
                figures_start_idx = i
                break
        
        if figures_start_idx is None:
            print(f"    ⚠ Could not find '==== FIGURES ====' in text.txt")
            return None
        
        # Search for "Fig. X" from figures section onwards
        fig_pattern = f"Fig. {fig_number}"
        
        for i in range(figures_start_idx, len(lines)):
            if fig_pattern in lines[i]:
                # Extract the next line as caption
                if i + 1 < len(lines):
                    caption = lines[i + 1].strip()
                    return caption
                else:
                    print(f"    ⚠ No line after '{fig_pattern}' in text.txt")
                    return None
        
        print(f"    ⚠ Could not find '{fig_pattern}' in figures section")
        return None
    
    except Exception as e:
        print(f"    ✗ Error reading text.txt: {str(e)}")
        return None


def find_sentence_with_figure(sentence_split_path: Path, fig_number: int) -> Optional[int]:
    """
    Find the row index (0-based) of the sentence containing "Fig. X" in sentence_split.csv.
    Returns the row index or None if not found.
    """
    try:
        df = pd.read_csv(sentence_split_path)
        
        if 'sentence' not in df.columns:
            print(f"    ⚠ 'sentence' column not found in sentence_split.csv")
            return None
        
        fig_pattern = f"Fig. {fig_number}"
        
        for idx, row in df.iterrows():
            sentence = str(row['sentence'])
            if fig_pattern in sentence:
                return idx
        
        print(f"    ⚠ Could not find '{fig_pattern}' in sentence_split.csv")
        return None
    
    except Exception as e:
        print(f"    ✗ Error reading sentence_split.csv: {str(e)}")
        return None


def extract_context_sentences(sentence_split_path: Path, center_idx: int, 
                              caption: str, window: int = 5) -> List[str]:
    """
    Extract sentences around the center index (5 before and 5 after),
    plus the caption sentence.
    Remove duplicate sentences.
    Returns list of unique sentences in order.
    """
    try:
        df = pd.read_csv(sentence_split_path)
        
        # Calculate range (5 before and 5 after)
        start_idx = max(0, center_idx - window)
        end_idx = min(len(df), center_idx + window + 1)
        
        # Extract sentences in range
        sentences = []
        for idx in range(start_idx, end_idx):
            sentence = str(df.iloc[idx]['sentence']).strip()
            sentences.append(sentence)
        
        # Add caption sentence
        if caption:
            sentences.append(caption.strip())
        
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


def save_sentences_to_file(sentences: List[str], output_path: Path) -> bool:
    """
    Save sentences to a text file, one sentence per line.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                f.write(sentence + '\n')
        
        print(f"    ✓ Saved: {output_path.name} ({len(sentences)} sentences)")
        return True
    
    except Exception as e:
        print(f"    ✗ Error saving file: {str(e)}")
        return False


def extract_ys_number(ys_folder_name: str) -> Optional[int]:
    """
    Extract number from YS folder name.
    Example: "YS2" -> 2, "YS008" -> 8, "YS123" -> 123
    """
    match = re.search(r'YS(\d+)', ys_folder_name)
    if match:
        return int(match.group(1))
    return None


def process_ys_folder(l_folder: Path, ys_folder: Path) -> bool:
    """
    Process a single YS folder:
    - Extract YS number
    - Find figure caption from text.txt
    - Find sentence in sentence_split.csv
    - Extract context sentences
    - Save to FigX.txt in L folder
    """
    ys_number = extract_ys_number(ys_folder.name)
    
    if ys_number is None:
        print(f"    ⚠ Could not extract number from {ys_folder.name}")
        return False
    
    print(f"    Processing YS{ys_number} -> Fig. {ys_number}")
    
    # Check for required files in L folder
    text_file = l_folder / 'text.txt'
    sentence_split_file = l_folder / 'sentence_split.csv'
    
    if not text_file.exists():
        print(f"    ⚠ text.txt not found in {l_folder.name}")
        return False
    
    if not sentence_split_file.exists():
        print(f"    ⚠ sentence_split.csv not found in {l_folder.name}")
        return False
    
    # Step 1: Extract caption from text.txt
    caption = extract_figure_caption_from_text(text_file, ys_number)
    if caption is None:
        return False
    
    print(f"      Caption: {caption[:60]}..." if len(caption) > 60 else f"      Caption: {caption}")
    
    # Step 2: Find sentence containing "Fig. X" in sentence_split.csv
    center_idx = find_sentence_with_figure(sentence_split_file, ys_number)
    if center_idx is None:
        return False
    
    print(f"      Found at row index: {center_idx}")
    
    # Step 3: Extract 5 sentences before and after, plus caption
    sentences = extract_context_sentences(sentence_split_file, center_idx, caption, window=5)
    
    if not sentences:
        print(f"    ⚠ No sentences extracted")
        return False
    
    # Step 4: Save to FigX.txt in L folder
    output_file = l_folder / f"Fig{ys_number}.txt"
    return save_sentences_to_file(sentences, output_file)


def process_l_folder(l_folder: Path) -> bool:
    """
    Process a single L#### folder:
    - Find all YS* folders with out_filter.csv
    - Process each one
    """
    # Find all YS* subfolders
    ys_folders = sorted([f for f in l_folder.iterdir() 
                        if f.is_dir() and f.name.startswith('YS')])
    
    if not ys_folders:
        return False
    
    # Filter YS folders that have out_filter.csv
    valid_ys_folders = []
    for ys_folder in ys_folders:
        if has_out_filter_in_subtree(ys_folder):
            valid_ys_folders.append(ys_folder)
    
    if not valid_ys_folders:
        return False
    
    print(f"  Found {len(valid_ys_folders)} YS* folder(s) with out_filter.csv")
    
    success_count = 0
    for ys_folder in valid_ys_folders:
        if process_ys_folder(l_folder, ys_folder):
            success_count += 1
    
    return success_count > 0


def main():
    """
    Main function to process all L#### folders.
    """
    current_path = Path.cwd()
    
    print("=" * 80)
    print("Starting figure context extraction")
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
    print(f"  Total L#### folders processed: {processed_count}")
    print(f"  Successfully extracted figures: {success_count}")
    print(f"  Failed: {processed_count - success_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()