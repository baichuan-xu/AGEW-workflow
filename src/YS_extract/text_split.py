import re
import csv
import os
import sys
from pathlib import Path

def extract_text_content(filename):
    """Extract text content between markers from the file."""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
        
        pattern = r"======================================== TEXT CONTENT.*?\n(.*?)(?:======================================== FIGURES|$)"
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        else:
            return ""
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return ""

def split_into_sentences(text):
    """Split text into sentences with improved handling."""
    # Normalize line breaks
    text = re.sub(r'\r\n|\r', '\n', text)
    
    # Replace newlines with spaces to keep sentences intact
    text = re.sub(r'\n', ' ', text)
    
    # Temporarily mark common abbreviations to protect them
    protected_text = text
    common_abbrs = [
        "Fig.", "et al.", "i.e.", "e.g.", "etc.", "vs.",
        "Dr.", "Mr.", "Mrs.", "Prof.", "Inc.", "Ltd."
    ]
    
    for abbr in common_abbrs:
        protected_text = protected_text.replace(abbr, abbr.replace(".", "PERIOD_MARKER"))
    
    # Also protect reference citations
    protected_text = re.sub(r'\[\d+\]\.', lambda m: m.group(0).replace(".", "PERIOD_MARKER"), protected_text)
    
    # Split sentences at period, question mark, or exclamation followed by space and capital letter
    sentences = []
    for s in re.split(r'[.!?]\s+(?=[A-Z])', protected_text):
        # Restore the periods in abbreviations
        restored = s.replace("PERIOD_MARKER", ".")
        # If this doesn't end with punctuation, add a period
        if not restored.rstrip().endswith(('.', '!', '?')):
            restored = restored.rstrip() + "."
        sentences.append(restored)
    
    # Clean up each sentence
    clean_sentences = []
    for sentence in sentences:
        # Remove surrounding quotes
        if sentence.startswith('"') and sentence.endswith('"'):
            sentence = sentence[1:-1].strip()
        # Remove any extra whitespace
        sentence = ' '.join(sentence.split())
        if sentence:
            clean_sentences.append(sentence)
    
    return clean_sentences

def create_csv(sentences, output_filename):
    """Create a CSV file with numbered sentences."""
    try:
        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write sentences with sequential numbering
            for i, sentence in enumerate(sentences, 1):
                writer.writerow([i, sentence])
        
        return True, len(sentences)
    except Exception as e:
        print(f"Error creating CSV {output_filename}: {e}")
        return False, 0

def process_folder(folder_path):
    """Process a single folder."""
    input_file = os.path.join(folder_path, "text.txt")
    output_file = os.path.join(folder_path, "text_split.csv")
    
    # Check if text.txt exists
    if not os.path.exists(input_file):
        print(f"Warning: {input_file} does not exist. Skipping folder.")
        return False
    
    # Extract content
    text_content = extract_text_content(input_file)
    if not text_content:
        print(f"Warning: No content extracted from {input_file}. Skipping folder.")
        return False
    
    # Process content
    sentences = split_into_sentences(text_content)
    if not sentences:
        print(f"Warning: No sentences extracted from {input_file}. Skipping folder.")
        return False
    
    # Create CSV
    success, count = create_csv(sentences, output_file)
    if success:
        print(f"Successfully processed {input_file} -> {output_file} with {count} sentences.")
        return True
    else:
        return False

def process_all_folders():
    """Process all L0001 to L2621 folders in the specified base directory."""
    # Use the specific base directory
    base_dir = r".\HEA_finetune\Yield_Strength"
    
    # Count successful and failed folders
    successful = 0
    failed = 0
    
    # Process each folder from L0001 to L2621
    for i in range(1, 3317):
        folder_name = f"L{i:04d}"  # Format with leading zeros: L0001, L0002, etc.
        folder_path = os.path.join(base_dir, folder_name)
        
        # Check if the folder exists
        if not os.path.isdir(folder_path):
            print(f"Warning: Folder {folder_path} does not exist. Skipping.")
            failed += 1
            continue
        
        # Process the folder
        print(f"Processing folder {folder_name} ({i}/2621)...")
        if process_folder(folder_path):
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful} folders")
    print(f"Failed to process: {failed} folders")

if __name__ == "__main__":
    print("Starting to process folders under the configured base path...")
    process_all_folders()
