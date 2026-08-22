import os
import re
import time
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm


def initialize_llm_client():
    """Initializes and returns the OpenAI-compatible client."""
    try:
        client = OpenAI(
            api_key=os.getenv('DASHSCOPE_API_KEY'),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        return client
    except Exception as e:
        print(f"Fatal Error: Could not initialize the LLM client: {e}")
        exit(1)


def read_file_content(file_path):
    """Reads and returns the content of a given file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
            return f.read().strip()
    except Exception as e:
        print(f"    ✗ Error reading {file_path}: {e}")
        return None


def extract_figure_caption(text_content, fig_number):
    """
    Extracts the caption for a specific figure from text.txt.
    
    Args:
        text_content: Content of text.txt file
        fig_number: Figure number (e.g., 2, 10)
    
    Returns:
        Caption text or None if not found
    """
    try:
        lines = text_content.split('\n')
        
        # Find the line with "==== FIGURES ===="
        figures_start = -1
        for i, line in enumerate(lines):
            if "==== FIGURES ====" in line:
                figures_start = i
                break
        
        if figures_start == -1:
            return None
        
        # Search from "==== FIGURES ====" onwards
        search_lines = lines[figures_start:]
        
        # Find the start line for the current figure
        current_fig_pattern = f"Fig. {fig_number}:"
        start_idx = -1
        for i, line in enumerate(search_lines):
            if current_fig_pattern in line:
                start_idx = i
                break
        
        if start_idx == -1:
            return None
        
        # Find the end line (next figure or end of text)
        next_fig_pattern = f"Fig. {fig_number + 1}:"
        end_idx = len(search_lines)
        for i in range(start_idx + 1, len(search_lines)):
            if next_fig_pattern in search_lines[i]:
                end_idx = i
                break
        
        # Extract caption
        caption_lines = search_lines[start_idx:end_idx]
        caption = '\n'.join(caption_lines).strip()
        
        return caption
    
    except Exception as e:
        print(f"    ✗ Error extracting caption for Fig. {fig_number}: {e}")
        return None


def find_all_fig_out_files(l_folder):
    """
    Finds all files starting with 'Fig' and ending with '_out.csv' in the L folder.
    Returns a list of Path objects sorted by figure number.
    Example: Fig2_out.csv, Fig10_out.csv
    """
    fig_files = []
    
    for file in l_folder.iterdir():
        if file.is_file() and file.name.startswith('Fig') and file.name.endswith('_out.csv'):
            fig_files.append(file)
    
    # Sort by the numeric part of the filename
    def extract_fig_number(path):
        try:
            # Extract number from filename like "Fig2_out.csv" -> 2
            name_without_ext = path.stem  # "Fig2_out"
            # Remove "Fig" prefix and "_out" suffix
            number_str = name_without_ext[3:].replace('_out', '')  # "2"
            return int(number_str)
        except (ValueError, IndexError):
            return 0
    
    fig_files.sort(key=extract_fig_number)
    return fig_files


def process_fig_with_llm(l_folder, fig_out_file, text_content, prompt_text, client):
    """
    Processes a single Fig*_out.csv file with corresponding Fig*_material.txt and caption.
    
    Args:
        l_folder: Path to the L folder
        fig_out_file: Path to the Fig*_out.csv file
        text_content: Content of text.txt file
        prompt_text: The prompt template content
        client: OpenAI client instance
    
    Returns:
        Status string
    """
    try:
        # Extract figure number and name
        # From "Fig2_out.csv" get figure number 2
        fig_stem = fig_out_file.stem  # "Fig2_out"
        fig_number_str = fig_stem[3:].replace('_out', '')  # "2"
        fig_number = int(fig_number_str)
        
        # Define paths
        fig_material_file = l_folder / f"Fig{fig_number}_material.txt"
        output_filename = f"Fig{fig_number}_out_step2.csv"
        output_path = l_folder / output_filename
        
        # Check if output file already exists
        if output_path.exists():
            return f"Skipped (output file {output_filename} already exists)"
        
        # Check if Fig*_material.txt exists
        if not fig_material_file.exists():
            return f"Skipped (Fig{fig_number}_material.txt not found)"
        
        # Read input files
        fig_out_content = read_file_content(fig_out_file)
        fig_material_content = read_file_content(fig_material_file)
        
        if fig_out_content is None or fig_material_content is None:
            return "Error (could not read input files)"
        
        # Extract figure caption from text.txt
        caption = extract_figure_caption(text_content, fig_number)
        if caption is None:
            caption = f"(Caption for Fig. {fig_number} not found in text.txt)"
        
        # Construct prompt
        start_time = time.time()
        
        system_prompt = "You are a materials science data organization specialist. Your task is to organize and structure material data according to the given instructions."
        
        # Note: Files are renamed for LLM
        # Fig2_out.csv -> Fig_out.csv
        # Fig2_material.txt -> Fig_material.txt
        user_prompt = (
            f"Figure Caption:\n{caption}\n\n"
            f"Content from Fig_out.csv:\n{fig_out_content}\n\n"
            f"Content from Fig_material.txt:\n{fig_material_content}\n\n"
            f"Based on the provided data, please process it according to the following instructions:\n{prompt_text}"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = client.chat.completions.create(
            model="qwen-max-latest",
            messages=messages,
            temperature=0.0,
            stream=False
        )
        
        output_content = response.choices[0].message.content.strip()
        
        # Save the result
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_content)
        
        process_time = time.time() - start_time
        token_usage = response.usage.total_tokens if hasattr(response, 'usage') else "N/A"
        
        return f"Success | Time: {process_time:.2f}s | Tokens: {token_usage}"
    
    except Exception as e:
        return f"Error ({e})"


def process_l_folder(l_folder, prompt_text, client):
    """
    Processes a single L folder:
    1. Find all Fig*_out.csv files
    2. Read text.txt for figure captions
    3. Process each Fig*_out.csv with corresponding material file and caption
    
    Returns a tuple: (processed_count, skipped_count, error_count)
    """
    processed = 0
    skipped = 0
    errors = 0
    
    # Step 1: Find all Fig*_out.csv files
    fig_out_files = find_all_fig_out_files(l_folder)
    
    if not fig_out_files:
        return (0, 1, 0)  # No fig_out files, count as skipped
    
    print(f"\n[{l_folder.name}] Found {len(fig_out_files)} Fig*_out.csv file(s): {[f.name for f in fig_out_files]}")
    
    # Step 2: Read text.txt
    text_file = l_folder / "text.txt"
    if not text_file.exists():
        print(f"  ✗ text.txt not found - cannot extract captions")
        return (0, len(fig_out_files), 0)
    
    text_content = read_file_content(text_file)
    if text_content is None:
        print(f"  ✗ Could not read text.txt")
        return (0, len(fig_out_files), 0)
    
    # Step 3: Process each Fig*_out.csv file
    for fig_out_file in fig_out_files:
        print(f"  Processing: {fig_out_file.name}")
        
        status = process_fig_with_llm(
            l_folder, fig_out_file, text_content, prompt_text, client
        )
        
        if "Success" in status:
            print(f"    ✓ {status}")
            processed += 1
        elif "Skipped" in status:
            print(f"    ⊘ {status}")
            skipped += 1
        else:  # Error
            print(f"    ✗ {status}")
            errors += 1
    
    return (processed, skipped, errors)


def main_process_loop(base_path, prompt_path):
    """
    Main loop to iterate through L0001-L2594 folders and organize labels.
    """
    client = initialize_llm_client()
    
    # Read the prompt template
    prompt_text = read_file_content(prompt_path)
    if prompt_text is None:
        print("Aborting: Could not read the main prompt file.")
        return
    
    print("=" * 80)
    print("Starting batch LLM processing for organizing labels")
    print("=" * 80)
    
    total_processed = 0
    total_skipped = 0
    total_errors = 0
    
    # Iterate through L0001 to L2594
    for i in tqdm(range(1, 3354), desc="Processing L folders"):
        folder_name = f"L{i:04d}"
        l_folder = Path(base_path) / folder_name
        
        # Check if L folder exists
        if not l_folder.exists() or not l_folder.is_dir():
            continue
        
        # Process the L folder
        processed, skipped, errors = process_l_folder(l_folder, prompt_text, client)
        
        total_processed += processed
        total_skipped += skipped
        total_errors += errors
    
    print("\n" + "=" * 80)
    print("Processing complete!")
    print(f"  Total Successful: {total_processed}")
    print(f"  Total Skipped:    {total_skipped}")
    print(f"  Total Errors:     {total_errors}")
    print("=" * 80)


if __name__ == "__main__":
    # Define the main paths for the operation
    base_directory = Path.cwd()  # Current working directory
    prompt_file_path = r".\stress-strain\prompt\organize_label.md"
    
    main_process_loop(base_directory, prompt_file_path)