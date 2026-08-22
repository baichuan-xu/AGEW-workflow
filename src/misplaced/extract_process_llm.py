import os
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


def find_out_filter_csv(ys_folder):
    """
    Recursively searches for out_filter.csv in YS folder and its subfolders.
    Returns the path if found, None otherwise.
    """
    for root, dirs, files in os.walk(ys_folder):
        if "out_filter.csv" in files:
            return Path(root) / "out_filter.csv"
    return None


def find_all_fig_material_files(l_folder):
    """
    Finds all files starting with 'Fig' and ending with '_material.txt' in the L folder.
    Returns a list of Path objects sorted by figure number.
    Example: Fig5_material.txt, Fig8_material.txt, Fig12_material.txt
    """
    fig_files = []
    
    for file in l_folder.iterdir():
        if file.is_file() and file.name.startswith('Fig') and file.name.endswith('_material.txt'):
            fig_files.append(file)
    
    # Sort by the numeric part of the filename
    def extract_fig_number(path):
        try:
            # Extract number from filename like "Fig5_material.txt" -> 5
            name_without_ext = path.stem  # "Fig5_material"
            # Remove "Fig" prefix and "_material" suffix
            number_str = name_without_ext[3:].replace('_material', '')  # "5"
            return int(number_str)
        except (ValueError, IndexError):
            return 0
    
    fig_files.sort(key=extract_fig_number)
    return fig_files


def process_fig_file_with_llm(l_folder, fig_file, sentence_split_path, prompt_text, client):
    """
    Processes a single Fig*_material.txt file with sentence_split.csv using LLM.
    
    Args:
        l_folder: Path to the L folder
        fig_file: Path to the Fig*_material.txt file
        sentence_split_path: Path to sentence_split.csv
        prompt_text: The prompt template content
        client: OpenAI client instance
    
    Returns:
        Status string
    """
    try:
        # Extract figure number and name
        # From "Fig5_material.txt" get "Fig5"
        fig_stem = fig_file.stem  # "Fig5_material"
        fig_number_part = fig_stem.replace('_material', '')  # "Fig5"
        output_filename = f"{fig_number_part}_process.csv"
        output_path = l_folder / output_filename
        
        # ✓ ALREADY IMPLEMENTED: Skip if output file already exists
        if output_path.exists():
            return f"Skipped (output file {output_filename} already exists)"
        
        # Read input files
        fig_content = read_file_content(fig_file)
        sentence_split_content = read_file_content(sentence_split_path)
        
        if fig_content is None or sentence_split_content is None:
            return "Error (could not read input files)"
        
        # Construct prompt and call LLM
        start_time = time.time()
        
        system_prompt = "You are a materials science data processing specialist. Your task is to analyze figure descriptions and sentence data to extract and process material information according to the given instructions."
        
        # Note: The fig file content is labeled as "Fig.txt" for the LLM
        user_prompt = (
            f"Content from Fig.txt:\n{fig_content}\n\n"
            f"Content from sentence_split.csv:\n{sentence_split_content}\n\n"
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
    1. Check if any YS subfolder contains out_filter.csv
    2. Find all Fig*.txt files in the L folder
    3. Process each Fig*.txt with sentence_split.csv
    
    Returns a tuple: (processed_count, skipped_count, error_count)
    """
    processed = 0
    skipped = 0
    errors = 0
    
    # Step 1: Find YS subfolders and check for out_filter.csv
    ys_folders = [f for f in l_folder.iterdir() 
                  if f.is_dir() and f.name.startswith('YS')]
    
    if not ys_folders:
        return (0, 1, 0)  # No YS folders, count as skipped
    
    # Check if any YS folder contains out_filter.csv
    has_out_filter = False
    for ys_folder in ys_folders:
        if find_out_filter_csv(ys_folder) is not None:
            has_out_filter = True
            break
    
    if not has_out_filter:
        return (0, 1, 0)  # No out_filter.csv found, skip
    
    print(f"\n[{l_folder.name}] Found {len(ys_folders)} YS folder(s) with out_filter.csv")
    
    # Step 2: Find sentence_split.csv in L folder
    sentence_split_path = l_folder / "sentence_split.csv"
    if not sentence_split_path.exists():
        print(f"  ⊘ Skipped: sentence_split.csv not found")
        return (0, 1, 0)
    
    # Step 3: Find all Fig*_material.txt files
    fig_files = find_all_fig_material_files(l_folder)
    
    if not fig_files:
        print(f"  ⊘ Skipped: No Fig*_material.txt files found")
        return (0, 1, 0)
    
    print(f"  Found {len(fig_files)} Fig*_material.txt file(s): {[f.name for f in fig_files]}")
    
    # Step 4: Process each Fig*_material.txt file
    for fig_file in fig_files:
        print(f"  Processing: {fig_file.name}")
        
        status = process_fig_file_with_llm(
            l_folder, fig_file, sentence_split_path, prompt_text, client
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
    Main loop to iterate through L0001-L2594 folders and process Fig files.
    """
    client = initialize_llm_client()
    
    # Read the prompt template
    prompt_text = read_file_content(prompt_path)
    if prompt_text is None:
        print("Aborting: Could not read the main prompt file.")
        return
    
    print("=" * 80)
    print("Starting batch LLM processing for Fig*_material.txt files with sentence_split.csv")
    print("=" * 80)
    
    total_processed = 0
    total_skipped = 0
    total_errors = 0
    
    # Iterate through L0001 to L2594
    for i in tqdm(range(1, 2595), desc="Processing L folders"):
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
    prompt_file_path = r".\stress-strain\prompt\material_process.md"
    
    main_process_loop(base_directory, prompt_file_path)