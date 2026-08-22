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


def find_fig_txt_in_l_folder(l_folder, ys_number):
    """
    Finds the Fig{ys_number}.txt file in the L folder.
    For example, if ys_number is 2, looks for Fig2.txt.
    Returns the path if found, None otherwise.
    """
    fig_filename = f"Fig{ys_number}.txt"
    fig_path = l_folder / fig_filename
    
    if fig_path.exists():
        return fig_path
    return None


def process_ys_folder_with_llm(l_folder, ys_folder, ys_number, prompt_text, client):
    """
    Processes a single YS folder:
    1. Find out_filter.csv in YS folder (or subfolders)
    2. Find Fig{ys_number}.txt in L folder
    3. Call LLM with both files
    4. Save output to Fig{ys_number}_material.txt
    
    Returns status string.
    """
    try:
        # Step 1: Find out_filter.csv
        out_filter_path = find_out_filter_csv(ys_folder)
        if not out_filter_path:
            return "Skipped (out_filter.csv not found)"
        
        # Step 2: Find Fig{ys_number}.txt in L folder
        fig_txt_path = find_fig_txt_in_l_folder(l_folder, ys_number)
        if not fig_txt_path:
            return f"Skipped (Fig{ys_number}.txt not found in L folder)"
        
        # Step 3: Define output path
        output_path = l_folder / f"Fig{ys_number}_material.txt"
        
        # Skip if already processed
        if output_path.exists():
            return "Skipped (already processed)"
        
        # Step 4: Read input files
        out_filter_content = read_file_content(out_filter_path)
        fig_txt_content = read_file_content(fig_txt_path)
        
        if out_filter_content is None or fig_txt_content is None:
            return "Error (could not read input files)"
        
        # Step 5: Construct prompt and call LLM
        start_time = time.time()
        
        system_prompt = "You are a materials science data extraction specialist. Your task is to analyze data and extract material names according to the given instructions."
        
        user_prompt = (
            f"Content from out_filter.csv:\n{out_filter_content}\n\n"
            f"Content from fig.txt:\n{fig_txt_content}\n\n"
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
        
        # Step 6: Save the result
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_content)
        
        process_time = time.time() - start_time
        token_usage = response.usage.total_tokens if hasattr(response, 'usage') else "N/A"
        
        return f"Success | Time: {process_time:.2f}s | Tokens: {token_usage}"
    
    except Exception as e:
        return f"Error ({e})"


def main_process_loop(base_path, prompt_path):
    """
    Main loop to iterate through L0001-L2594 folders and process YS subfolders.
    """
    client = initialize_llm_client()
    
    # Read the prompt template
    prompt_text = read_file_content(prompt_path)
    if prompt_text is None:
        print("Aborting: Could not read the main prompt file.")
        return
    
    print("=" * 80)
    print("Starting batch LLM processing for material extraction")
    print("=" * 80)
    
    processed_count = 0
    error_count = 0
    skipped_count = 0
    
    # Iterate through L0001 to L2594
    for i in tqdm(range(1, 3356), desc="Processing L folders"):
        folder_name = f"L{i:04d}"
        l_folder = Path(base_path) / folder_name
        
        # Check if L folder exists
        if not l_folder.exists() or not l_folder.is_dir():
            continue
        
        # Find all YS* subfolders
        ys_folders = sorted([f for f in l_folder.iterdir() 
                            if f.is_dir() and f.name.startswith('YS')])
        
        if not ys_folders:
            skipped_count += 1
            continue
        
        print(f"\n[{folder_name}] Found {len(ys_folders)} YS folder(s)")
        
        # Process each YS folder
        for ys_folder in ys_folders:
            ys_name = ys_folder.name
            
            # Extract YS number (e.g., YS2 -> 2, YS8 -> 8)
            try:
                ys_number = int(ys_name[2:])  # Remove "YS" prefix
            except ValueError:
                print(f"  ⚠ Warning: Could not extract number from {ys_name}, skipping")
                skipped_count += 1
                continue
            
            print(f"  Processing: {ys_name} (looking for Fig{ys_number}.txt)")
            
            status = process_ys_folder_with_llm(
                l_folder, ys_folder, ys_number, prompt_text, client
            )
            
            if "Success" in status:
                print(f"    ✓ {status}")
                processed_count += 1
            elif "Skipped" in status:
                print(f"    ⊘ {status}")
                skipped_count += 1
            else:  # Error
                print(f"    ✗ {status}")
                error_count += 1
    
    print("\n" + "=" * 80)
    print("Processing complete!")
    print(f"  Total Successful: {processed_count}")
    print(f"  Total Skipped:    {skipped_count}")
    print(f"  Total Errors:     {error_count}")
    print("=" * 80)


if __name__ == "__main__":
    # Define the main paths for the operation
    base_directory = Path.cwd()  # Current working directory
    prompt_file_path = r".\stress-strain\prompt\material_name.md"
    
    main_process_loop(base_directory, prompt_file_path)