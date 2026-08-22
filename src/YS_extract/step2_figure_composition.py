import os
import time
import pandas as pd
import numpy as np
from openai import OpenAI

# Initialize Qwen's API client
try:
    client = OpenAI(
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
except Exception as e:
    print(f"Error initializing client: {e}")
    exit(1)

def read_prompt_file(prompt_path):
    """Read the prompt from the specified file."""
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading prompt file {prompt_path}: {e}")
        return None

def process_composition_with_llm(folder_path, prompt_text):
    """
    Processes data+composition_mod.json with an LLM if figure_data_composition.csv has at least one row
    with all zeros in columns 6 to 108, and saves the result to data+composition_mod2.json.
    """
    try:
        start_time = time.time()
        
        # Step 2: Check figure_data_composition.csv
        csv_path = os.path.join(folder_path, "figure_data_composition.csv")
        if not os.path.exists(csv_path):
            return False, "Error (no figure_data_composition.csv found)"
        
        # Read CSV and check columns 6 to 108
        df = pd.read_csv(csv_path)
        if len(df.columns) < 108:
            return False, "Error (fewer than 108 columns in CSV)"
        
        cols = df.iloc[:, 5:108].apply(pd.to_numeric, errors='coerce').fillna(0)
        # Check if any row has all zeros in columns 6 to 108
        zero_rows = (cols == 0).all(axis=1)
        if not zero_rows.any():
            return False, "Error (no rows with all zeros in columns 6 to 108)"
        
        # Step 3: Check for data+composition_mod.json
        source_json_path = os.path.join(folder_path, "data+composition_mod.json")
        if not os.path.exists(source_json_path):
            return False, "Error (no data+composition_mod.json found)"
        
        # Define the output file path
        output_path = os.path.join(folder_path, "data+composition_mod2.json")
        
        # Skip if output file already exists to avoid reprocessing
        if os.path.exists(output_path):
            return False, "Skipped (already processed)"
        
        # Read the JSON file content
        with open(source_json_path, "r", encoding="utf-8") as f:
            json_content = f.read()
        
        # Define system and user prompts
        system_prompt = "You are an expert in materials science, specializing in integrating and standardizing chemical composition and experimental data from scientific literature."
        user_prompt = f"Input data (data+composition.json):\n{json_content}\n\nBased on the provided data, please process and integrate the information according to the following instructions:\n{prompt_text}"
        
        # Build the messages for the API call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Call the LLM model
        response = client.chat.completions.create(
            model="qwen-max-latest",
            messages=messages,
            temperature=0.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stream=False
        )
        
        # Extract the response content
        output_content = response.choices[0].message.content.strip()
        
        # Performance statistics
        process_time = time.time() - start_time
        token_usage = response.usage.total_tokens if hasattr(response, 'usage') else "N/A"
        
        # Save the result to the new JSON file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_content)
        
        print(f"Processed: {folder_path} | Time: {process_time:.2f}s | Tokens: {token_usage}")
        return True, "Success"
    
    except Exception as e:
        print(f"Failed to process files in {folder_path}: {str(e)}")
        return False, f"Error ({e})"

def process_all_folders(base_path, prompt_path):
    """Iterates through all folders and processes those with required files."""
    prompt_text = read_prompt_file(prompt_path)
    if prompt_text is None:
        print("Aborting due to prompt file error.")
        return
    
    processed_count = 0
    error_count = 0
    skipped_count = 0
    
    # Step 1: Iterate through all potential subfolders
    for i in range(1, 3317):
        folder_name = f"L{i:04d}"
        folder_path = os.path.join(base_path, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
        
        # Process folder if required files exist
        success, status = process_composition_with_llm(folder_path, prompt_text)
        
        if success:
            processed_count += 1
        elif "Skipped" in status:
            skipped_count += 1
        else:
            error_count += 1
        
        if i % 20 == 0:
            print(f"--- Progress: Up to folder {i}/3316 ---")
    
    print("\n-------------------------------------------")
    print(f"Processing complete!")
    print(f"Total Successful: {processed_count}")
    print(f"Total Skipped: {skipped_count}")
    print(f"Total Errors: {error_count}")
    print("-------------------------------------------")

# --- Main execution block ---
if __name__ == "__main__":
    base_path = r".\HEA_finetune\Yield_Strength"
    prompt_path = r".\HEA_finetune\Yield_Strength\prompt\step2_figure_composition.md"
    process_all_folders(base_path, prompt_path)