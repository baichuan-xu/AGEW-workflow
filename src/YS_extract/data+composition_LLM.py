import os
import time
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

def process_composition_and_data_with_llm(folder_path, prompt_text):
    """
    Processes composition and table/figure data with an LLM,
    and saves the result to data+composition.json.
    """
    try:
        start_time = time.time()
        
        # Step 2: Determine which composition JSON file to use
        composition_files = [
            'composition3.json',
            'composition2.json',
            'composition_mod.json'
        ]
        
        source_composition_path = None
        for comp_file in composition_files:
            potential_path = os.path.join(folder_path, comp_file)
            if os.path.exists(potential_path):
                source_composition_path = potential_path
                break
        
        if not source_composition_path:
            return False, "Error (no composition JSON found)"

        # Step 3: Check for table.json and/or figure.json
        table_path = os.path.join(folder_path, "table.json")
        figure_path = os.path.join(folder_path, "figure.json")
        
        data_content = ""
        if os.path.exists(table_path) and os.path.exists(figure_path):
            with open(table_path, "r", encoding="utf-8") as f:
                data_content += f.read() + "\n"
            with open(figure_path, "r", encoding="utf-8") as f:
                data_content += f.read()
        elif os.path.exists(table_path):
            with open(table_path, "r", encoding="utf-8") as f:
                data_content = f.read()
        elif os.path.exists(figure_path):
            with open(figure_path, "r", encoding="utf-8") as f:
                data_content = f.read()
        else:
            return False, "Error (no table.json or figure.json found)"

        # Define the output file path
        output_path = os.path.join(folder_path, "data+composition.json")
        
        # Skip if output file already exists to avoid reprocessing
        if os.path.exists(output_path):
            return False, "Skipped (already processed)"

        # Read the composition file content
        with open(source_composition_path, "r", encoding="utf-8") as f:
            composition_content = f.read()

        # Define system and user prompts
        system_prompt = "You are an expert in materials science, specializing in integrating and standardizing chemical composition and experimental data from scientific literature."
        user_prompt = f"Composition data (composition.json):\n{composition_content}\n\nExperimental data (data.json):\n{data_content}\n\nBased on the provided data, please process and integrate the information according to the following instructions:\n{prompt_text}"

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
        
        print(f"Processed: {folder_path} | Source: {os.path.basename(source_composition_path)} | Time: {process_time:.2f}s | Tokens: {token_usage}")
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
        print(f"\nChecking folder {folder_path}...")
        success, status = process_composition_and_data_with_llm(folder_path, prompt_text)
        
        if success:
            processed_count += 1
        elif "Skipped" in status:
            skipped_count += 1
            print(f"Skipped: {folder_path} ({status})")
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
    prompt_path = r".\HEA_finetune\Yield_Strength\prompt\data+composition.md"
    process_all_folders(base_path, prompt_path)