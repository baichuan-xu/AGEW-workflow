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

def process_value_errors_with_llm(folder_path, prompt_text):
    """
    Processes composition data and value error reports with an LLM,
    and saves the result to composition_value.json.
    """
    try:
        start_time = time.time()
        
        # Define paths for the required input files
        error_txt_path = os.path.join(folder_path, "composition_value_error.txt")

        # 3. Determine which JSON file to use as the source
        composition2_path = os.path.join(folder_path, "composition2.json")
        composition_path = os.path.join(folder_path, "composition.json")
        
        source_json_path = ""
        if os.path.exists(composition2_path):
            source_json_path = composition2_path
        elif os.path.exists(composition_path):
            source_json_path = composition_path
        else:
            print(f"Error: Neither composition2.json nor composition.json found in {folder_path}!")
            return False, "Error (missing source JSON)"

        # Define the output file path
        output_path = os.path.join(folder_path, "composition_value.json")
        
        # Skip if output file already exists to avoid reprocessing
        if os.path.exists(output_path):
            return False, "Skipped (already processed)"

        # Read the content of the input files
        with open(error_txt_path, "r", encoding="utf-8") as f:
            error_content = f.read()
            
        with open(source_json_path, "r", encoding="utf-8") as f:
            composition_content = f.read()

        # Define system and user prompts
        system_prompt = "You are an expert in materials science, specializing in correcting and standardizing chemical composition data from scientific literature based on error reports."
        # 3. Note: The prompt always refers to the input as "composition.json"
        user_prompt = f"Original composition.json content:\n{composition_content}\n\nIdentified Errors in composition_value_error.txt:\n{error_content}\n\nBased on the errors, please correct the original json content according to the following instructions:\n{prompt_text}"

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
        
        # 4. Save the result to the new JSON file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_content)
        
        print(f"Processed: {folder_path} | Source: {os.path.basename(source_json_path)} | Time: {process_time:.2f}s | Tokens: {token_usage}")
        return True, "Success"

    except Exception as e:
        print(f"Failed to process files in {folder_path}: {str(e)}")
        return False, f"Error ({e})"

def process_all_folders(base_path, prompt_path):
    """Iterates through all folders and processes those with a composition_value_error.txt file."""
    prompt_text = read_prompt_file(prompt_path)
    if prompt_text is None:
        print("Aborting due to prompt file error.")
        return

    processed_count = 0
    error_count = 0
    skipped_count = 0
    
    # 1. Iterate through all potential subfolders
    for i in range(1, 3317):
        folder_name = f"L{i:04d}"
        folder_path = os.path.join(base_path, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
        
        # 2. Check if 'composition_value_error.txt' exists
        error_file_path = os.path.join(folder_path, "composition_value_error.txt")
        if not os.path.isfile(error_file_path):
            continue # If it doesn't exist, skip this folder.
        
        # If the error file exists, proceed to process
        print(f"\nFound error file in {folder_path}. Attempting to process...")
        success, status = process_value_errors_with_llm(folder_path, prompt_text)
        
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
    # 4. Point to the correct prompt file
    prompt_path = r".\HEA_finetune\Yield_Strength\prompt\composition_value_error.md"
    process_all_folders(base_path, prompt_path)
