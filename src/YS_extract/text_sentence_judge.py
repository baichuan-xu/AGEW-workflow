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

def process_text_csv_with_llm(folder_path, prompt_text):
    """Process a text.csv file with the LLM model and save results to text_sentence_judge.csv."""
    try:
        start_time = time.time()
        csv_path = os.path.join(folder_path, "text_split.csv")
        
        # Check file existence
        if not os.path.exists(csv_path):
            print(f"Error: text_split.csv not found in {folder_path}!")
            return False
            
        output_filename = "text_sentence_judge.csv"
        output_path = os.path.join(folder_path, output_filename)
        
        # Skip if output file already exists
        if os.path.exists(output_path):
            print(f"Skipping {folder_path} (already processed)")
            return False

        # Read the content of the CSV file
        with open(csv_path, "r", encoding="utf-8") as f:
            csv_content = f.read()

        # Define system and user prompts
        system_prompt = "You are an expert in extracting key information from scientific literature related to material properties and processing."
        user_prompt = f"CSV Content:\n{csv_content}\n\n{prompt_text}"

        # Build the messages for the API
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
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
        
        # Save the result as CSV
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_content)
        
        print(f"Processed text_split.csv in {folder_path} | Time: {process_time:.2f}s | Tokens: {token_usage}")
        return True

    except Exception as e:
        print(f"Failed to process text_split.csv in {folder_path}: {str(e)}")
        return False

def process_all_folders(base_path, prompt_path):
    """Process text_split.csv files in all folders L0001 to L3316."""
    # Read the prompt file
    prompt_text = read_prompt_file(prompt_path)
    if prompt_text is None:
        print("Aborting due to prompt file error.")
        return

    processed_count = 0
    error_count = 0
    
    for i in range(1, 3317):  # Including 3316
        folder_name = f"L{i:04d}"
        folder_path = os.path.join(base_path, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"Folder {folder_name} does not exist. Skipping.")
            continue
        
        # Process the text.csv file
        success = process_text_csv_with_llm(folder_path, prompt_text)
        
        if success:
            processed_count += 1
        else:
            error_count += 1
        
        # Progress update every 10 folders
        if i % 10 == 0:
            print(f"Progress: {i}/3316 folders processed. Successful: {processed_count}, Errors: {error_count}")
    
    print(f"Processing complete! Total processed: {processed_count}, Total errors: {error_count}")

# Usage
if __name__ == "__main__":
    base_path = r".\HEA_finetune\Yield_Strength"
    prompt_path = r".\HEA_finetune\Yield_Strength\prompt\text_sentence_judge.md"
    process_all_folders(base_path, prompt_path)