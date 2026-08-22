import dashscope
import os
import time

# --- Configuration ---
BASE_PATH = os.getcwd()
SUBFOLDER_PREFIX = "L"
NUM_SUBFOLDERS = 3356
YS_PREFIX = "YS"
SPLIT_PREFIX = "split"
INPUT_IMAGE_NAME = "in.png"
# This file must exist in a directory for in.png to be processed.
REQUIRED_INPUT_CSV = "split_check.csv"
# The full path to the new markdown file containing the LLM prompt.
PROMPT_FILE_PATH = r".\stress-strain\prompt\line_label.md"
# The new output filename.
OUTPUT_CSV_NAME = "label.csv"

def read_text_file(file_path):
    """Reads and returns the content of a text file."""
    try:
        with open(file_path, "r", encoding="utf-8") as text_file:
            return text_file.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def write_to_csv(file_path, content):
    """Writes the given string content directly to a file, overwriting if it exists."""
    try:
        if not isinstance(content, str):
            raise TypeError(f"Content to be written must be a string, not {type(content)}")
        with open(file_path, "w", encoding="utf-8") as csv_file:
            csv_file.write(content)
        print(f"Successfully wrote output to {file_path}")
    except Exception as e:
        print(f"Error writing to file {file_path}: {e}")

def process_image_with_llm(image_path, prompt_text, output_csv_path, api_key):
    """Sends a single image to the DashScope LLM and saves the response."""
    # This function assumes checks for image existence have already been made.
    local_image_path_for_api = f"file://{os.path.abspath(image_path)}"
    messages = [
        {"role": "system", "content": "You are a highly accurate assistant for scientific figure analysis. Your task is to label the lines or elements in the provided image as instructed."},
        {"role": "user", "content": [{"image": local_image_path_for_api}, {"text": prompt_text}]}
    ]

    print(f"Sending request to DashScope LLM for {image_path}...")
    try:
        response = dashscope.MultiModalConversation.call(
            api_key=api_key,
            model='qwen-vl-max',
            messages=messages,
            temperature=0.0
        )
        if response.status_code == 200 and response.output and response.output.choices:
            llm_output = response.output.choices[0].message.content
            final_text_output = ""
            if isinstance(llm_output, list) and llm_output:
                content_part = llm_output[0]
                final_text_output = content_part.get("text", "") if isinstance(content_part, dict) else str(content_part)
            else:
                final_text_output = str(llm_output)
            
            write_to_csv(output_csv_path, final_text_output)

            if response.usage:
                print(f"Token Usage: Input={response.usage.get('input_tokens', 'N/A')}, Output={response.usage.get('output_tokens', 'N/A')}")
            return True
        else:
            print(f"Error from API: Status {response.status_code}, Response: {response}")
            return False
    except Exception as e:
        print(f"An exception occurred during API call for {image_path}: {e}")
        return False

def main():
    """Main function to iterate through directories and process images based on the new logic."""
    prompt_text = read_text_file(PROMPT_FILE_PATH)
    if prompt_text is None:
        print("Aborting: Could not read prompt file.")
        return

    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("Warning: Using hardcoded API key. Consider setting DASHSCOPE_API_KEY environment variable.")
    if not api_key:
        print("Error: No API key provided.")
        return

    print(f"Starting batch processing in: {BASE_PATH}")
    for i in range(1, NUM_SUBFOLDERS + 1):
        l_subfolder_path = os.path.join(BASE_PATH, f"{SUBFOLDER_PREFIX}{i:04d}")
        if not os.path.isdir(l_subfolder_path):
            continue

        print(f"\n--- Checking {l_subfolder_path} ---")

        for ys_item in os.listdir(l_subfolder_path):
            ys_subfolder_path = os.path.join(l_subfolder_path, ys_item)
            if ys_item.startswith(YS_PREFIX) and os.path.isdir(ys_subfolder_path):
                
                # Check for subfolders starting with 'split' first
                split_folders = [os.path.join(ys_subfolder_path, s) for s in os.listdir(ys_subfolder_path) if s.startswith(SPLIT_PREFIX) and os.path.isdir(os.path.join(ys_subfolder_path, s))]
                
                if split_folders:
                    # Case 1: Process 'split' subfolders
                    print(f"Found {len(split_folders)} 'split' subfolder(s) in {ys_subfolder_path}.")
                    for split_folder_path in split_folders:
                        image_path = os.path.join(split_folder_path, INPUT_IMAGE_NAME)
                        required_csv_path = os.path.join(split_folder_path, REQUIRED_INPUT_CSV)
                        output_csv_path = os.path.join(split_folder_path, OUTPUT_CSV_NAME)

                        # Conditional check: both in.png and split_check.csv must exist
                        if os.path.exists(image_path) and os.path.exists(required_csv_path):
                            print(f"Processing conditional image in: {split_folder_path}")
                            process_image_with_llm(image_path, prompt_text, output_csv_path, api_key)
                        else:
                            print(f"Skipping {split_folder_path}: missing '{INPUT_IMAGE_NAME}' or '{REQUIRED_INPUT_CSV}'.")
                else:
                    # Case 2: No 'split' folders, check the YS folder directly
                    image_path = os.path.join(ys_subfolder_path, INPUT_IMAGE_NAME)
                    required_csv_path = os.path.join(ys_subfolder_path, REQUIRED_INPUT_CSV)
                    output_csv_path = os.path.join(ys_subfolder_path, OUTPUT_CSV_NAME)

                    # Conditional check for the YS folder
                    if os.path.exists(image_path) and os.path.exists(required_csv_path):
                        print(f"No 'split' subfolders found. Processing conditional image in: {ys_subfolder_path}")
                        process_image_with_llm(image_path, prompt_text, output_csv_path, api_key)
                    else:
                        print(f"Skipping {ys_subfolder_path}: no 'split' folders and missing root '{INPUT_IMAGE_NAME}' or '{REQUIRED_INPUT_CSV}'.")

    print("\n--- Batch processing complete. ---")

if __name__ == "__main__":
    main()
