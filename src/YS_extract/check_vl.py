import dashscope
import os
import json

# --- Configuration ---
BASE_PATH = r".\HEA_finetune\Yield_Strength"
SUBFOLDER_PREFIX = "L"
NUM_SUBFOLDERS = 3316
YS_PREFIX = "YS"
DATA_PREFIX = "data"
INPUT_IMAGE_NAME = "in.png"
PROMPT_FILE_PATH = r".\HEA_finetune\Yield_Strength\Prompt\check_graph.md"
OUTPUT_JSON_NAME = "check_out.json"

# Function to read text file content
def read_text_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as text_file:
            return text_file.read()
    except FileNotFoundError:
        print(f"Error: Text file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading text file {file_path}: {e}")
        return None

# Function to write JSON to a file
def write_json_to_file(file_path, content):
    try:
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(content, json_file, ensure_ascii=False, indent=2)
        print(f"Successfully wrote output to {file_path}")
    except Exception as e:
        print(f"Error writing to file {file_path}: {e}")

# Function to process a single image with the LLM
def process_image_with_llm(image_path, prompt_text, output_json_path, api_key):
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return False

    local_image_path_for_api = f"file://{os.path.abspath(image_path)}"
    messages = [
        {
            "role": "system",
            "content": "You are an advanced image analysis assistant. Your task is to process the provided image, `in.png`, which typically contains data graphs or microscope imagery from scientific literature."
        },
        {
            "role": "user",
            "content": [
                {"image": local_image_path_for_api},
                {"text": f"Please follow these instructions from check_graph.md:\n\n{prompt_text}"}
            ]
        }
    ]

    print(f"Sending request to DashScope LLM for {image_path}...")
    try:
        response = dashscope.MultiModalConversation.call(
            api_key=api_key,
            model='qwen-vl-max',
            messages=messages,
            temperature=0.0,
            vl_high_resolution_images=True
        )

        if response.status_code == 200 and response.output and response.output.choices:
            llm_output = response.output.choices[0].message.content
            # Ensure output is a dictionary or list for JSON serialization
            if isinstance(llm_output, list) and llm_output and isinstance(llm_output[0], dict):
                llm_output = llm_output[0].get("text", {})
            elif not isinstance(llm_output, (dict, list)):
                llm_output = {"text": str(llm_output)}
            write_json_to_file(output_json_path, llm_output)
            if response.usage:
                print("Token Usage:")
                print(f"  Input Total Tokens: {response.usage.get('input_tokens', 'N/A')}")
                print(f"  Input Image Tokens: {response.usage.get('image_tokens', 'N/A')}")
                print(f"  Output Text Tokens: {response.usage.get('output_tokens', 'N/A')}")
            return True
        else:
            print(f"Error from API: Status Code {response.status_code}")
            print(f"Response: {response}")
            return False
    except Exception as e:
        print(f"Error during DashScope API call for {image_path}: {e}")
        return False

def main():
    """
    Iterate through L0001 to L3316, process 'data' subfolders within YS subfolders.
    """
    if not os.path.exists(BASE_PATH):
        print(f"Error: Base path '{BASE_PATH}' does not exist.")
        return

    # Read the prompt file once
    prompt_text = read_text_file(PROMPT_FILE_PATH)
    if prompt_text is None:
        print("Error: Could not read prompt file. Aborting.")
        return

    # API Key
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("Warning: Using hardcoded API key. Consider setting DASHSCOPE_API_KEY environment variable.")
    if not api_key:
        print("Error: No API key provided.")
        return

    print(f"Processing subfolders in {BASE_PATH}...")
    for i in range(1, NUM_SUBFOLDERS + 1):
        subfolder_name = f"{SUBFOLDER_PREFIX}{i:04d}"
        subfolder_path = os.path.join(BASE_PATH, subfolder_name)

        if not os.path.exists(subfolder_path):
            print(f"Subfolder {subfolder_path} does not exist. Skipping...")
            continue

        # Check for YS subfolders
        for item in os.listdir(subfolder_path):
            if item.startswith(YS_PREFIX) and os.path.isdir(os.path.join(subfolder_path, item)):
                ys_subfolder_path = os.path.join(subfolder_path, item)

                # Check for 'data' subfolders
                for sub_item in os.listdir(ys_subfolder_path):
                    if sub_item.startswith(DATA_PREFIX) and os.path.isdir(os.path.join(ys_subfolder_path, sub_item)):
                        data_subfolder_path = os.path.join(ys_subfolder_path, sub_item)
                        image_path = os.path.join(data_subfolder_path, INPUT_IMAGE_NAME)
                        output_json_path = os.path.join(data_subfolder_path, OUTPUT_JSON_NAME)

                        if not os.path.exists(image_path):
                            print(f"Image {image_path} not found. Skipping...")
                            continue

                        print(f"\nProcessing subfolder: {data_subfolder_path}")
                        process_image_with_llm(image_path, prompt_text, output_json_path, api_key)

    print("\nBatch processing complete.")

if __name__ == "__main__":
    main()