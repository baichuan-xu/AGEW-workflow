import dashscope
import os
import glob
import json

# --- Configuration ---
BASE_PATH = r".\HEA_finetune\Yield_Strength"
PROMPT_FILE_PATH = r".\HEA_finetune\Yield_Strength\Prompt\bar_description.md"
IMAGE_FILE_NAME = "in.png"
OUTPUT_JSON_FILE_NAME = "description.json"

# Function to read text file content
def read_text_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as text_file:
            return text_file.read().strip()
    except FileNotFoundError:
        print(f"Error: Text file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading text file {file_path}: {e}")
        return None

# Function to write text to a file
def write_text_to_file(file_path, content):
    try:
        with open(file_path, "w", encoding="utf-8") as text_file:
            text_file.write(content)
        if content and content.strip():
            print(f"Successfully wrote LLM output to {file_path}")
        else:
            print(f"LLM provided empty or whitespace-only output. {file_path} will reflect this.")
    except Exception as e:
        print(f"Error writing to file {file_path}: {e}")

# Function to process image with LLM
def process_image_with_llm(image_path, prompt_text, output_path, api_key):
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        write_text_to_file(output_path, '{"error": "IMAGE_NOT_FOUND"}')
        return False

    local_image_path_for_api = f"file://{os.path.abspath(image_path)}"
    messages = [
        {
            "role": "system",
            "content": "You are an expert image analysis AI specializing in extracting details from graphs."
        },
        {
            "role": "user",
            "content": [
                {"image": local_image_path_for_api},
                {"text": f"The image provided is '{os.path.basename(image_path)}', a bar plot with colored data series.\n\n"
                         f"Please follow the instructions in 'description.md' to analyze the image, extract the label and color for each data series from the legend, and output a JSON object with the structure defined in the instructions.\n\n"
                         f"Instructions: {prompt_text}\n\n"}
            ]
        }
    ]

    print(f"Sending request to DashScope LLM for {image_path}...")
    try:
        response = dashscope.MultiModalConversation.call(
            api_key=api_key,
            model='qwen-vl-max-latest',
            messages=messages,
            temperature=0.0,
            vl_high_resolution_images=True
        )

        if response.status_code == 200 and response.output and response.output.choices:
            llm_output_text = ""
            if response.output.choices[0].message.content and isinstance(response.output.choices[0].message.content, list):
                for item in response.output.choices[0].message.content:
                    if "text" in item:
                        llm_output_text = item["text"]
                        break
            elif isinstance(response.output.choices[0].message.content, str):
                llm_output_text = response.output.choices[0].message.content
            
            processed_llm_output = llm_output_text.strip()
            write_text_to_file(output_path, processed_llm_output)

            if response.usage:
                print("Token Usage:")
                print(f"  Input Tokens: {response.usage.get('input_tokens', 'N/A')}")
                print(f"  Output Tokens: {response.usage.get('output_tokens', 'N/A')}")
            return True
        else:
            print(f"Error from API: Status Code {response.status_code}")
            print(f"Response: {response}")
            write_text_to_file(output_path, f'{{"error": "API_ERROR", "status_code": {response.status_code}}}')
            return False
    except Exception as e:
        print(f"Error during DashScope API call for {image_path}: {e}")
        write_text_to_file(output_path, f'{{"error": "SCRIPT_ERROR", "exception_type": "{type(e).__name__}", "message": "{str(e)}"}}')
        return False

def process_folders(base_path, prompt_file_path):
    """
    Iterates through folders and processes in.png based on check_data.txt and graph_out_fixed.json.
    """
    # Read the prompt file once
    prompt_text = read_text_file(prompt_file_path)
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

    for i in range(1, 3317):
        folder_name = f"L{i:04d}"
        folder_path = os.path.join(base_path, folder_name)

        if not os.path.isdir(folder_path):
            print(f"Folder {folder_path} does not exist, skipping")
            continue

        print(f"Checking folder: {folder_path}")

        ys_folders = glob.glob(os.path.join(folder_path, "YS*"))
        if not ys_folders:
            print(f"No YS* folders found in {folder_path}, skipping")
            continue

        for ys_folder in ys_folders:
            if not os.path.isdir(ys_folder):
                print(f"{ys_folder} is not a directory, skipping")
                continue

            print(f"Found YS folder: {ys_folder}")

            data_folders = glob.glob(os.path.join(ys_folder, "data*"))
            if not data_folders:
                print(f"No data* folders found in {ys_folder}, skipping")
                continue

            for data_folder in data_folders:
                if not os.path.isdir(data_folder):
                    print(f"{data_folder} is not a directory, skipping")
                    continue

                print(f"Found data folder: {data_folder}")

                # Check check_data.txt
                txt_file = os.path.join(data_folder, "check_data.txt")
                if not os.path.isfile(txt_file):
                    print(f"check_data.txt not found in {data_folder}, skipping")
                    continue

                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    if content != "No Numerical annotation":
                        print(f"check_data.txt content '{content}' in {data_folder} does not match 'No Numerical annotation', skipping")
                        continue
                except Exception as e:
                    print(f"Error reading {txt_file}: {e}, skipping")
                    continue

                # Check graph_out_fixed.json
                json_file = os.path.join(data_folder, "graph_out_fixed.json")
                if not os.path.isfile(json_file):
                    print(f"graph_out_fixed.json not found in {data_folder}, skipping")
                    continue

                try:
                    with open(json_file, 'r') as f:
                        json_data = json.load(f)
                    image_type = json_data.get("image_type", "")
                    if image_type != "bar chart":
                        print(f"image_type '{image_type}' in {json_file} is not 'bar chart', skipping")
                        continue
                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON in {json_file}, skipping")
                    continue
                except Exception as e:
                    print(f"Error reading {json_file}: {e}, skipping")
                    continue

                # Process in.png if conditions are met
                input_file = os.path.join(data_folder, IMAGE_FILE_NAME)
                output_file = os.path.join(data_folder, OUTPUT_JSON_FILE_NAME)

                if not os.path.isfile(input_file):
                    print(f"in.png not found in {data_folder}, skipping")
                    continue

                print(f"Processing {input_file}...")
                process_image_with_llm(input_file, prompt_text, output_file, api_key)

# Usage
if __name__ == "__main__":
    process_folders(BASE_PATH, PROMPT_FILE_PATH)