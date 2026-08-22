import dashscope
import os
import json

# Base directory containing L0001 to L3316 folders
base_dir = r".\HEA_finetune\Yield_Strength"
prompt_dir = r".\HEA_finetune\Yield_Strength\Prompt"

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

# Function to write text to a file (creates or overwrites file)
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

# Function to check first line of check_data.txt
def check_data_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as text_file:
            first_line = text_file.readline().strip()
            return first_line == "No Numerical annotation"
    except FileNotFoundError:
        print(f"Error: check_data.txt not found at {file_path}")
        return False
    except Exception as e:
        print(f"Error reading check_data.txt {file_path}: {e}")
        return False

# Function to process a single folder
def process_folder(folder_path, folder_name):
    print(f"\nProcessing folder: {folder_name}")
    
    # Check for YS* subfolders
    ys_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f)) and f.startswith("YS")]
    
    for ys_folder in ys_folders:
        ys_folder_path = os.path.join(folder_path, ys_folder)
        print(f"Found YS folder: {ys_folder}")
        
        # Check for subfolders starting with "data"
        data_folders = [f for f in os.listdir(ys_folder_path) if os.path.isdir(os.path.join(ys_folder_path, f)) and f.startswith("data")]
        if not data_folders:
            print(f"No folder starting with 'data' found in {ys_folder}. Skipping.")
            continue
        
        for data_folder in data_folders:
            data_folder_path = os.path.join(ys_folder_path, data_folder)
            print(f"Found data folder: {data_folder}")
            
            # Check for check_data.txt and its first line
            check_data_path = os.path.join(data_folder_path, "check_data.txt")
            if not check_data_file(check_data_path):
                print(f"check_data.txt in {data_folder_path} does not start with 'No Numerical annotation' or is missing. Skipping.")
                continue
            
            # Check for graph_out_fixed.json
            json_file_path = os.path.join(data_folder_path, "graph_out_fixed.json")
            if not os.path.exists(json_file_path):
                print(f"graph_out_fixed.json not found in {data_folder_path}. Skipping.")
                continue
            
            # Read graph_out_fixed.json
            try:
                with open(json_file_path, "r", encoding="utf-8") as json_file:
                    graph_data = json.load(json_file)
                
                # Check x_axis_type
                x_axis_type = graph_data.get("x_axis_type", "")
                if x_axis_type != "text labels":
                    print(f"x_axis_type '{x_axis_type}' in {json_file_path} is not 'text label'. Skipping.")
                    continue
                
                # Check image_type
                image_type = graph_data.get("image_type", "")
                if image_type not in ["dot-line plot", "scatter plot"]:
                    print(f"image_type '{image_type}' in {json_file_path} is not 'dot-line plot' or 'scatter plot'. Skipping.")
                    continue
            except Exception as e:
                print(f"Error reading {json_file_path}: {e}. Skipping.")
                continue
            
            # Check for in.png and out_point.png
            input_files = ["in.png", "out_point.png"]
            input_file_paths = [os.path.join(data_folder_path, f) for f in input_files]
            if not all(os.path.exists(f) for f in input_file_paths):
                print(f"One or more input files {input_files} not found in {data_folder_path}. Skipping.")
                continue
            
            # Read prompt file
            prompt_file_name = "Dot_extract_point3.md"
            prompt_file_path = os.path.join(prompt_dir, prompt_file_name)
            prompt_instructions = read_text_file(prompt_file_path)
            if prompt_instructions is None:
                print(f"Prompt file {prompt_file_name} could not be read. Skipping.")
                continue
            
            # API Key
            api_key_to_use = os.getenv('DASHSCOPE_API_KEY')
            if not api_key_to_use:
                print("Warning: Using a placeholder API key. Please set the DASHSCOPE_API_KEY environment variable.")
            if not api_key_to_use:
                print("Error: DASHSCOPE_API_KEY not found.")
                write_text_to_file(os.path.join(data_folder_path, "scatter_data.csv"), "ERROR_NO_API_KEY")
                continue
            
            # Prepare API call with explicit image labeling
            messages = [
                {
                    "role": "system",
                    "content": "Your task is to analyze the provided images 'in.png' and 'out_point.png' and extract data points according to the prompt instructions. Output the results in CSV format."
                },
                {
                    "role": "user",
                    "content": [
                        {"text": f"Image 'in.png': This is the primary graph image containing data points to analyze."},
                        {"image": f"file://{os.path.abspath(input_file_paths[0])}"},
                        {"text": f"Image 'out_point.png': This image contains additional point markings to assist with data extraction."},
                        {"image": f"file://{os.path.abspath(input_file_paths[1])}"},
                        {"text": f"Please analyze the images according to the instructions in '{prompt_file_name}' to extract data points. "
                                 f"Use 'in.png' as the primary graph and 'out_point.png' for additional point markings. "
                                 f"Output a CSV with the extracted coordinates and their validity:\n\n{prompt_instructions}"}
                    ]
                }
            ]
            
            print(f"Sending request to DashScope LLM for {data_folder} in {ys_folder} with images {input_files} and prompt {prompt_file_name}...")
            try:
                response = dashscope.MultiModalConversation.call(
                    api_key=api_key_to_use,
                    model='qwen2.5-vl-72b-instruct',
                    messages=messages,
                    temperature=0.0,
                    vl_high_resolution_images=True
                )
                
                output_csv_path = os.path.join(data_folder_path, "scatter_data.csv")
                
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
                    write_text_to_file(output_csv_path, processed_llm_output)
                    
                    if response.usage:
                        print("\nToken Usage:")
                        print(f"  Input Tokens: {response.usage.get('input_tokens', 'N/A')}")
                        print(f"  Output Tokens: {response.usage.get('output_tokens', 'N/A')}")
                else:
                    print(f"Error from API: Status Code {response.status_code}")
                    print(f"Response: {response}")
                    write_text_to_file(output_csv_path, f"API_ERROR_STATUS_{response.status_code}")
            
            except Exception as e:
                print(f"An error occurred during the DashScope API call or processing in {data_folder}: {e}")
                write_text_to_file(output_csv_path, f"SCRIPT_ERROR_{type(e).__name__}")

# Main loop to process L0001 to L3316
for i in range(1, 3317):
    folder_name = f"L{i:04d}"
    folder_path = os.path.join(base_dir, folder_name)
    if os.path.exists(folder_path):
        process_folder(folder_path, folder_name)
    else:
        print(f"Folder {folder_name} does not exist, skipping")