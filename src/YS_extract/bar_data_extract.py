import dashscope
import os

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

# Function to process a single subfolder under data* folder
def process_subfolder(subfolder_path, input_files, prompt_file_name, prompt_instructions, api_key_to_use):
    input_file_paths = [os.path.join(subfolder_path, f) for f in input_files]
    if not all(os.path.exists(f) for f in input_file_paths):
        print(f"One or more input files {input_files} not found in {subfolder_path}. Skipping.")
        return
    
    # Prepare API call with explicit image labeling
    messages = [
        {
            "role": "system",
            "content": "Your task is to analyze the provided image(s) and extract data points according to the prompt instructions. Each image will be labeled clearly (e.g., 'in.png' or 'out.png'). Output the results in CSV format."
        },
        {
            "role": "user",
            "content": [
                {"text": f"Image 'in.png': This is the primary graph image containing data points to analyze."},
                {"image": f"file://{os.path.abspath(input_file_paths[0])}"}
            ] + (
                [
                    {"text": f"Image 'out.png': This image contains additional point markings to assist with data extraction."},
                    {"image": f"file://{os.path.abspath(input_file_paths[1])}"}
                ] if len(input_file_paths) > 1 else []
            ) + [
                {"text": f"Please analyze the provided images according to the instructions in '{prompt_file_name}' to extract data points. "
                         f"Use 'in.png' as the primary graph and 'out.png' (if provided) for additional point markings. "
                         f"Output a CSV with the extracted coordinates and their validity:\n\n{prompt_instructions}"}
            ]
        }
    ]
    
    print(f"Sending request to DashScope LLM for {subfolder_path} with images {input_files} and prompt {prompt_file_name}...")
    try:
        response = dashscope.MultiModalConversation.call(
            api_key=api_key_to_use,
            model='qwen2.5-vl-72b-instruct',
            messages=messages,
            temperature=0.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            vl_high_resolution_images=True
        )
        
        output_csv_path = os.path.join(subfolder_path, "bar_data.csv")
        
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
        print(f"An error occurred during the DashScope API call or processing in {subfolder_path}: {e}")
        write_text_to_file(output_csv_path, f"SCRIPT_ERROR_{type(e).__name__}")

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
            
            # Check for description.json in data* folder
            json_file_path = os.path.join(data_folder_path, "description.json")
            if not os.path.exists(json_file_path):
                print(f"description.json not found in {data_folder_path}. Skipping.")
                continue
            
            # Read check_data.txt from data* folder
            check_data_path = os.path.join(data_folder_path, "check_data.txt")
            check_data_content = read_text_file(check_data_path)
            if check_data_content is None:
                print(f"check_data.txt could not be read in {data_folder_path}. Skipping.")
                continue
            
            # Determine input files and prompt based on check_data.txt content
            if check_data_content == "No Numerical annotation":
                input_files = ["in.png", "out.png"]
                prompt_file_name = "Bar_extract_point.md"
            elif check_data_content == "Numerical annotation":
                input_files = ["in.png"]
                prompt_file_name = "Bar_extract_point2.md"
            else:
                print(f"Invalid content in check_data.txt: {check_data_content}. Skipping.")
                continue
            
            # Read prompt file
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
                write_text_to_file(os.path.join(data_folder_path, "bar_data.csv"), "ERROR_NO_API_KEY")
                continue
            
            # Find subfolders under data* folder
            subfolders = [f for f in os.listdir(data_folder_path) if os.path.isdir(os.path.join(data_folder_path, f))]
            if not subfolders:
                print(f"No subfolders found in {data_folder_path}. Skipping.")
                continue
            
            # Process each subfolder
            for subfolder in subfolders:
                subfolder_path = os.path.join(data_folder_path, subfolder)
                print(f"Processing subfolder: {subfolder}")
                process_subfolder(subfolder_path, input_files, prompt_file_name, prompt_instructions, api_key_to_use)

# Main loop to process L0001 to L3316
for i in range(1, 3317):
    folder_name = f"L{i:04d}"
    folder_path = os.path.join(base_dir, folder_name)
    if os.path.exists(folder_path):
        process_folder(folder_path, folder_name)
    else:
        print(f"Folder {folder_name} does not exist. Skipping.")