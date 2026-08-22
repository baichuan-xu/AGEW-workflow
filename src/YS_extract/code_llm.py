import os
import glob
import json
import shutil
from openai import OpenAI

def process_subfolder(subfolder_path, sample_py_path, prompt_template, api_key):
    """
    Processes a subfolder containing description.json, copies sample.py,
    and generates extract.py using the LLM API.
    """
    # Check for description.json
    desc_json_path = os.path.join(subfolder_path, "description.json")
    if not os.path.isfile(desc_json_path):
        print(f"description.json not found in {subfolder_path}, skipping")
        return

    # Copy sample.py to the subfolder
    dest_sample_path = os.path.join(subfolder_path, "sample.py")
    try:
        shutil.copy2(sample_py_path, dest_sample_path)
        print(f"Copied sample.py to {subfolder_path}")
    except Exception as e:
        print(f"Error copying sample.py to {subfolder_path}: {e}")
        return

    # Read description.json
    try:
        with open(desc_json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"Error reading description.json in {subfolder_path}: {e}")
        return

    # Read sample.py
    try:
        with open(dest_sample_path, "r", encoding="utf-8") as f:
            sample_code = f.read()
    except Exception as e:
        print(f"Error reading sample.py in {subfolder_path}: {e}")
        return

    # Construct prompt
    prompt = f"{prompt_template}\n\n**Provided JSON Data:**\n```json\n{json.dumps(json_data, indent=2)}\n```\n\n**Provided Python Script (sample.py):**\n```python\n{sample_code}\n```"

    # Initialize OpenAI client
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    except Exception as e:
        print(f"Error initializing client for {subfolder_path}: {e}")
        return

    # Call LLM API
    try:
        completion = client.chat.completions.create(
            model="qwen-max-latest",
            messages=[
                {"role": "system", "content": "You are an expert Python script modifier specializing in image processing with OpenCV."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        
        # Get LLM output
        modified_code = completion.choices[0].message.content.strip()
        
        # Write output to extract.py
        output_path = os.path.join(subfolder_path, "extract.py")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(modified_code)
        
        print(f"Generated {output_path} successfully")
        
    except Exception as e:
        print(f"Error during LLM API call or writing output in {subfolder_path}: {e}")

def process_folders(base_path, sample_py_path, code_md_path):
    """
    Iterates through folders and processes subfolders with description.json.
    """
    # Read code.md for prompt
    try:
        with open(code_md_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
    except Exception as e:
        print(f"Error reading code.md at {code_md_path}: {e}")
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

                # Check for description.json in data* folder
                desc_json_path = os.path.join(data_folder, "description.json")
                if not os.path.isfile(desc_json_path):
                    print(f"description.json not found in {data_folder}, skipping")
                    continue

                # Find all subfolders in the data* folder
                subfolders = [f for f in glob.glob(os.path.join(data_folder, "*")) if os.path.isdir(f)]
                for subfolder in subfolders:
                    print(f"Processing subfolder: {subfolder}")
                    process_subfolder(subfolder, sample_py_path, prompt_template, api_key)

# Usage
if __name__ == "__main__":
    base_path = r".\HEA_finetune\Yield_Strength"
    sample_py_path = r".\HEA_finetune\Yield_Strength\code\sample.py"
    code_md_path = r".\HEA_finetune\Yield_Strength\Prompt\code.md"
    process_folders(base_path, sample_py_path, code_md_path)