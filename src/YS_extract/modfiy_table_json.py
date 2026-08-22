import os
import re

# Base directory containing L0001 to L3316 folders
base_path = r".\HEA_finetune\Yield_Strength"

# Function to clean table.json by keeping content between ```json and ```
def clean_table_json(folder_path, folder_name):
    print(f"\nProcessing folder: {folder_name}")
    
    # Check for table.json in the folder
    json_file = os.path.join(folder_path, "table.json")
    if not os.path.isfile(json_file):
        print(f"table.json not found in {folder_path}, skipping")
        return
    
    # Read and process table.json
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract content between ```json and ```
        match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if not match:
            print(f"No JSON block found in {json_file}, skipping")
            return
        
        json_content = match.group(1).strip()
        
        # Write cleaned content back to table.json
        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                f.write(json_content)
            print(f"Successfully cleaned and saved {json_file}")
        except Exception as e:
            print(f"Error writing to {json_file}: {e}")
    
    except Exception as e:
        print(f"Error reading {json_file}: {e}, skipping")

# Main loop to process L0001 to L3316
for i in range(1, 3317):
    folder_name = f"L{i:04d}"
    folder_path = os.path.join(base_path, folder_name)
    if os.path.exists(folder_path):
        clean_table_json(folder_path, folder_name)
    else:
        print(f"Folder {folder_name} does not exist, skipping")