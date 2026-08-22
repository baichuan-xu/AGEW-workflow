import os
import json

# Base directory containing L0001 to L3316 folders
base_path = r".\HEA_finetune\Yield_Strength"

# Function to count items under yield_strength_data in table.json
def count_yield_strength_items(folder_path, folder_name):
    global total_n
    print(f"\nProcessing folder: {folder_name}")
    
    # Check for table.json in the folder
    json_file = os.path.join(folder_path, "table.json")
    if not os.path.isfile(json_file):
        print(f"table.json not found in {folder_path}, skipping")
        return
    
    # Read and process table.json
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get the number of items under yield_strength_data
        yield_strength_data = data.get("yield_strength_data", [])
        if not isinstance(yield_strength_data, list):
            print(f"yield_strength_data is not a list in {json_file}, skipping")
            return
        
        n = len(yield_strength_data)
        print(f"File {json_file}: n = {n}")
        total_n += n
    
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in {json_file}: {e}, skipping")
    except Exception as e:
        print(f"Error processing {json_file}: {e}, skipping")

# Main loop to process L0001 to L3316
total_n = 0
for i in range(1, 3317):
    folder_name = f"L{i:04d}"
    folder_path = os.path.join(base_path, folder_name)
    if os.path.exists(folder_path):
        count_yield_strength_items(folder_path, folder_name)
    else:
        print(f"Folder {folder_name} does not exist, skipping")

# Output the total sum of n
print(f"\nTotal sum of items under yield_strength_data across all table.json files: {total_n}")