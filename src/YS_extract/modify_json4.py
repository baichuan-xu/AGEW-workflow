import os
import re
import json

def clean_last_json_block(base_path):
    """
    Iterates through subfolders to find 'composition_element.json'.
    It extracts the content from the LAST json markdown block in the file
    and saves it to 'composition_element_mod.json'.
    """
    # 1. Iterate through the specified range of subfolders
    for i in range(1, 3317):
        folder_name = f"L{i:04d}"
        folder_path = os.path.join(base_path, folder_name)

        if not os.path.isdir(folder_path):
            continue

        # 2. Check for composition_element.json in the current subfolder
        source_file_path = os.path.join(folder_path, "table_data_raw.json")
        if not os.path.isfile(source_file_path):
            continue

        print(f"Processing file: {source_file_path}")

        try:
            with open(source_file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 3. Find all non-overlapping matches for the JSON block pattern
            matches = re.findall(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            
            # If no matches are found, skip the file
            if not matches:
                print(f"  - No JSON markdown block found in {source_file_path}, skipping.")
                continue

            # Take the content of the VERY LAST match
            json_content_string = matches[-1].strip()

            # Validate that the extracted string is valid JSON
            try:
                json_data = json.loads(json_content_string)
            except json.JSONDecodeError as e:
                print(f"  - Invalid JSON in the last block of {source_file_path}: {e}, skipping.")
                continue

            # 3. Write the retained content to the new file
            output_file_path = os.path.join(folder_path, "table_data.json")
            try:
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    # Use json.dump for pretty-printing and ensuring correctness
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                print(f"  - Successfully created {output_file_path}")
            except Exception as e:
                print(f"  - Error writing to {output_file_path}: {e}")

        except Exception as e:
            print(f"  - Error reading or processing {source_file_path}: {e}")

# --- Main execution block ---
if __name__ == "__main__":
    # 1. Define the base path containing L0001 to L3316 folders
    base_path = r".\HEA_finetune\Yield_Strength"
    clean_last_json_block(base_path)
    print("\nScript finished.")
