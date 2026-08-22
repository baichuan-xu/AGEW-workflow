import os
import re
import json

def process_composition_json(base_path):
    """
    Iterates through folders L0001 to L3316, finds composition.json,
    extracts the clean JSON content from markdown blocks, and saves it
    to a new file named composition_mod.json.
    """
    # 1. Iterate through the specified range of subfolders
    for i in range(1, 3317):
        folder_name = f"L{i:04d}"
        folder_path = os.path.join(base_path, folder_name)

        if not os.path.isdir(folder_path):
            # print(f"Folder {folder_path} does not exist, skipping.")
            continue

        # 2. Check for composition.json in the current subfolder
        source_file_path = os.path.join(folder_path, "composition.json")
        if not os.path.isfile(source_file_path):
            # print(f"composition.json not found in {folder_path}, skipping.")
            continue

        print(f"Processing file: {source_file_path}")

        # Process the found composition.json file
        try:
            with open(source_file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 3. Keep only the content between "```json" and "```"
            match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if not match:
                print(f"No JSON markdown block found in {source_file_path}, skipping.")
                continue

            # Extract the raw string of the JSON content
            json_content_string = match.group(1).strip()

            # Validate that the extracted string is valid JSON
            try:
                json_data = json.loads(json_content_string)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON content in {source_file_path}: {e}, skipping.")
                continue

            # 3. Write the retained result to a new file: composition_mod.json
            output_file_path = os.path.join(folder_path, "composition_mod.json")
            try:
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    # Use json.dump for pretty-printing and ensuring correctness
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                print(f"Successfully created {output_file_path}")
            except Exception as e:
                print(f"Error writing to {output_file_path}: {e}")

        except Exception as e:
            print(f"Error reading or processing {source_file_path}: {e}")

# --- Main execution block ---
if __name__ == "__main__":
    # 1. Define the base path containing L0001 to L3316 folders
    base_path = r".\HEA_finetune\Yield_Strength"
    process_composition_json(base_path)
    print("\nScript finished.")