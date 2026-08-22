import json
import re
import os
import glob

def fix_malformed_json(input_file, output_file):
    try:
        # Read the malformed JSON file
        with open(input_file, 'r') as file:
            content = file.read()

        print(f"Processing {input_file}")
        print(f"Raw content:\n{content}\n")

        # Step 1: Remove outer quotes if present
        cleaned_content = content.strip()
        if cleaned_content.startswith('"') and cleaned_content.endswith('"'):
            cleaned_content = cleaned_content[1:-1]

        # Step 2: Remove markdown code fences
        cleaned_content = re.sub(r'^```json\s*', '', cleaned_content, flags=re.MULTILINE)
        cleaned_content = re.sub(r'```\s*$', '', cleaned_content, flags=re.MULTILINE)

        # Step 3: Replace escaped newlines with actual newlines
        cleaned_content = cleaned_content.replace('\\n', '\n')

        # Step 4: Remove backslashes before quotes in property names and values
        # Replace \" with " globally, but preserve \" within string values that are intentionally escaped
        cleaned_content = re.sub(r'(?<!\\)\\"', '"', cleaned_content)

        # Step 5: Strip excessive whitespace and validate JSON start
        cleaned_content = cleaned_content.strip()
        print(f"Cleaned content:\n{cleaned_content}\n")

        if not cleaned_content.startswith('{'):
            raise ValueError("Content does not start with a valid JSON object")

        # Step 6: Parse the cleaned content to ensure it's valid JSON
        json_data = json.loads(cleaned_content)

        # Step 7: Write the properly formatted JSON to a new file
        with open(output_file, 'w') as file:
            json.dump(json_data, file, indent=4)

        print(f"Successfully converted {input_file} to standard JSON format and saved as {output_file}")

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {input_file} - {e}")
        print(f"Cleaned content was: {cleaned_content}")
    except FileNotFoundError:
        print(f"Error: File {input_file} not found")
    except ValueError as e:
        print(f"Error in {input_file}: {e}")
    except Exception as e:
        print(f"Unexpected error in {input_file}: {e}")

def process_folders(base_path):
    # Iterate through folders L0001 to L3316
    for i in range(1, 3317):
        folder_name = f"L{i:04d}"  # Format as L0001, L0002, ..., L3316
        folder_path = os.path.join(base_path, folder_name)

        if not os.path.isdir(folder_path):
            print(f"Folder {folder_path} does not exist, skipping")
            continue

        print(f"Checking folder: {folder_path}")

        # Check for subfolders starting with YS
        ys_folders = glob.glob(os.path.join(folder_path, "YS*"))
        if not ys_folders:
            print(f"No YS* folders found in {folder_path}, skipping")
            continue

        for ys_folder in ys_folders:
            if not os.path.isdir(ys_folder):
                print(f"{ys_folder} is not a directory, skipping")
                continue

            print(f"Found YS folder: {ys_folder}")

            # Check for subfolders starting with data
            data_folders = glob.glob(os.path.join(ys_folder, "data*"))
            if not data_folders:
                print(f"No data* folders found in {ys_folder}, skipping")
                continue

            for data_folder in data_folders:
                if not os.path.isdir(data_folder):
                    print(f"{data_folder} is not a directory, skipping")
                    continue

                print(f"Found data folder: {data_folder}")

                # Look for graph_out.json in the data folder
                input_file = os.path.join(data_folder, "graph_out.json")
                output_file = os.path.join(data_folder, "graph_out_fixed.json")

                if not os.path.isfile(input_file):
                    print(f"graph_out.json not found in {data_folder}, skipping")
                    continue

                # Process the JSON file
                fix_malformed_json(input_file, output_file)

# Usage
if __name__ == "__main__":
    base_path = r".\HEA_finetune\Yield_Strength"
    process_folders(base_path)