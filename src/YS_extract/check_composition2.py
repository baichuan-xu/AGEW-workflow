import os
import json

def validate_composition_sum(base_path):
    """
    Scans subfolders for 'composition_mod.json'. For each file, it checks if
    the sum of values in any 'composition' dictionary minus 100 is greater
    than 5 (i.e., sum > 105). If it is, it creates a
    'composition_value_error.txt' file listing all materials with errors.
    Skips folders that already contain a 'composition_result.txt' file.

    Args:
        base_path (str): The root directory containing L0001 to L3316 folders.
    """
    print(f"Scanning folders in '{base_path}' for composition sums greater than 105...")
    found_folders_count = 0

    # 1. Iterate through the specified range of subfolders
    for i in range(1, 3317):
        folder_name = f"L{i:04d}"
        folder_path = os.path.join(base_path, folder_name)

        if not os.path.isdir(folder_path):
            continue

        # 2. Check for composition_mod.json
        json_path = os.path.join(folder_path, "composition2.json")
        if not os.path.isfile(json_path):
            continue
    

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            records_to_check = []
            if isinstance(data, list):
                records_to_check = data
            elif isinstance(data, dict) and 'materials' in data and isinstance(data['materials'], list):
                records_to_check = data['materials']
            else:
                continue

            # A list to hold the names of materials with errors in this file
            error_lines = []

            # Iterate through each record in the JSON array to find all errors
            for item in records_to_check:
                if "composition" in item and isinstance(item["composition"], dict):
                    composition_values = item["composition"].values()
                    
                    if not all(isinstance(v, (int, float)) for v in composition_values):
                        continue

                    total_sum = sum(composition_values)
                    
                    # 1. Check if the sum minus 100 is greater than 5
                    if (total_sum - 100.0) > 5.0:
                        material_name = item.get("name", "Unknown Material")
                        # 2. Modify the error line to include the sum
                        error_line = f"{material_name}: composition value sum ({total_sum}) is wrong (>100)"
                        error_lines.append(error_line)
            
            # After checking all items, if any errors were found, write them to a file
            if error_lines:
                print(f"Error found in: {folder_path}")
                found_folders_count += 1
                output_file_path = os.path.join(folder_path, "composition_value_error.txt")
                
                # Create the error file and write all found errors to it
                try:
                    with open(output_file_path, 'w', encoding='utf-8') as f_out:
                        f_out.write("\n".join(error_lines))
                except Exception as e:
                    print(f"  - Could not write to {output_file_path}: {e}")

        except (json.JSONDecodeError, TypeError):
            continue
        except Exception as e:
            # print(f"An error occurred in {folder_path}: {e}")
            continue
            
    if found_folders_count == 0:
        print("\nScan complete. No folders with composition sum errors (>105) were found.")
    else:
        print(f"\nScan complete. Found {found_folders_count} folder(s) with composition sum errors.")


# --- Main execution block ---
if __name__ == "__main__":
    # Define the base path containing L0001 to L3316 folders
    base_path = r".\HEA_finetune\Yield_Strength"
    validate_composition_sum(base_path)
