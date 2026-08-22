import os
import json

def merge_composition_files(base_path):
    """
    Iterates through subfolders. If a folder contains both
    'composition_element_mod.json' and 'composition_mod.json', it merges them.

    The script creates 'composition2.json' based on 'composition_mod.json'.
    For any material with a 'name' that also exists in 'composition_element_mod.json',
    the item is replaced with the corrected version from 'composition_element_mod.json'.
    The final file has the same number of items as 'composition_mod.json'.
    """
    print(f"Starting to merge composition files in '{base_path}'...")
    processed_count = 0

    # 1. Iterate through the specified range of subfolders
    for i in range(1, 3317):
        folder_name = f"L{i:04d}"
        folder_path = os.path.join(base_path, folder_name)

        if not os.path.isdir(folder_path):
            continue

        # Define paths and check for both required input files
        elem_mod_path = os.path.join(folder_path, "composition_value_mod.json")
        mod_path = os.path.join(folder_path, "composition2.json")

        if not os.path.isfile(elem_mod_path) or not os.path.isfile(mod_path):
            continue

        print(f"Processing folder: {folder_path}")

        try:
            # Open and load the two source JSON files
            with open(elem_mod_path, 'r', encoding='utf-8') as f:
                elem_mod_data = json.load(f) # The "correct" items
            with open(mod_path, 'r', encoding='utf-8') as f:
                mod_data = json.load(f)      # The "base" file with more items

            # Ensure the base structure is as expected
            if 'materials' not in elem_mod_data or 'materials' not in mod_data:
                print(f"  - Skipping, 'materials' key missing in one or both files.")
                continue

            # 1. Create a lookup dictionary from the CORRECT file for efficient searching
            correct_items_lookup = {item['name']: item for item in elem_mod_data.get('materials', [])}

            # Build the new list of materials for the output file
            new_materials = []
            # Iterate through the BASE file (composition_mod.json)
            for base_item in mod_data.get('materials', []):
                item_name = base_item.get('name')
                
                # If a corrected version exists in the lookup, use that one.
                if item_name in correct_items_lookup:
                    new_materials.append(correct_items_lookup[item_name])
                else:
                    # Otherwise, keep the original item from the base file.
                    new_materials.append(base_item)

            # 2. Structure the final data. It will have the same number of items as composition_mod.json
            final_data = {"materials": new_materials}
            
            # Write the merged results into the new composition2.json file
            output_path = os.path.join(folder_path, "composition3.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, ensure_ascii=False, indent=2)
            
            print(f"  - Successfully created {output_path}")
            processed_count += 1

        except json.JSONDecodeError as e:
            print(f"  - Skipping due to JSON decoding error: {e}")
            continue
        except Exception as e:
            print(f"  - An unexpected error occurred: {e}")
            continue

    print(f"\nScript finished. Processed {processed_count} folders.")


# --- Main execution block ---
if __name__ == "__main__":
    base_path = r".\HEA_finetune\Yield_Strength"
    merge_composition_files(base_path)
