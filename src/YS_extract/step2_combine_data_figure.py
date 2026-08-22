import os
import json

# Define the root directory
root_dir = r".\HEA_finetune\Yield_Strength"

def process_json_files(subfolder_path):
    try:
        # Check for data+composition_mod2_mod.json
        mod2_json_path = os.path.join(subfolder_path, "data+composition_mod2_mod.json")
        if not os.path.isfile(mod2_json_path):
            return
        
        # Check for data+composition_mod.json
        mod_json_path = os.path.join(subfolder_path, "data+composition_mod.json")
        if not os.path.isfile(mod_json_path):
            return
        
        # Read data+composition_mod2_mod.json
        with open(mod2_json_path, 'r', encoding='utf-8') as f:
            mod2_data = json.load(f)
        
        # Read data+composition_mod.json
        with open(mod_json_path, 'r', encoding='utf-8') as f:
            mod_data = json.load(f)
        
        # Check if yield_strength_data exists in both files
        if 'yield_strength_data' not in mod2_data or 'yield_strength_data' not in mod_data:
            return
        
        # Create a dictionary for mod2 items by material_name
        mod2_items = {item['material_name']: item for item in mod2_data['yield_strength_data'] 
                      if 'material_name' in item}
        
        # Replace items in mod_data with matching material_name from mod2_data
        updated_items = []
        for mod_item in mod_data['yield_strength_data']:
            if 'material_name' in mod_item and mod_item['material_name'] in mod2_items:
                # Replace with mod2 item
                updated_items.append(mod2_items[mod_item['material_name']])
            else:
                # Keep original item
                updated_items.append(mod_item)
        
        # Update mod_data with new items
        mod_data['yield_strength_data'] = updated_items
        
        # Write updated data back to data+composition_mod.json
        with open(mod_json_path, 'w', encoding='utf-8') as f:
            json.dump(mod_data, f, indent=2)
        
    except Exception:
        pass  # Silent on errors

def main():
    # Iterate through subfolders L0001 to L3316
    for i in range(1, 3317):
        subfolder = f"L{i:04d}"
        subfolder_path = os.path.join(root_dir, subfolder)
        
        if not os.path.isdir(subfolder_path):
            continue
        
        process_json_files(subfolder_path)

if __name__ == "__main__":
    main()