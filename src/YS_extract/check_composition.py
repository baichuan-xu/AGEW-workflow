import os
import json

def check_elements_in_composition(base_path):
    """
    Scans subfolders for 'composition_mod.json'. For each file, it checks
    if the 'composition' field contains keys that are not valid chemical
    element symbols. If any are found, it creates a
    'composition_element_error.txt' file that lists each material and its
    corresponding non-elemental components.
    """
    # A comprehensive set of chemical element symbols for validation
    element_symbols = {
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si',
        'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co',
        'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y',
        'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
        'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu',
        'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re',
        'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr',
        'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es',
        'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
        'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
    }
    
    print(f"Scanning folders in '{base_path}' for non-elemental composition keys...")
    found_folders_count = 0

    # Iterate through the specified range of subfolders
    for i in range(1, 3317):
        folder_name = f"L{i:04d}"
        folder_path = os.path.join(base_path, folder_name)

        if not os.path.isdir(folder_path):
            continue

        # Check for composition_mod.json in the current subfolder
        json_path = os.path.join(folder_path, "composition_mod.json")
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

            # List to hold the detailed error strings for the entire file
            file_error_lines = []

            # Iterate through each material in the file
            for item in records_to_check:
                if "composition" in item and isinstance(item["composition"], dict):
                    
                    # List to hold non-element keys for the current material
                    wrong_elements = []
                    for element_key in item["composition"].keys():
                        if element_key not in element_symbols:
                            wrong_elements.append(element_key)
                    
                    # If any wrong elements were found for this material, format the error string
                    if wrong_elements:
                        material_name = item.get("name", "Unknown Material")
                        # Join the list of wrong elements with a semicolon and space
                        wrong_elements_str = "; ".join(wrong_elements)
                        error_line = f"{material_name} contain wrong element: {wrong_elements_str}"
                        file_error_lines.append(error_line)

            # After checking all materials, if any errors were found, write the error file
            if file_error_lines:
                print(f"Non-elemental components found in: {folder_path}")
                found_folders_count += 1
                
                # (1) Create a new composition_element_error.txt file
                output_file_path = os.path.join(folder_path, "composition_element_error.txt")
                
                try:
                    # (2) & (3) Write the detailed error messages to the file
                    with open(output_file_path, 'w', encoding='utf-8') as f_out:
                        f_out.write("\n".join(file_error_lines))
                except Exception as e:
                    print(f"  - Could not write to {output_file_path}: {e}")

        except (json.JSONDecodeError, TypeError):
            continue
        except Exception as e:
            # print(f"An error occurred in {folder_path}: {e}")
            continue
            
    if found_folders_count == 0:
        print("\nScan complete. No folders with non-elemental components were found.")
    else:
        print(f"\nScan complete. Found {found_folders_count} folder(s) with non-elemental components.")


# --- Main execution block ---
if __name__ == "__main__":
    base_path = r".\HEA_finetune\Yield_Strength"
    check_elements_in_composition(base_path)
