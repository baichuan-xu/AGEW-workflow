import csv
import os
import json

def check_group_file(directory_path, filename, column_index, label):
    """
    Checks if the difference between the max and min values in a specified
    column of a CSV file is less than 30.
    """
    filepath = os.path.join(directory_path, filename)
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            values = [float(row[column_index]) for row in reader if row and len(row) > column_index]
        
        if not values or len(values) < 2:
            return True

        if (max(values) - min(values)) >= 30:
            print(f"  - Validation failed for '{filename}': Difference in {label} is not less than 30.")
            return False
        
        return True

    except FileNotFoundError:
        print(f"  - Error: '{filename}' not found.")
        return False
    except (ValueError, IndexError) as e:
        print(f"  - Error processing '{filename}': {e}")
        return False

def get_fit_parameters(directory_path, filename="axis_fit.csv"):
    """
    Extracts a1, b1, a2, b2 from axis_fit.csv.
    """
    filepath = os.path.join(directory_path, filename)
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            lines = list(reader)
        
        if len(lines) < 5:
            print(f"  - Validation failed for '{filename}': Not enough rows to get all parameters.")
            return None, None, None, None
            
        a1 = float(lines[1][1])
        b1 = float(lines[2][1])
        a2 = float(lines[3][1])
        b2 = float(lines[4][1])
        
        return a1, b1, a2, b2

    except FileNotFoundError:
        print(f"  - Error: '{filename}' not found.")
        return None, None, None, None
    except (ValueError, IndexError):
        print(f"  - Validation failed for '{filename}': one or more parameters are not valid numbers.")
        return None, None, None, None

def get_labels(directory_path, filename="image_label.csv"):
    """
    Extracts x_label and y_label from image_label.csv.
    """
    filepath = os.path.join(directory_path, filename)
    x_label = "N/A"
    y_label = "N/A"
    
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            lines = list(reader)
    except FileNotFoundError:
        print(f"  - Error: '{filename}' not found.")
        return x_label, y_label

    for row in lines:
        if len(row) > 0 and "horizontal axis" in row[0].lower():
            if len(row) == 2 and row[1].lower().strip() != "n/a":
                x_label = row[1]
            break

    found_y_label = False
    for row in lines:
        if len(row) > 1 and "yield strength" in row[1].lower():
            y_label = row[1]
            found_y_label = True
            break
            
    if not found_y_label:
        for row in lines:
            if len(row) > 1 and "strength" in row[1].lower():
                y_label = row[1]
                break

    return x_label, y_label

def get_scatter_data(directory_path, filename="scatter_data.csv"):
    """
    Extracts data from between the ```csv and ``` markers in scatter_data.csv.
    """
    filepath = os.path.join(directory_path, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"  - Error: '{filename}' not found.")
        return [], [], []

    try:
        start_marker = "```csv"
        end_marker = "```"
        start_index = content.index(start_marker) + len(start_marker)
        end_index = content.index(end_marker, start_index)
        csv_content = content[start_index:end_index].strip()
    except ValueError:
        print(f"  - Error: Could not find ```csv and ``` markers in '{filename}'.")
        return [], [], []

    reader = csv.reader(csv_content.splitlines())
    y_values, labels, x_values = [], [], []
    
    try:
        for row in reader:
            if len(row) >= 3:
                x_values.append(float(row[0]))
                y_values.append(float(row[1]))
                labels.append(row[2])
    except (ValueError, IndexError) as e:
        print(f"  - Error parsing a data row in '{filename}': {e}")
        return [], [], []
            
    return x_values, y_values, labels

def process_directory(directory_path):
    """
    Main function to orchestrate the data processing and file creation for a single directory.
    """
    print(f"  - Processing directory: {directory_path}")
    # Step 1: Validate group files
    if not check_group_file(directory_path, "y_group.csv", 1, "column 2"):
        print(f"  - wrong (y_group validation failed)")
        return
    if not check_group_file(directory_path, "x_group.csv", 2, "column 3"):
        print(f"  - wrong (x_group validation failed)")
        return

    # Step 2: Get parameters from axis_fit.csv
    a1, b1, a2, b2 = get_fit_parameters(directory_path)
    if a1 is None:
        print(f"  - wrong (fit parameters validation failed)")
        return

    # Step 3: Get labels from image_label.csv
    x_label, y_label = get_labels(directory_path)

    # Step 4: Get data from scatter_data.csv
    x_values, y_values, labels = get_scatter_data(directory_path)
    if not x_values:
        print(f"  - No data extracted from scatter_data.csv. Cannot create final file.")
        return
        
    N = len(x_values)

    # Step 5: Create the final CSV file
    output_filename = os.path.join(directory_path, "image_data_final.csv")
    header = ["Horizontal Axis Label", "Horizontal Axis Value", "Data Point Label", 
              "Vertical Axis Label", "Vertical Axis Value"]
              
    try:
        with open(output_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            for i in range(N):
                transformed_y = y_values[i] * a1 + b1
                transformed_x = x_values[i] * a2 + b2
                
                new_row = [x_label, transformed_x, labels[i], y_label, transformed_y]
                writer.writerow(new_row)
        
        print(f"  - SUCCESS: Created '{output_filename}' with {N} data rows.")

    except IOError as e:
        print(f"  - FAILED: Error creating final file '{output_filename}': {e}")

def main():
    """
    Main function to traverse directories and orchestrate the data processing.
    """
    base_path = r".\HEA_finetune\Yield_Strength"
    
    for i in range(1, 3317):
        l_folder_name = f"L{i:04d}"
        l_folder_path = os.path.join(base_path, l_folder_name)
        
        if not os.path.isdir(l_folder_path):
            continue
        
        for ys_folder_name in os.listdir(l_folder_path):
            if ys_folder_name.startswith("YS") and os.path.isdir(os.path.join(l_folder_path, ys_folder_name)):
                ys_folder_path = os.path.join(l_folder_path, ys_folder_name)
                
                for data_folder_name in os.listdir(ys_folder_path):
                    if data_folder_name.startswith("data") and os.path.isdir(os.path.join(ys_folder_path, data_folder_name)):
                        data_folder_path = os.path.join(ys_folder_path, data_folder_name)
                        print(f"\nChecking folder: {data_folder_path}")

                        # Condition 4: Check check_data.txt
                        check_data_path = os.path.join(data_folder_path, 'check_data.txt')
                        try:
                            with open(check_data_path, 'r', encoding='utf-8') as f:
                                first_line = f.readline().strip()
                            if first_line != "No Numerical annotation":
                                print("  - SKIPPING: 'check_data.txt' first line is not 'No Numerical annotation'.")
                                continue
                        except FileNotFoundError:
                            print("  - SKIPPING: 'check_data.txt' not found.")
                            continue
                        except Exception as e:
                            print(f"  - SKIPPING: Error reading 'check_data.txt': {e}")
                            continue

                        # Conditions 5 & 6: Check graph_out_fixed.json
                        json_path = os.path.join(data_folder_path, 'graph_out_fixed.json')
                        try:
                            with open(json_path, 'r', encoding='utf-8') as f:
                                json_data = json.load(f)
                            
                            if json_data.get("x_axis_type") != "changing value":
                                print(f"  - SKIPPING: 'x_axis_type' is '{json_data.get('x_axis_type')}', not 'changing value'.")
                                continue

                            image_type = json_data.get("image_type", "").lower()
                            if not ("dot-line plot" in image_type or "scatter plot" in image_type):
                                print(f"  - SKIPPING: 'image_type' is '{image_type}', not 'dot-line plot' or 'scatter plot'.")
                                continue

                        except FileNotFoundError:
                            print(f"  - SKIPPING: 'graph_out_fixed.json' not found.")
                            continue
                        except (json.JSONDecodeError, KeyError) as e:
                            print(f"  - SKIPPING: Error reading or parsing 'graph_out_fixed.json': {e}")
                            continue
                        
                        # Condition 7: If all checks pass, run the main logic
                        process_directory(data_folder_path)


if __name__ == '__main__':
    main()
