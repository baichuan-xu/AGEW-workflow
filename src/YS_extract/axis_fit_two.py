import csv
import numpy as np
import os

def find_best_linear_fit_group(data, filter_col_idx, x_col_idx, y_col_idx, iterations=100, threshold=1.0):
    """
    Finds the largest subset of data that fits a linear model using a RANSAC-like approach,
    after correctly pre-filtering the data.

    Args:
        data (list of list of float): The dataset to be processed.
        filter_col_idx (int): The index of the column to use for the grouping (e.g., 1 for Y-group, 2 for X-group).
        x_col_idx (int): The index of the column to be treated as 'x' for the linear fit.
        y_col_idx (int): The index of the column to be treated as 'y' for the linear fit.
        iterations (int): How many random samples to try.
        threshold (float): The distance threshold for a point to be considered an "inlier".

    Returns:
        tuple: A tuple containing (list_of_rows, slope, intercept).
               Returns ([], None, None) if no suitable group is found.
    """
    if not data or len(data) < 2:
        return [], None, None

    # --- Step 1: Pre-filter data to find the largest group that meets the filter criteria ---
    sorted_data = sorted(data, key=lambda row: row[filter_col_idx])
    
    best_group_start_idx = 0
    max_group_size = 0
    
    left_ptr = 0
    for right_ptr in range(len(sorted_data)):
        while sorted_data[right_ptr][filter_col_idx] - sorted_data[left_ptr][filter_col_idx] >= 30:
            left_ptr += 1
        current_size = right_ptr - left_ptr + 1
        if current_size > max_group_size:
            max_group_size = current_size
            best_group_start_idx = left_ptr
    
    initial_group = sorted_data[best_group_start_idx : best_group_start_idx + max_group_size]

    if len(initial_group) < 2:
        return [], None, None

    # --- Step 2: RANSAC to find the best linear fit WITHIN the pre-filtered group ---
    best_inliers_indices = []
    best_model = (None, None)
    
    points = np.array([[row[x_col_idx], row[y_col_idx]] for row in initial_group])

    for _ in range(iterations):
        sample_indices = np.random.choice(len(initial_group), 2, replace=False)
        p1, p2 = points[sample_indices[0]], points[sample_indices[1]]
        
        A = p2[1] - p1[1]
        B = p1[0] - p2[0]
        C = -A * p1[0] - B * p1[1]
        
        if A**2 + B**2 == 0:
            continue

        distances = np.abs(A * points[:, 0] + B * points[:, 1] + C) / np.sqrt(A**2 + B**2)
        inlier_candidates_indices = np.where(distances < threshold)[0]

        if len(inlier_candidates_indices) > len(best_inliers_indices):
            inlier_points = points[inlier_candidates_indices]
            if len(inlier_points) > 1:
                x_coords = inlier_points[:, 0]
                y_coords = inlier_points[:, 1]
                m, c = np.polyfit(x_coords, y_coords, 1)
                
                best_inliers_indices = inlier_candidates_indices
                best_model = (m, c)

    final_group = [initial_group[i] for i in best_inliers_indices]
    return final_group, best_model[0], best_model[1]

def analyze_image_label(file_path):
    """
    Analyzes image_label.csv to find the target row and axis type.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            lines = list(reader)
    except Exception:
        return None

    if len(lines) != 5:
        return None

    rows_to_check = lines[2:5] # Lines 3, 4, 5
    
    target_row = None
    # First pass: look for "yield strength"
    for row in rows_to_check:
        if len(row) > 1 and "yield strength" in row[1].lower():
            target_row = row
            break
            
    # Second pass: if not found, look for "strength"
    if not target_row:
        for row in rows_to_check:
            if len(row) > 1 and "strength" in row[1].lower():
                target_row = row
                break

    if target_row:
        axis_type = target_row[0].lower()
        if "vertical axis (left)" in axis_type:
            return "left"
        elif "vertical axis (right)" in axis_type:
            return "right"
    
    return None

def process_directory(directory_path, y_axis_type):
    """
    Performs the RANSAC analysis based on the determined y_axis_type.
    """
    input_filename = os.path.join(directory_path, 'axis.csv')
    fit_output_filename = os.path.join(directory_path, 'axis_fit.csv')
    y_group_output_filename = os.path.join(directory_path, 'y_group.csv')
    x_group_output_filename = os.path.join(directory_path, 'x_group.csv')
    
    try:
        with open(input_filename, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            all_data = [list(map(float, row)) for row in reader if row]
    except (FileNotFoundError, ValueError, IndexError) as e:
        print(f"  - SKIPPING: Could not read or parse '{input_filename}'. Error: {e}")
        return

    # --- Y-Group Processing (Conditional) ---
    y_group, a1, b1 = [], None, None
    y_subset = []
    if y_axis_type == "left":
        y_subset = [row for row in all_data if row[1] < 1000]
    elif y_axis_type == "right":
        y_subset = [row for row in all_data if row[1] > 1000]

    if y_subset:
        y_group, a1, b1 = find_best_linear_fit_group(y_subset, filter_col_idx=1, x_col_idx=2, y_col_idx=0)
    else:
        print("  - INFO: No data met the criteria for Y-Group processing.")

    # --- X-Group Processing (Unconditional) ---
    x_group, a2, b2 = find_best_linear_fit_group(all_data, filter_col_idx=2, x_col_idx=1, y_col_idx=0)

    # --- Output Generation ---
    try:
        with open(fit_output_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Label', 'Value'])
            writer.writerow(['a1', a1 if a1 is not None else 'N/A'])
            writer.writerow(['b1', b1 if b1 is not None else 'N/A'])
            writer.writerow(['a2', a2 if a2 is not None else 'N/A'])
            writer.writerow(['b2', b2 if b2 is not None else 'N/A'])
        print(f"  - SUCCESS: Created '{fit_output_filename}'")
    except IOError as e:
        print(f"  - FAILED: Could not write to '{fit_output_filename}': {e}")
        
    if y_group:
        try:
            with open(y_group_output_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(y_group)
            print(f"  - SUCCESS: Created '{y_group_output_filename}' with {len(y_group)} rows.")
        except IOError as e:
            print(f"  - FAILED: Could not write to '{y_group_output_filename}': {e}")
            
    if x_group:
        try:
            with open(x_group_output_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(x_group)
            print(f"  - SUCCESS: Created '{x_group_output_filename}' with {len(x_group)} rows.")
        except IOError as e:
            print(f"  - FAILED: Could not write to '{x_group_output_filename}': {e}")

def main():
    """
    Main function to traverse the directory structure and trigger the analysis.
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
                        print(f"Checking folder: {data_folder_path}")

                        image_label_path = os.path.join(data_folder_path, 'image_label.csv')
                        if not os.path.exists(image_label_path):
                            print("  - SKIPPING: 'image_label.csv' not found.")
                            continue
                        
                        y_axis_type = analyze_image_label(image_label_path)
                        
                        if y_axis_type:
                            print(f"  - Condition met (5 lines, target found: '{y_axis_type}'). Processing...")
                            process_directory(data_folder_path, y_axis_type)
                        else:
                            print("  - SKIPPING: 'image_label.csv' does not meet criteria (not 5 lines or target not found).")

if __name__ == '__main__':
    main()
