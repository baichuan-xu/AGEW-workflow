import csv
import numpy as np
import os

def find_best_linear_fit_group(data, filter_col_idx, x_col_idx, y_col_idx, iterations=100, threshold=1.0):
    """
    Finds the largest subset of data that fits a linear model using a RANSAC-like approach,
    after correctly pre-filtering the data.

    Args:
        data (list of list of float): The full dataset.
        filter_col_idx (int): The index of the column to use for the initial grouping (e.g., 1 for Y-group, 2 for X-group).
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
    # This is the corrected logic to ensure the primary rule is met.
    
    # Sort the data based on the column we are filtering by.
    sorted_data = sorted(data, key=lambda row: row[filter_col_idx])
    
    best_group_start_idx = 0
    max_group_size = 0
    
    # Use a two-pointer/sliding window approach to find the largest group of rows
    # where the difference in the filter column is less than 30.
    left_ptr = 0
    for right_ptr in range(len(sorted_data)):
        # While the window is too large (>= 30), shrink it from the left.
        while sorted_data[right_ptr][filter_col_idx] - sorted_data[left_ptr][filter_col_idx] >= 30:
            left_ptr += 1
            
        # Check if the current window [left_ptr, right_ptr] is the largest valid one we've seen.
        current_size = right_ptr - left_ptr + 1
        if current_size > max_group_size:
            max_group_size = current_size
            best_group_start_idx = left_ptr
    
    # This is the largest group of rows that correctly satisfies the filter condition.
    initial_group = sorted_data[best_group_start_idx : best_group_start_idx + max_group_size]

    if len(initial_group) < 2:
        # No group of at least 2 points satisfies the primary condition.
        return [], None, None

    # --- Step 2: RANSAC to find the best linear fit WITHIN the pre-filtered group ---
    best_inliers_indices = []
    best_model = (None, None) # (slope, intercept)
    
    # Extract the x and y coordinates for fitting from the pre-filtered group.
    points = np.array([[row[x_col_idx], row[y_col_idx]] for row in initial_group])

    # The RANSAC iterations now operate ONLY on the correctly filtered `initial_group`.
    for _ in range(iterations):
        # Randomly select 2 points to form a line
        sample_indices = np.random.choice(len(initial_group), 2, replace=False)
        p1, p2 = points[sample_indices[0]], points[sample_indices[1]]
        
        # Fit a line to these two points: Ax + By + C = 0
        A = p2[1] - p1[1]
        B = p1[0] - p2[0]
        C = -A * p1[0] - B * p1[1]
        
        # Avoid degenerate lines
        if A**2 + B**2 == 0:
            continue

        # Find all "inliers" - points that are close to this line
        distances = np.abs(A * points[:, 0] + B * points[:, 1] + C) / np.sqrt(A**2 + B**2)
        
        inlier_candidates_indices = np.where(distances < threshold)[0]

        # Check if the found inliers form a better model
        if len(inlier_candidates_indices) > len(best_inliers_indices):
            inlier_points = points[inlier_candidates_indices]
            if len(inlier_points) > 1:
                # Refit the line using ALL the inliers for better accuracy
                x_coords = inlier_points[:, 0]
                y_coords = inlier_points[:, 1]
                m, c = np.polyfit(x_coords, y_coords, 1) # y = mx + c
                
                best_inliers_indices = inlier_candidates_indices
                best_model = (m, c)

    # Return the actual rows corresponding to the best inlier set and the model
    final_group = [initial_group[i] for i in best_inliers_indices]
    return final_group, best_model[0], best_model[1]

def process_directory(directory_path):
    """
    Performs the RANSAC analysis on an axis.csv file in a given directory.
    """
    input_filename = os.path.join(directory_path, 'axis.csv')
    fit_output_filename = os.path.join(directory_path, 'axis_fit.csv')
    y_group_output_filename = os.path.join(directory_path, 'y_group.csv')
    x_group_output_filename = os.path.join(directory_path, 'x_group.csv')
    
    # --- Read the data from axis.csv ---
    try:
        with open(input_filename, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            data = [list(map(float, row)) for row in reader if row]
    except FileNotFoundError:
        print(f"  - SKIPPING: '{input_filename}' not found.")
        return
    except (ValueError, IndexError) as e:
        print(f"  - SKIPPING: Error reading '{input_filename}': {e}")
        return

    # --- Find the Y Group ---
    y_group, a1, b1 = find_best_linear_fit_group(data, filter_col_idx=1, x_col_idx=2, y_col_idx=0)
    
    # --- Find the X Group ---
    x_group, a2, b2 = find_best_linear_fit_group(data, filter_col_idx=2, x_col_idx=1, y_col_idx=0)

    # --- Create the fit.csv file with the results ---
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
        
    # --- Create the y_group.csv file ---
    if y_group:
        try:
            with open(y_group_output_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header) # Write original header
                writer.writerows(y_group)
            print(f"  - SUCCESS: Created '{y_group_output_filename}' with {len(y_group)} rows.")
        except IOError as e:
            print(f"  - FAILED: Could not write to '{y_group_output_filename}': {e}")
            
    # --- Create the x_group.csv file ---
    if x_group:
        try:
            with open(x_group_output_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header) # Write original header
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
        
        # Find subfolders starting with 'YS'
        for ys_folder_name in os.listdir(l_folder_path):
            if ys_folder_name.startswith("YS") and os.path.isdir(os.path.join(l_folder_path, ys_folder_name)):
                ys_folder_path = os.path.join(l_folder_path, ys_folder_name)
                
                # Find subfolder starting with 'data'
                for data_folder_name in os.listdir(ys_folder_path):
                    if data_folder_name.startswith("data") and os.path.isdir(os.path.join(ys_folder_path, data_folder_name)):
                        data_folder_path = os.path.join(ys_folder_path, data_folder_name)
                        print(f"Checking folder: {data_folder_path}")

                        # Check for image_label.csv and its line count
                        image_label_path = os.path.join(data_folder_path, 'image_label.csv')
                        
                        if not os.path.exists(image_label_path):
                            print("  - SKIPPING: 'image_label.csv' not found.")
                            continue

                        try:
                            with open(image_label_path, 'r', encoding='utf-8', errors='ignore') as f:
                                lines = f.readlines()
                            
                            if len(lines) == 4:
                                # Condition met, process this directory
                                print("  - Condition met (4 lines found). Processing...")
                                process_directory(data_folder_path)
                            else:
                                print(f"  - SKIPPING: Found {len(lines)} lines in 'image_label.csv' (expected 4).")
                        except Exception as e:
                            print(f"  - SKIPPING: Could not read 'image_label.csv'. Error: {e}")

if __name__ == '__main__':
    main()
