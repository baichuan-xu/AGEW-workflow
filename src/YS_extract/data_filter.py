import os
import csv

# Base directory containing L0001 to L3316 folders
base_dir = r".\HEA_finetune\Yield_Strength"

def extract_min_max(file_path):
    """
    Extracts min-max values from axis_min-max.csv.
    x_min/x_max can be None if they are not valid numbers.
    y_min/y_max must be valid numbers, otherwise the function fails.
    The file has no header.
    """
    x_min, x_max, y_min, y_max = None, None, None, None
    try:
        with open(file_path, "r", encoding="utf-8") as csv_file:
            csv_reader = list(csv.reader(csv_file))
            if len(csv_reader) < 6:
                print(f"Error: {file_path} has fewer than 6 rows")
                return None, None, None, None
            
            # Try to get x-axis limits (row 3 and 4)
            try:
                x_min = float(csv_reader[2][1])
                x_max = float(csv_reader[3][1])
            except (ValueError, IndexError):
                # If x-values are text or missing, keep them as None and print a message
                print(f"Info: x_min/x_max in {file_path} are not numbers. Skipping x-axis filter.")
                x_min, x_max = None, None

            # Try to get y-axis limits (row 5 and 6)
            try:
                y_min = float(csv_reader[4][1])
                y_max = float(csv_reader[5][1])
            except (ValueError, IndexError) as e:
                # If y-values are invalid, the entire operation for this file must fail.
                print(f"Error: y_min/y_max are invalid in {file_path}: {e}")
                return None, None, None, None
            
            return x_min, x_max, y_min, y_max

    except FileNotFoundError:
        print(f"Error: {file_path} not found")
        return None, None, None, None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None, None, None

def filter_image_data(input_path, output_path, x_min, x_max, y_min, y_max):
    """
    Filters image_data_final.csv based on min-max values.
    The x-axis filter is skipped if x_min or x_max is None.
    """
    try:
        with open(input_path, "r", encoding="utf-8") as input_file:
            csv_reader = csv.reader(input_file)
            rows = list(csv_reader)
            if not rows:
                print(f"Error: {input_path} is empty")
                return
            
            header = rows[0]
            filtered_rows = [header]
            
            for row in rows[1:]:
                if len(row) < 5:
                    continue
                
                # First, try to get the y-value. If it's not a number, the row is invalid.
                try:
                    y_value = float(row[4])
                except ValueError:
                    continue

                # Check y-filter (mandatory)
                y_filter_passed = (y_min <= y_value <= y_max)
                
                # If the y-filter fails, no need to check x. Skip to the next row.
                if not y_filter_passed:
                    continue

                # Check x-filter (optional)
                # It passes by default if the x-limits are not numbers.
                x_filter_passed = (x_min is None or x_max is None)
                
                if not x_filter_passed:
                    # Only if we have numeric x-limits, do we try to check the x_value.
                    try:
                        x_value = float(row[1])
                        if x_min <= x_value <= x_max:
                            x_filter_passed = True
                    except ValueError:
                        # The x_value is text, so it cannot pass a numeric filter.
                        # x_filter_passed remains False.
                        pass

                # The row is kept only if both filters pass.
                if y_filter_passed and x_filter_passed:
                    filtered_rows.append(row)
            
            with open(output_path, "w", newline="", encoding="utf-8") as output_file:
                csv_writer = csv.writer(output_file)
                csv_writer.writerows(filtered_rows)
            print(f"Successfully created {output_path} with {len(filtered_rows)-1} filtered data rows")
            
    except FileNotFoundError:
        print(f"Error: {input_path} not found")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def process_folder(folder_path, folder_name):
    """
    Processes a single 'L' folder, looking for 'YS' and 'data' subdirectories.
    """
    print(f"\nProcessing folder: {folder_name}")
    
    ys_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f)) and f.startswith("YS")]
    
    for ys_folder in ys_folders:
        ys_folder_path = os.path.join(folder_path, ys_folder)
        
        data_folders = [f for f in os.listdir(ys_folder_path) if os.path.isdir(os.path.join(ys_folder_path, f)) and f.startswith("data")]
        if not data_folders:
            continue
        
        for data_folder in data_folders:
            data_folder_path = os.path.join(ys_folder_path, data_folder)
            print(f"--- Checking data subfolder: {data_folder_path} ---")
            
            min_max_path = os.path.join(data_folder_path, "axis_min-max.csv")
            if not os.path.exists(min_max_path):
                print(f"axis_min-max.csv not found. Skipping.")
                continue
            
            input_path = os.path.join(data_folder_path, "image_data_final.csv")
            if not os.path.exists(input_path):
                print(f"image_data_final.csv not found. Skipping.")
                continue
            
            # Extract min-max values
            x_min, x_max, y_min, y_max = extract_min_max(min_max_path)
            
            # We can only proceed if we have valid y-axis limits.
            if y_min is None or y_max is None:
                print(f"Skipping due to invalid y-axis min-max values.")
                continue
            
            output_path = os.path.join(data_folder_path, "image_data_filter.csv")
            filter_image_data(input_path, output_path, x_min, x_max, y_min, y_max)

# Main loop to process L0001 to L3316
for i in range(1, 3317):
    folder_name = f"L{i:04d}"
    folder_path = os.path.join(base_dir, folder_name)
    if os.path.exists(folder_path):
        process_folder(folder_path, folder_name)

