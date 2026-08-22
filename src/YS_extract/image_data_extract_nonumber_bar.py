import csv
import sys
import io
from pathlib import Path

# ==============================================================================
#  TASK: Reusable processing function (This part remains unchanged and correct)
# ==============================================================================

def process_single_directory(target_path: Path):
    """
    Performs all file processing tasks for a single valid directory.
    This is the logic from the previous request, now in a function.

    Args:
        target_path (Path): The path to the directory (e.g., '.../data_xxxx')
                            containing the input files.

    Returns:
        bool: True if processing was successful, False otherwise.
    """
    print(f"\n{'='*20}\n[START] Processing Directory: {target_path}\n{'='*20}")

    # --- Task 1: Validate y_group.csv ---
    try:
        y_group_file = target_path / 'y_group.csv'
        with open(y_group_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            previous_value = None
            for i, row in enumerate(reader):
                if len(row) < 2: continue
                try:
                    current_value = float(row[1])
                    if previous_value is not None and abs(current_value - previous_value) > 30:
                        print(f"[FAIL] 'wrong' - Difference > 30 in {y_group_file}")
                        return False
                    previous_value = current_value
                except (ValueError, IndexError):
                    print(f"[FAIL] 'wrong' - Non-numeric value in {y_group_file.name} (row {i+2})")
                    return False
    except FileNotFoundError:
        print(f"[FAIL] 'wrong' - File not found: {y_group_file}")
        return False

    # --- Task 2: Extract a1 and b1 from axis_fit.csv ---
    a1, b1 = None, None
    try:
        axis_fit_file = target_path / 'axis_fit.csv'
        with open(axis_fit_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            data_rows = list(reader)[1:] # Skip header, get all data rows
            if len(data_rows) < 2 or len(data_rows[0]) < 2 or len(data_rows[1]) < 2:
                 raise ValueError("File does not have enough rows or columns.")
            a1 = float(data_rows[0][1])
            b1 = float(data_rows[1][1])
    except (FileNotFoundError, ValueError, IndexError) as e:
        print(f"[FAIL] 'wrong' - Error processing {axis_fit_file.name}: {e}")
        return False

    # --- Task 3: Extract labels from image_label.csv ---
    x_label, y_label = "N/A", "N/A"
    temp_strength_label = None
    image_label_file = target_path / 'image_label.csv'
    try:
        with open(image_label_file, 'r', newline='', encoding='utf-8') as f:
            rows = list(csv.reader(f))
            for row in rows:
                if not row: continue
                if row[0].strip() == "horizontal axis":
                    x_label = row[1].strip() if len(row) == 2 and row[1].strip() != "N/A" else "N/A"
                    break
            for row in rows:
                if len(row) < 2: continue
                col2_lower = row[1].lower()
                if "yield strength" in col2_lower:
                    y_label = row[1].strip()
                    break
                if "strength" in col2_lower and not temp_strength_label:
                    temp_strength_label = row[1].strip()
            if y_label == "N/A" and temp_strength_label:
                y_label = temp_strength_label
    except FileNotFoundError:
        print(f"[INFO] {image_label_file.name} not found. Labels will be 'N/A'.")

    # --- Task 4: Process all bar_data.csv files in subfolders ---
    all_x, all_y, all_labels = [], [], []
    subfolders = [p for p in target_path.iterdir() if p.is_dir()]
    for folder in subfolders:
        bar_data_file = folder / 'bar_data.csv'
        if bar_data_file.exists():
            try:
                content = bar_data_file.read_text(encoding='utf-8')
                if "```csv" in content and "```" in content:
                    start_marker, end_marker = "```csv\n", "```"
                    start_index = content.find(start_marker) + len(start_marker)
                    end_index = content.find(end_marker, start_index)
                    csv_data_str = content[start_index:end_index].strip()
                    csvfile = io.StringIO(csv_data_str)
                    reader = csv.reader(csvfile)
                    for row in reader:
                        if len(row) >= 4:
                            all_y.append(row[1].strip())
                            all_labels.append(row[2].strip())
                            all_x.append(row[3].strip())
            except Exception as e:
                print(f"[WARN] Error parsing {bar_data_file}: {e}")
    
    N = len(all_x)
    if N == 0:
        print("[WARN] No valid data found in any bar_data.csv files. Final CSV will be empty.")

    # --- Task 5: Create the final CSV file ---
    output_filename = target_path / 'image_data_final.csv'
    header = ["Horizontal Axis Label", "Horizontal Axis Value", "Data Point Label", "Vertical Axis Label", "Vertical Axis Value"]
    try:
        with open(output_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            processed_rows = 0
            for i in range(N):
                try:
                    y_final = (float(all_y[i]) * a1) + b1
                    writer.writerow([x_label, all_x[i], all_labels[i], y_label, y_final])
                    processed_rows += 1
                except ValueError:
                    print(f"[WARN] Could not convert y-value '{all_y[i]}' to a number. Skipping row.")
        
        print(f"[SUCCESS] Created '{output_filename.name}' with {processed_rows} data rows.")
        return True
    except IOError as e:
        print(f"[FAIL] 'wrong' - Could not write to output file: {e}")
        return False

# ==============================================================================
#  MAIN SCRIPT: Corrected Batch processing logic
# ==============================================================================

def run_batch_processing():
    main_path = Path(r".\HEA_finetune\Yield_Strength")

    if not main_path.is_dir():
        print(f"Error: Base directory not found at '{main_path}'")
        return

    processed_count = 0
    skipped_count = 0
    
    print(f"Starting batch process in: {main_path}\n")

    for i in range(1, 3317):
        l_folder_name = f"L{i:04d}"
        l_folder_path = main_path / l_folder_name
        if not l_folder_path.is_dir(): continue

        ys_folders = [p for p in l_folder_path.glob('YS*') if p.is_dir()]
        if not ys_folders: continue

        for ys_folder_path in ys_folders:
            # 4. Find ALL subfolders starting with "data"
            data_folders = [p for p in ys_folder_path.glob('data*') if p.is_dir()]
            if not data_folders:
                continue # Skip this YS folder if it has no data subfolders

            # <-- CRITICAL CHANGE: Loop through ALL found data folders
            for data_folder_path in data_folders:
                
                # 5. Check for description.json for THIS specific data folder
                if not (data_folder_path / 'description.json').exists():
                    print(f"[SKIP] {data_folder_path.relative_to(main_path)}: Missing description.json")
                    skipped_count += 1
                    continue # Move to the next data folder in the list (e.g., from data1 to data2)

                # 6. Check content of check_data.txt
                check_data_file = data_folder_path / 'check_data.txt'
                try:
                    if not check_data_file.is_file():
                        print(f"[SKIP] {data_folder_path.relative_to(main_path)}: Missing check_data.txt")
                        skipped_count += 1
                        continue

                    with open(check_data_file, 'r', encoding='utf-8') as f:
                        first_line = f.readline().strip()

                    if first_line != "No Numerical annotation":
                        print(f"[SKIP] {data_folder_path.relative_to(main_path)}: check_data.txt content is not 'No Numerical annotation'.")
                        skipped_count += 1
                        continue
                except Exception as e:
                    print(f"[SKIP] {data_folder_path.relative_to(main_path)}: Could not read check_data.txt. Error: {e}")
                    skipped_count += 1
                    continue
                
                # 7. If all conditions are met, run the processing function
                print(f"[MATCH] Conditions met for: {data_folder_path.relative_to(main_path)}")
                if process_single_directory(data_folder_path):
                    processed_count += 1
                else:
                    skipped_count += 1

    print("\n--- Batch Processing Summary ---")
    print(f"Total Directories Processed Successfully: {processed_count}")
    print(f"Total Directories Skipped (or Failed):   {skipped_count}")
    print("---------------------------------")


if __name__ == "__main__":
    run_batch_processing()