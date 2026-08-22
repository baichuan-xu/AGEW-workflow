import argparse
import os
from pathlib import Path

# --- Configuration ---
SUBFOLDER_PREFIX = "L"
NUM_SUBFOLDERS = 3316
YS_PREFIX = "YS"
SPLIT_RANGE_PREFIX = "split_range"
SPLIT_RANGE_EXTENSION = ".csv"
IN_SPLIT_PREFIX = "in_split"
IN_SPLIT_EXTENSION = ".png"

def clean_split_files(base_path, execute=False):
    """
    Iterate through L0001 to L3316, delete split_range*.csv and in_split*.png in YS subfolders.
    """
    for i in range(1, NUM_SUBFOLDERS + 1):
        # Construct subfolder name (e.g., L0001)
        subfolder_name = f"{SUBFOLDER_PREFIX}{i:04d}"
        subfolder_path = os.path.join(base_path, subfolder_name)

        # Check if subfolder exists
        if not os.path.exists(subfolder_path):
            print(f"Subfolder {subfolder_path} does not exist. Skipping...")
            continue

        # Check for YS subfolders
        for item in os.listdir(subfolder_path):
            if item.startswith(YS_PREFIX) and os.path.isdir(os.path.join(subfolder_path, item)):
                ys_subfolder_path = os.path.join(subfolder_path, item)
                print(f"\nChecking subfolder: {ys_subfolder_path}")

                # Iterate through files in the YS subfolder
                for filename in os.listdir(ys_subfolder_path):
                    file_path = os.path.join(ys_subfolder_path, filename)

                    # Check for split_range*.csv files
                    if filename.startswith(SPLIT_RANGE_PREFIX) and filename.endswith(SPLIT_RANGE_EXTENSION):
                        if execute:
                            try:
                                os.remove(file_path)
                                print(f"Deleted {file_path}")
                            except Exception as e:
                                print(f"Error deleting {file_path}: {e}")
                        else:
                            print(f"Would delete {file_path} (dry-run)")

                    # Check for in_split*.png files
                    elif filename.startswith(IN_SPLIT_PREFIX) and filename.endswith(IN_SPLIT_EXTENSION):
                        if execute:
                            try:
                                os.remove(file_path)
                                print(f"Deleted {file_path}")
                            except Exception as e:
                                print(f"Error deleting {file_path}: {e}")
                        else:
                            print(f"Would delete {file_path} (dry-run)")

def main():
    """
    Main function to verify base path and start processing.
    """
    parser = argparse.ArgumentParser(description="Legacy cleanup utility (dry-run by default).")
    parser.add_argument("--target", required=True, help="Explicit Yield_Strength directory")
    parser.add_argument("--execute", action="store_true", help="Actually delete matched files")
    args = parser.parse_args()
    base_path = Path(args.target).expanduser().resolve()
    if not base_path.is_dir() or base_path == Path(base_path.anchor):
        parser.error("--target must resolve to an existing non-root directory")

    print(f"Processing subfolders in {base_path} (execute={args.execute})...")
    clean_split_files(str(base_path), execute=args.execute)
    print("\nCleaning complete.")

if __name__ == "__main__":
    main()
