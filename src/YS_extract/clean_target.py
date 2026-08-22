import argparse
import os
import shutil
from pathlib import Path

# --- Configuration ---
SUBFOLDER_PREFIX = "L"
NUM_SUBFOLDERS = 3316
YS_PREFIX = "YS"
CSV_FILE = "target_text.csv"

def clean_ys_subfolders(base_path, execute=False):
    """
    Iterate through L0001 to L3316, check YS subfolders, and delete those without target_text.csv.
    """
    for i in range(1, NUM_SUBFOLDERS + 1):
        # Construct subfolder name (e.g., L0001)
        subfolder_name = f"{SUBFOLDER_PREFIX}{i:04d}"
        subfolder_path = os.path.join(base_path, subfolder_name)

        # Check if subfolder exists
        if not os.path.exists(subfolder_path):
            print(f"Subfolder {subfolder_path} does not exist. Skipping...")
            continue

        # Iterate through items in the subfolder
        for item in os.listdir(subfolder_path):
            # Check for subfolders starting with YS
            if item.startswith(YS_PREFIX) and os.path.isdir(os.path.join(subfolder_path, item)):
                ys_subfolder_path = os.path.join(subfolder_path, item)
                csv_path = os.path.join(ys_subfolder_path, CSV_FILE)

                # Check if target_text.csv exists
                if os.path.exists(csv_path):
                    print(f"Found {CSV_FILE} in {ys_subfolder_path}. Keeping subfolder.")
                else:
                    if not execute:
                        print(f"Would delete {ys_subfolder_path} (dry-run).")
                        continue
                    try:
                        shutil.rmtree(ys_subfolder_path)
                        print(f"Deleted {ys_subfolder_path} (no {CSV_FILE} found).")
                    except Exception as e:
                        print(f"Error deleting {ys_subfolder_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Legacy cleanup utility (dry-run by default).")
    parser.add_argument("--target", required=True, help="Explicit Yield_Strength directory")
    parser.add_argument("--execute", action="store_true", help="Actually delete matched folders")
    args = parser.parse_args()
    base_path = Path(args.target).expanduser().resolve()
    if not base_path.is_dir() or base_path == Path(base_path.anchor):
        parser.error("--target must resolve to an existing non-root directory")

    print(f"Processing subfolders in {base_path} (execute={args.execute})...")
    clean_ys_subfolders(str(base_path), execute=args.execute)
    print("Processing complete.")

if __name__ == "__main__":
    main()
