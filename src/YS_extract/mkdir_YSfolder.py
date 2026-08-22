import os
import shutil

# --- Configuration ---
BASE_PATH = r".\HEA_finetune\Yield_Strength"
SUBFOLDER_PREFIX = "L"
NUM_SUBFOLDERS = 3316
IMAGE_PREFIX = "YS"
IMAGE_EXTENSION = ".jpg"
OUTPUT_IMAGE_NAME = "in.png"

def process_subfolders():
    """
    Iterate through subfolders L0001 to L3316, process images starting with 'YS' and ending with '.jpg',
    create a subfolder named after the image, copy and rename the image to 'in.png'.
    """
    for i in range(1, NUM_SUBFOLDERS + 1):
        # Construct subfolder name (e.g., L0001, L0262)
        subfolder_name = f"{SUBFOLDER_PREFIX}{i:04d}"
        subfolder_path = os.path.join(BASE_PATH, subfolder_name)

        # Check if subfolder exists
        if not os.path.exists(subfolder_path):
            print(f"Subfolder {subfolder_path} does not exist. Skipping...")
            continue

        # Iterate through files in the subfolder
        for filename in os.listdir(subfolder_path):
            # Check if file starts with 'YS' and ends with '.jpg'
            if filename.startswith(IMAGE_PREFIX) and filename.lower().endswith(IMAGE_EXTENSION):
                # Extract the image name without extension (e.g., YS9 from YS9.jpg)
                image_name = os.path.splitext(filename)[0]
                image_path = os.path.join(subfolder_path, filename)

                # Create a subfolder named after the image (e.g., YS9)
                new_subfolder_path = os.path.join(subfolder_path, image_name)
                try:
                    os.makedirs(new_subfolder_path, exist_ok=True)
                except Exception as e:
                    print(f"Error creating subfolder {new_subfolder_path}: {e}")
                    continue

                # Define the destination path for the renamed image (in.png)
                destination_path = os.path.join(new_subfolder_path, OUTPUT_IMAGE_NAME)

                # Copy and rename the image
                try:
                    shutil.copy2(image_path, destination_path)
                    print(f"Copied {filename} from {subfolder_path} to {new_subfolder_path} as {OUTPUT_IMAGE_NAME}")
                except Exception as e:
                    print(f"Error copying {filename} to {destination_path}: {e}")

def main():
    # Verify base path exists
    if not os.path.exists(BASE_PATH):
        print(f"Base path {BASE_PATH} does not exist.")
        return

    print(f"Processing subfolders in {BASE_PATH}...")
    process_subfolders()
    print("Processing complete.")

if __name__ == "__main__":
    main()