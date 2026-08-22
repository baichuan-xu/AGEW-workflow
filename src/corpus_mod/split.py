import os
import random
import shutil

# Define paths
source_dir = r".\HEA_corpus\xml"
train_dir = r".\HEA_corpus\corpus\train"
val_dir = r".\HEA_corpus\corpus\val"

# Create train and val directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Get list of all xml files in the source directory
xml_files = [f for f in os.listdir(source_dir) if f.endswith('.xml')]

# Shuffle the list of xml files randomly
random.shuffle(xml_files)

# Calculate the split index (85% for training, 15% for validation)
split_index = int(len(xml_files) * 0.85)

# Split the files into train and validation sets
train_files = xml_files[:split_index]
val_files = xml_files[split_index:]

# Function to move files with handling for long paths
def move_files(files, destination):
    for file in files:
        src_path = os.path.join(source_dir, file)
        dst_path = os.path.join(destination, file)
        
        # Handle long file paths on Windows by using the \\?\ prefix
        if os.name == 'nt':  # Check if running on Windows
            src_path = '\\\\?\\' + os.path.abspath(src_path)
            dst_path = '\\\\?\\' + os.path.abspath(dst_path)
        
        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        
        # Move the file
        shutil.move(src_path, dst_path)

# Move the files to their respective directories
move_files(train_files, train_dir)
move_files(val_files, val_dir)

print(f"Moved {len(train_files)} files to {train_dir}")
print(f"Moved {len(val_files)} files to {val_dir}")