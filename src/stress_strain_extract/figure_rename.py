import os
import re

def find_figures_with_keyword(text_content):
    """
    Extract figure numbers from the FIGURES section of text.txt
    that contain the keywords 'yield strength', 'Yield strength', or 'Yield Strength'.
    """
    # Find the FIGURES section
    figures_section = re.search(r'={40} FIGURES ={40}(.*?)$', text_content, re.DOTALL)
    if not figures_section:
        return []

    # Extract all figure entries
    figures = re.findall(r'Fig\.\s*(\d+):(.+?)(?=Fig\.\s*\d+:|$)', figures_section.group(1), re.DOTALL)

    matched_figures = []
    for fig_num, description in figures:
        if re.search(r'\bstress-strain\b', description, re.IGNORECASE):
            matched_figures.append(fig_num.strip())

    return matched_figures

def rename_files_in_subfolder(folder_path, matched_figures):
    """
    Rename image files in the current subfolder based on matched figure numbers.
    """
    for fig_num in matched_figures:
        original_name = f"gr{fig_num}.jpg"
        new_name = f"YS{fig_num}.jpg"
        original_path = os.path.join(folder_path, original_name)
        new_path = os.path.join(folder_path, new_name)

        if os.path.exists(original_path):
            os.rename(original_path, new_path)
            print(f"Renamed {original_name} -> {new_name} in folder: {folder_path}")
        else:
            print(f"File {original_name} not found in folder: {folder_path}")

def process_subfolders(base_path):
    """
    Process all subfolders (L001 to L2620) and rename images based on text.txt.
    """
    for i in range(1, 3356):  # From 1 to 2620 inclusive
        folder_name = f"L{i:04d}"  # Format folder name with leading zeros (e.g., L0001, L0002)
        folder_path = os.path.join(base_path, folder_name)

        # Check if the folder exists
        if not os.path.exists(folder_path):
            print(f"Folder does not exist: {folder_path}")
            continue

        # Locate the text.txt file in the folder
        text_file = os.path.join(folder_path, "text.txt")
        if not os.path.exists(text_file):
            print(f"No text.txt file found in folder: {folder_path}")
            continue

        # Read the content of text.txt
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading text.txt in folder {folder_path}: {str(e)}")
            continue

        # Find figures containing "yield strength"
        matched_figures = find_figures_with_keyword(content)

        if matched_figures:
            # Rename images in the current subfolder
            rename_files_in_subfolder(folder_path, matched_figures)
        else:
            print(f"No figures containing 'yield strength' found in folder: {folder_path}")

# Main function
if __name__ == "__main__":
    base_path = r".\stress-strain2"
    process_subfolders(base_path)

print("All operations completed.")