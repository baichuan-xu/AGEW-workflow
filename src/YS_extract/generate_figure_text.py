import os
import re
import csv
import glob # To find files matching a pattern

# Define the base path
base_path = r".\HEA_finetune\Yield_Strength"
start_folder_num = 1
end_folder_num = 3317

print(f"Starting processing for folders L{start_folder_num:04d} to L{end_folder_num:04d} in '{base_path}'...")

# Loop through the specified range of subfolders
for i in range(start_folder_num, end_folder_num + 1):
    folder_name = f"L{i:04d}" # Format number with leading zeros (e.g., L0001)
    subfolder_path = os.path.join(base_path, folder_name)

    # Check if the subfolder exists
    if not os.path.isdir(subfolder_path):
        # print(f"Subfolder '{subfolder_path}' not found. Skipping.")
        continue # Skip to the next folder number

    # --- Step 2: Find figure_out*.csv files ---
    figure_csv_pattern = os.path.join(subfolder_path, "figure_out*.csv")
    figure_csv_files = glob.glob(figure_csv_pattern)

    # If no figure_out files are found, skip this subfolder
    if not figure_csv_files:
        # print(f"No 'figure_out*.csv' files found in '{subfolder_path}'. Skipping.")
        continue # Skip to the next folder number

    print(f"\nProcessing folder: {folder_name}")

    # Extract numbers and sort files numerically
    figure_files_data = {}
    for f_path in figure_csv_files:
        match = re.search(r'figure_out(\d+)\.csv$', os.path.basename(f_path))
        if match:
            num = int(match.group(1))
            figure_files_data[num] = f_path
        else:
             print(f"Warning: Could not extract number from '{f_path}' in {folder_name}. Skipping this file.")


    # --- Prepare data storage ---
    figure_descriptions = {} # Store descriptions from text.txt {fig_num: description}
    figure_related_sentences = {} # Store sentences from text.csv {fig_num: [sentence1, sentence2]}
    figure_csv_content = {} # Store processed CSV content {fig_num: content_string}

    # --- Step 5: Process text.txt for descriptions (MODIFIED SECTION) ---
    text_txt_path = os.path.join(subfolder_path, "text.txt")
    try:
        if os.path.exists(text_txt_path):
            with open(text_txt_path, 'r', encoding='utf-8') as f_txt:
                content = f_txt.read()
                # Define the starting marker for the figures section
                start_marker = "======================================== FIGURES ========================================"
                start_index = content.find(start_marker)

                # Check if the start marker is found
                if start_index != -1:
                    # Extract the content from the start marker to the end of the file
                    figures_section = content[start_index + len(start_marker):].strip()
                    
                    # Use regex to find "Fig. <number>:\nDescription..." patterns
                    # This regex looks for "Fig." followed by a number, a colon, optional whitespace,
                    # a newline, and then captures the description until the next "Fig." or end of section.
                    # It handles multi-line descriptions.
                    desc_matches = re.findall(r'Fig\.\s*(\d+):\s*\n(.*?)(?=\nFig\.|\Z)', figures_section, re.DOTALL)
                    for num_str, desc in desc_matches:
                         fig_num = int(num_str)
                         # Clean up description (remove leading/trailing whitespace from each line and join)
                         cleaned_desc = "\n".join(line.strip() for line in desc.strip().splitlines())
                         figure_descriptions[fig_num] = f"Fig. {fig_num}:\n{cleaned_desc}"
                         # print(f"  Found description for Fig. {fig_num} in {folder_name}")

                else:
                     # This 'else' block now correctly reports that the start marker was not found
                     print(f"Warning: Could not find the figure start marker in '{text_txt_path}'.")
        else:
            print(f"Warning: '{text_txt_path}' not found in {folder_name}.")
    except Exception as e:
        print(f"Error processing '{text_txt_path}': {e}")


    # --- Step 6: Process text.csv for related sentences ---
    text_csv_path = os.path.join(subfolder_path, "text_split.csv")
    try:
        if os.path.exists(text_csv_path):
            with open(text_csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                # Skip header if exists (optional, adjust if no header)
                # next(reader, None)
                for row in reader:
                    if len(row) >= 2:
                        sentence = row[1].strip()
                        # Find all figure references in the sentence
                        sentence_fig_matches = re.findall(r'Fig\.\s*(\d+)', sentence)
                        for num_str in sentence_fig_matches:
                             fig_num = int(num_str)
                             if fig_num not in figure_related_sentences:
                                 figure_related_sentences[fig_num] = []
                             if sentence not in figure_related_sentences[fig_num]: # Avoid duplicates
                                 figure_related_sentences[fig_num].append(sentence)
                                 # print(f"  Found related sentence for Fig. {fig_num} in {folder_name}")
        else:
             print(f"Warning: '{text_csv_path}' not found in {folder_name}.")
    except Exception as e:
        print(f"Error processing '{text_csv_path}': {e}")


    # --- Step 3 & 4: Read figure_out*.csv content and replace comma with pipe ---
    for fig_num, csv_path in figure_files_data.items():
        try:
            with open(csv_path, 'r', encoding='utf-8') as f_csv:
                content = f_csv.read()
                processed_content = content.replace(',', '|')
                figure_csv_content[fig_num] = processed_content.strip() # Store processed content
        except Exception as e:
            print(f"Error reading or processing '{csv_path}': {e}")


    # --- Step 3 (cont.): Assemble figure.txt ---
    output_figure_txt_path = os.path.join(subfolder_path, "figure.txt")
    try:
        with open(output_figure_txt_path, 'w', encoding='utf-8') as outfile:
            print(f"  Writing to '{output_figure_txt_path}'...")
            # Iterate through the figures found, sorted by number
            for fig_num in sorted(figure_files_data.keys()):
                # Write description from text.txt (if found)
                if fig_num in figure_descriptions:
                    outfile.write(figure_descriptions[fig_num] + "\n\n") # Add extra newline
                else:
                    print(f"  Warning: No description found for Fig. {fig_num} in {text_txt_path}.")
                    outfile.write(f"(No description found for Fig. {fig_num})\n\n")


                # Write processed content from figure_outX.csv (if processed)
                if fig_num in figure_csv_content:
                    outfile.write(figure_csv_content[fig_num] + "\n\n") # Add extra newline
                else:
                     print(f"  Warning: No content found/processed for figure_out{fig_num}.csv.")
                     outfile.write(f"(No content found for figure_out{fig_num}.csv)\n\n")


                # Write related sentences from text.csv (if found)
                if fig_num in figure_related_sentences:
                    outfile.write(f"Description of Fig. {fig_num}:\n")
                    for sentence in figure_related_sentences[fig_num]:
                        outfile.write(sentence + "\n")
                    outfile.write("\n") # Add extra newline after sentences section
                else:
                    # If no sentences were found specifically mentioning "Fig. <num>", indicate that.
                     outfile.write(f"(No sentences found referencing Fig. {fig_num} in {text_csv_path})\n\n")


                # Add a separator between figure blocks for readability (optional)
                outfile.write("-----------------------------------------\n\n")

        print(f"  Successfully created/updated '{output_figure_txt_path}'.")

    except Exception as e:
        print(f"Error writing to '{output_figure_txt_path}': {e}")


print("\nScript finished.")