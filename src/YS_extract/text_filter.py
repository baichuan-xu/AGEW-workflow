import csv
import os

def create_filtered_csv(folder_path):
    """
    Create a filtered CSV file based on YES/NO judgments for a specific folder.
    
    Parameters:
    - folder_path: Path to the folder containing text.csv and text_sentence_judge.csv
    
    Returns:
    - Boolean indicating success and count of filtered sentences
    """
    # Define file paths within this folder
    input_file = os.path.join(folder_path, "text_split.csv")
    judgment_file = os.path.join(folder_path, "text_sentence_judge.csv")
    output_file = os.path.join(folder_path, "text_filter.csv")
    
    folder_name = os.path.basename(folder_path)
    
    # Check if both required files exist
    if not os.path.exists(input_file):
        # print(f"Skipping {folder_name}: text.csv not found")
        return False, 0
    if not os.path.exists(judgment_file):
        # print(f"Skipping {folder_name}: text_sentence_judge.csv not found")
        return False, 0
    
    # Skip if output file already exists
    if os.path.exists(output_file):
        # print(f"Skipping {folder_name}: text_split.csv already exists")
        return False, 0
    
    # Read the sentence file
    sentences = {}
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 2:
                    sentence_num = row[0]
                    sentence_text = row[1]
                    sentences[sentence_num] = sentence_text
    except Exception as e:
        print(f"Error in {folder_name}: Failed to read text.csv - {str(e)}")
        return False, 0
    
    # Read the judgment file and filter based on YES judgments
    filtered_sentences = []
    try:
        with open(judgment_file, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 2:
                    sentence_num = row[0]
                    judgment = row[1].strip().upper()  # Normalize to uppercase and trim whitespace
                    
                    if judgment == "YES" and sentence_num in sentences:
                        filtered_sentences.append((sentence_num, sentences[sentence_num]))
    except Exception as e:
        print(f"Error in {folder_name}: Failed to read text_sentence_judge.csv - {str(e)}")
        return False, 0
    
    # Write the filtered sentences to the output file
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for sentence_data in filtered_sentences:
                writer.writerow(sentence_data)
    except Exception as e:
        print(f"Error in {folder_name}: Failed to write text_split.csv - {str(e)}")
        return False, 0
    
    print(f"Processed {folder_name}: Created text_split.csv with {len(filtered_sentences)} sentences")
    return True, len(filtered_sentences)

def process_all_folders(base_path):
    """Process all folders from L0001 to L2621"""
    processed_count = 0
    skipped_count = 0
    total_sentences = 0
    
    for i in range(1, 3317):
        folder_name = f"L{i:04d}"
        folder_path = os.path.join(base_path, folder_name)
        
        if not os.path.isdir(folder_path):
            skipped_count += 1
            continue
        
        # Process the folder
        success, sentence_count = create_filtered_csv(folder_path)
        
        if success:
            processed_count += 1
            total_sentences += sentence_count
        else:
            skipped_count += 1
        
        # Progress update every 100 folders
        if i % 100 == 0:
            print(f"Progress: {i}/2621 folders processed. Created: {processed_count}, Skipped: {skipped_count}")
    
    print(f"\nProcessing complete!")
    print(f"Total folders with text_split.csv created: {processed_count}")
    print(f"Total sentences extracted: {total_sentences}")
    print(f"Total folders skipped: {skipped_count}")

def main():
    base_path = r".\HEA_finetune\Yield_Strength"
    print(f"Starting to process folders in {base_path}...")
    process_all_folders(base_path)

if __name__ == "__main__":
    main()