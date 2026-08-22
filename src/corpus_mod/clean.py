def remove_separator(input_file, output_file):
    # Define the separator to be removed
    separator = "=" * 80

    try:
        # Open the input file for reading
        with open(input_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()

        # Open the output file for writing
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for line in lines:
                # Write the line to the output file only if it doesn't match the separator
                if line.strip() != separator:
                    outfile.write(line)

        print(f"Separator '{separator}' removed successfully! Output saved to: {output_file}")

    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    input_file = r".\HEA_corpus\corpus\val_corpus.txt"
    output_file = r".\HEA_corpus\corpus\val_corpus_cleaned.txt"

    # Process the file to remove separators
    remove_separator(input_file, output_file)