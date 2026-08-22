import random
from tqdm import tqdm
from tokenizers.normalizers import BertNormalizer

# Define file paths
input_file = r".\HEA_corpus\corpus\val_corpus_cleaned.txt"
vocab_mappings_file = r".\HEA_corpus\corpus\vocab_mappings.txt"
output_file = r".\HEA_corpus\corpus\val_corpus_norm.txt"

# Load vocab mappings
with open(vocab_mappings_file, 'r', encoding='utf-8') as f:
    mappings = f.read().strip().split('\n')
mappings = {m[0]: m[2:] for m in mappings}

# Initialize the BertNormalizer
norm = BertNormalizer(lowercase=False, strip_accents=True, clean_text=True, handle_chinese_chars=True)

def normalize_and_save(input_file, output_file):
    # Read the input corpus
    with open(input_file, 'r', encoding='utf-8') as f:
        corpus = f.read().strip().split('\n')

    # Normalize each sentence in the corpus
    random.seed(42)
    corpus = [norm.normalize_str(sent) for sent in tqdm(corpus, desc="Normalizing sentences")]
    corpus_norm = []

    for sent in tqdm(corpus, desc="Applying mappings"):
        norm_sent = ""
        for c in sent:
            if c in mappings:
                norm_sent += mappings[c]
            elif random.uniform(0, 1) < 0.3:
                norm_sent += c
            else:
                norm_sent += ' '
        corpus_norm.append(norm_sent)

    # Save the normalized corpus to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(corpus_norm))

    print(f"Normalization completed! Output saved to: {output_file}")

# Normalize the train_corpus_cleaned.txt file
normalize_and_save(input_file, output_file)