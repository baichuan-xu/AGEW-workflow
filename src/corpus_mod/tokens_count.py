from datasets import load_dataset
from transformers import AutoTokenizer
from pathlib import Path
import logging
from tqdm import tqdm

# Define file paths
train_norm_file = r".\test\text\train_corpus_norm.txt"
save_dir = r".\test\text"

# Create subdirectories for saving results
tokens_len_dir = Path(save_dir) / "tokens_len"
tokenizer_saved_dir = Path(save_dir) / "tokenizer_saved"

# Ensure directories exist
tokens_len_dir.mkdir(parents=True, exist_ok=True)
tokenizer_saved_dir.mkdir(parents=True, exist_ok=True)

# Load tokenizer (ensure the tokenizer is already saved in tokenizer_saved_dir)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_saved_dir)

# Load the normalized corpus as a dataset
raw_datasets = load_dataset("text", data_files={"train": train_norm_file})

print(f"Raw datasets loaded: {raw_datasets}")

def count_len(example):
    """
    Tokenize the text and calculate the length of each tokenized sequence.
    """
    outputs = tokenizer(
        example["text"],
        truncation=False,
        padding=False,
        return_length=True,
    )
    # Extract token lengths
    tokens_len = [length for length in outputs["length"]]
    return {"tokens_len": tokens_len}

# Apply tokenization and length counting
tokens_lens = raw_datasets.map(
    count_len,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,  # Remove original text column
)

# Save the token lengths to disk
tokens_lens.save_to_disk(tokens_len_dir)

print(f"Token lengths saved to: {tokens_len_dir}")