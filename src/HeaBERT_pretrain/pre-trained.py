import os
from pathlib import Path
from argparse import ArgumentParser
import torch
from transformers import (
    AutoConfig,
    BertForMaskedLM,
    AutoTokenizer,
    DataCollatorForWholeWordMask,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Ensure directory exists
def ensure_dir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dir_path

# Argument parsing with default values
parser = ArgumentParser()
parser.add_argument('--train_file', type=str, default="train_corpus_norm.txt", help="Path to training file")
parser.add_argument('--val_file', type=str, default="val_corpus_norm.txt", help="Path to validation file")
parser.add_argument('--output_dir', type=str, default="./output", help="Output directory")
parser.add_argument('--model_name_or_path', type=str, default="./matscibert_local", help="Path to pretrained model or model identifier from huggingface.co/models")
parser.add_argument('--per_device_train_batch_size', type=int, default=64, help="Batch size per GPU/CPU for training.")
parser.add_argument('--per_device_eval_batch_size', type=int, default=64, help="Batch size per GPU/CPU for evaluation.")
parser.add_argument('--learning_rate', type=float, default=5e-5, help="The initial learning rate for AdamW.")
parser.add_argument('--num_train_epochs', type=int, default=30, help="Total number of training epochs to perform.")
parser.add_argument('--logging_steps', type=int, default=100, help="Log every X updates steps.")
parser.add_argument('--eval_steps', type=int, default=500, help="Run an evaluation every X steps.")
parser.add_argument('--save_steps', type=int, default=500, help="Save checkpoint every X updates steps.")
parser.add_argument('--save_total_limit', type=int, default=3, help="Limit the total amount of checkpoints. Deletes the older checkpoints.")
parser.add_argument('--seed', type=int, default=42, help="Random seed for initialization.")
parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit (mixed) precision training instead of 32-bit training.")
parser.add_argument('--max_seq_length', type=int, default=512, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.")
parser.add_argument('--local_rank', type=int, default=-1, help="Local GPU rank for distributed training")
args = parser.parse_args()

# Set seed for reproducibility
set_seed(args.seed)

# Model and tokenizer setup
cache_dir = ensure_dir(os.path.join(args.output_dir, "cache"))

# Load configuration and tokenizer
print(f"Loading model and tokenizer from: {args.model_name_or_path}")
config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=cache_dir)

# Load model
model = BertForMaskedLM.from_pretrained(args.model_name_or_path, config=config, cache_dir=cache_dir)
model.to(device) # Move model to the device

# Dataset class
class LineByLineTextDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, file_path, block_size): # block_size is passed here
        if not os.path.isfile(file_path):
            raise ValueError(f"Input file not found: {file_path}")

        print(f"Reading file: {file_path}")
        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        # Tokenize all texts
        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]
        print(f"Created {len(self.examples)} examples from {file_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # The DataCollatorForWholeWordMask will handle padding and creating labels
        return torch.tensor(self.examples[i], dtype=torch.long)

# Determine effective block size for tokenization
# Use tokenizer.model_max_length but cap it to a sensible value like args.max_seq_length
# to prevent OverflowError if tokenizer.model_max_length is excessively large.
if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length < 1e10: # Check if it's a sensible, finite number
    tokenizer_defined_max_len = tokenizer.model_max_length
else:
    tokenizer_defined_max_len = args.max_seq_length # Fallback if not defined or too large

effective_block_size = min(tokenizer_defined_max_len, args.max_seq_length)

if effective_block_size != tokenizer_defined_max_len :
    print(f"Warning: tokenizer.model_max_length ({tokenizer_defined_max_len}) was potentially very large or different from args.max_seq_length. Using effective_block_size: {effective_block_size}")
elif effective_block_size != args.max_seq_length:
     print(f"Using effective_block_size: {effective_block_size} (from args.max_seq_length as tokenizer.model_max_length was too large or same)")


# Create datasets
print("Creating training dataset...")
train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=args.train_file,
    block_size=effective_block_size # Use the capped/sensible block size
)
print("Creating validation dataset...")
eval_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=args.val_file,
    block_size=effective_block_size # Use the capped/sensible block size
)

# Data collator
data_collator = DataCollatorForWholeWordMask(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# Training arguments
# To match user's original hardcoded fp16=True, we set it directly here.
# If you want it to be flag-dependent, use args.fp16 and ensure the flag is passed.
current_fp16_setting = True # Matches original script's hardcoded value
if not torch.cuda.is_available() and args.fp16:
    print("Warning: FP16 selected but no CUDA GPU available. Disabling FP16.")
    current_fp16_setting = False
elif args.fp16: # if --fp16 flag is explicitly used
    current_fp16_setting = True


training_args = TrainingArguments(
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    prediction_loss_only=True,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    learning_rate=args.learning_rate,
    weight_decay=0.01,
    adam_epsilon=1e-6,
    max_grad_norm=1.0,
    num_train_epochs=args.num_train_epochs,
    seed=args.seed,
    logging_dir=os.path.join(args.output_dir, "logs"),
    logging_steps=args.logging_steps,
    fp16=current_fp16_setting, # Use the determined FP16 setting
    local_rank=args.local_rank,

    eval_steps=args.eval_steps,
    save_strategy="steps",
    save_steps=args.save_steps,
    save_total_limit=args.save_total_limit,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Check if there are existing checkpoints to resume from
resume_from_checkpoint = None
if os.path.isdir(args.output_dir):
    checkpoints = [
        os.path.join(args.output_dir, d)
        for d in os.listdir(args.output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(args.output_dir, d))
    ]
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
        latest_checkpoint = checkpoints[-1]
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
        resume_from_checkpoint = latest_checkpoint
    else:
        print("No valid checkpoints found in output directory. Starting training from scratch.")
elif os.path.exists(args.output_dir) and not os.path.isdir(args.output_dir):
    print(f"Warning: Output directory path {args.output_dir} exists but is a file. Training from scratch.")
else:
    print("Output directory does not exist. It will be created. Starting training from scratch.")

# Start training
print("Starting training...")
train_res = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
print("Training finished.")
print(train_res)

# Save the final model, tokenizer, and training arguments
print("Saving final model...")
trainer.save_model()
trainer.save_state()
print(f"Final model saved to {args.output_dir}")

# Evaluate the model on the validation set after training
print("Evaluating on validation set...")
eval_results = trainer.evaluate(eval_dataset=eval_dataset)
print(f"Evaluation results: {eval_results}")

print("Script finished.")