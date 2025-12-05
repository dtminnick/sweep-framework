
"""
Example script: Run a single LSTM model on SST2 using the sweep framework.

Steps:
1. Load the Stanford Sentiment Treebank (SST2) dataset from HuggingFace.
2. Tokenize sentences with a HuggingFace tokenizer.
3. Wrap data in the framework's Dataset class and build DataLoaders.
4. Configure an LSTM run with ModelConfig.
5. Train and evaluate using ModelRun.
6. Print summary metrics.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from sweep_framework.config.model_config import ModelConfig
from sweep_framework.data.dataset import Dataset
from sweep_framework.model.model_run import ModelRun


# 1. Load HuggingFace SST2 dataset
hf_ds = load_dataset("stanfordnlp/sst2")

# 2. Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# 3. Prepare examples as (text, label) pairs
train_examples = [(ex["sentence"], int(ex["label"])) 
                  for ex in hf_ds["train"] if ex["label"] != -1]
val_examples   = [(ex["sentence"], int(ex["label"])) 
                  for ex in hf_ds["validation"] if ex["label"] != -1]
test_examples  = [(ex["sentence"], int(ex["label"])) 
                  for ex in hf_ds["test"] if ex["label"] != -1]

# labels = [label for _, label in train_examples + val_examples + test_examples]
# print(set(labels))

# Combine into one dataset object
dataset = Dataset(train_examples + val_examples + test_examples)

# Stratified split (80/10/10)
dataset.stratify_split(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)

# Build DataLoaders with tokenizer
dataset.build_loaders(tokenizer, batch_size=32, max_len=128)


# 4. Configure model run
config = ModelConfig(
    run_type="LSTM",          # Choose "LSTM" or "GRU"
    learning_rate=1e-3,
    num_epochs=1,
    patience=2,
    optimizer_type="Adam",
    hidden_dim=128,
    dropout=0.3,
    embedding_dim=100,
    vocab_size=tokenizer.vocab_size,  # must set vocab size
    num_classes=2                     # SST2 is binary classification
)

# 5. Run training and evaluation
run = ModelRun(config, dataset)
run.run()

# 6. Print summary
print(run.export_summary())
