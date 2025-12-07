
# Dataset

## Overview

The `Dataset` class is responsible for managing raw examples, splitting them into training/validation/test sets, and building PyTorch `DataLoader` objects.  
It acts as the **data pipeline layer** in the Sweep Framework, bridging external datasets (e.g., HuggingFace GLUE SST‑2) with the training loop in `ModelRun`.

### Purpose in the Architecture
- **Input management**: Accepts raw `(text, label)` pairs.
- **Splitting**: Performs stratified splits to preserve class distribution across train/val/test sets.
- **Loader construction**: Uses a HuggingFace tokenizer to convert text into tensors (`input_ids`, `attention_mask`, `labels`).
- **Integration**: Supplies batches to the model during training and evaluation.

---

## Class Reference

### `Dataset`

`Dataset(examples: List[Tuple[str, int]])`

#### Parameters
- **`examples: List[Tuple[str, int]]`**  
  Raw dataset examples as `(text, label)` pairs.  
  - `text`: string input (sentence/document).  
  - `label`: integer class index. Must be in `[0, num_classes-1]`.  

  Important: Filter out invalid labels (e.g., `-1` in SST‑2 test set).

---

### Attributes
- **`examples`**: List of all raw `(text, label)` pairs.  
- **`train_examples`**: Subset used for training.  
- **`val_examples`**: Subset used for validation.  
- **`test_examples`**: Subset used for testing.  
- **`train_loader`**: PyTorch `DataLoader` for training set.  
- **`val_loader`**: PyTorch `DataLoader` for validation set.  
- **`test_loader`**: PyTorch `DataLoader` for test set.  

---

### Methods

#### `stratify_split`

`stratify_split(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)`

Splits the dataset into train/val/test sets while preserving class distribution.

- **Parameters**:
  - `train_ratio: float` → proportion of examples for training, default 0.8.
  - `val_ratio: float` → proportion for validation, default 0.1.
  - `test_ratio: float` → proportion for testing, default 0.1.
  - `seed: int` → random seed for reproducibility, default 42.

- **Outputs**:
  - Populates `train_examples`, `val_examples`, `test_examples`.

---

#### `build_loaders`

`build_loaders(tokenizer, batch_size=32, max_len=128)`

Builds PyTorch `DataLoader` objects for each split.

- **Parameters**:
  - `tokenizer`: HuggingFace tokenizer (e.g., `AutoTokenizer`).  
  - `batch_size: int` → batch size for loaders, default 32.
  - `max_len: int` → maximum sequence length for tokenization, default 128.

- **Outputs**:
  - Populates `train_loader`, `val_loader`, `test_loader`.  
  - Each loader yields batches of `(input_ids, attention_mask, labels)` tensors.

---

## Usage Example

```python
from sweep_framework.data.dataset import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset

# Load SST-2 dataset
hf_ds = load_dataset("stanfordnlp/sst2")

# Filter out invalid labels (-1)
train_examples = [(ex["sentence"], int(ex["label"])) for ex in hf_ds["train"] if ex["label"] != -1]
val_examples   = [(ex["sentence"], int(ex["label"])) for ex in hf_ds["validation"] if ex["label"] != -1]
test_examples  = [(ex["sentence"], int(ex["label"])) for ex in hf_ds["test"] if ex["label"] != -1]

# Combine into one dataset object
dataset = Dataset(train_examples + val_examples + test_examples)

# Stratified split
dataset.stratify_split(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)

# Build loaders
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
dataset.build_loaders(tokenizer, batch_size=32, max_len=128)

# Inspect loaders
print(len(dataset.train_loader), len(dataset.val_loader), len(dataset.test_loader))
```