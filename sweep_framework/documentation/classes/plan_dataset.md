
# PlanDataset

## Overview

The `PlanDataset` class extends `BaseDataset` to manage retirement plan data.  It performs stratified training, validation, and test splits, computes normalization statistics from the training split, and pre-processes each plan into model-ready tensors.

---

## Purpose in the Architecture

* **Splitting**: Divides raw plan dictionaries into training, validation, and test sets while preseving class distribution.
* **Normalization**: Computes statistics from the training split and applies them consistently across all splits.
* **Pre-processing**: Converts raw monthly and static fields into dense tensors and embedding indices.
* **Integration**: Supplies batches to models via PyTorch `DataLoader` objects for training and evaluation.

---

## Class Reference

`PlanDataset`

`PlanDataset(raw_data, config_path = 'features.yml', train_ratio = 0.7, val_ratio = 0.2, test_ratio = 0.1, seed = 42, verbose = False)`

### Parameters

`raw_data: List[dict]`

List of plan dictionaries, each containing:

* `months`: List of period dicts with dynamic features.
* `static`: Dict of static features.
* `label`: Integer class label.

`config_path: str`

Path to YAML schema configuration file; default `features.yml`.

`train_ratio: float`

Proportion of data for training; default `0.7`.

`val_ratio: float`

Proportion of data for validation; default `0.2`.

`test_ratio: float`

Proportion of data for testing; default `0.1`.

`seed`

Random seed for reproducibility; default `42`.

`verbose: bool`

If `True`, logs pre-processing steps for debugging.

---

### Attributes

`raw_train`

Raw plan dictionaries for training.

`raw_val`

Raw plan dictionaries for validation.

`raw_test`

Raw plan dictionaries for test.

`train_examples`

Pre-processed training examples.

`val_examples`

Pre-processed validation examples.

`test_examples`

Pre-processed test examples.

`train_loader`

PyTorch `DataLoader` for training set.

`val_loader`

PyTorch `DataLoader` for validation set.

`test_loader`

PyTorch `DataLoader` for test set.

---

### Methods

#### `__len__`

`__len__() -> int`

Returns the total number of examples across train, validation, and test splits.

##### Returns

`int`

Total number of pre-processed examples.

--- 

#### `__getitem__`

`__getitem__(idx: int) -> Tuple`

Retrieves a single example by index from the training set.

##### Parameters

`idx: int`

Index of the training example.

##### Returns

`Tuple`

Pre-processed training example.

---

#### `build_loaders`

`build_loaders(batch_size = 32) -> None`

Builds PyTorch `DataLoader` objects for train, validation, and test splits.

##### Parameters

`batch_size: int`

Number of examples per batch; default `32`.

##### Returns

`None`

Populates `train_loader`, `val_loader`, `test_loader`.

---

## Usage Example

```{python}
from sweep_framework.data.plan_dataset import PlanDataset
import pandas as pd

# Load expanded raw input CSV
df = pd.read_csv("sweep_framework/documentation/usage/raw_input_data.txt")

# Convert to plan dictionaries
plans = []
for pid, group in df.groupby("plan_id"):
    months = group.to_dict(orient="records")
    static_fields = {col: group.iloc[0][col] for col in [
        "hardship_allowed","loan_allowed","inserv_allowed",
        "participants","median_balance","median_age",
        "fee_distribution","fee_loan_origination","vendor_type","fee_tier"
    ]}
    label = group.iloc[0]["label"]
    plans.append({"plan_id": str(pid), "months": months, "static": static_fields, "label": label})

# Instantiate dataset (handles split + stats internally)
dataset = PlanDataset(plans, config_path="sweep_framework/config/features.yml", verbose=True)

# Inspect normalization stats
print(dataset.stats)

# Build loaders
dataset.build_loaders(batch_size=32)
batch = next(iter(dataset.train_loader))
print(batch[0].shape, batch[1].shape, batch[4])  # sequences, static_vecs, labels
```
