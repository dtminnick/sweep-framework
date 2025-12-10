
# BaseDataset

## Overview

The `BaseDataset` class provides the core data pipeline layer in the Sweep Framrwork.  It standardizes how raw plan dictionaries are transformed into tensors for model training and evaluation.

---

## Purpose in the Architecture

* **Schema management**: Loads feature and static schemas from a YAML configuration file.
* **Normalization**: Computes statistics (mean, standard deviation, minimum, maximum) from the training split and applies them consistently across the training, validation, and test splits.
* **Feature construction**: Converts raw monthly and static fields into dense tensors and embedding indices.
* **Integration**: Supplies pre-processed batches to models via PyTorch `DataLoader`.

---

## Class Reference

`BaseDataset`

`BaseDataset(config_path = 'features.yml', verbose = False)`

---

### Parameters

`config_path: str`

Path to the YAML configuration file defining `feature_schema`, `static_schema`, and `time_window`.

`verbose: bool`

If `True`, logs pre-processing steps for debugging.

---

### Attributes

`feature_schema`

Dictionary of dynamic feature definitions.

`static_schema`

Dictionary of static feature definitions.

`time_window`

Granularity of dynamic data, e.g. `daily`, `weekly`, `monthly`, `quarterly`.

`stats`

Normalization statistics computed from the training data.

---

### Methods

---

#### `compute_stats`

`compute_stats(raw_data: List[dict]) -> dict`

Computes normalization statistics for continuous features using only the training split.

##### Parameters

`raw_data: List[dict]`

List of plan dictionaries.  Each plan must contain:

`months` 

List of period dictionaries with dynamic feature values.

`static`

Dictionary of static feature values.

##### Returns

`dict`

Dictionary mapping feature `{mean, std, min, max}`.

---

#### `normalize`

`normalize(value, feature, spec) -> float | int | torch.Tensor`

Normalizes a single feature value according to its schema.  Handles binary, continuous (z-score, min-max), ordinal, and nominal (embedding or one-hot) types.  

##### Parameters

`value`

Raw feature value (float, int or string).

`feature: str`

Name of the feature being normalized.

`spec: dict`

Schema definition for the feature (type, normalization, categories, encoding).

##### Returns

`float | int | torch.Tensor`

A scalar, embedding index, or one-hot tensor.

---

#### `to_features`

`to_features(period_dict: dict) -> Tuple[torch.Tensor, dict]`

Converts one period's raw values into a dense vector and embedding indices.

##### Parameters

`period_dict: dict`

Dictionary of raw feature values for a single time period.

##### Returns

`torch.Tensor`

Dense feature vector.

`dict`

Dictionary of embedding indices for nominal features.

---

#### `preprocess_plan`

`preprocess_plan(plan: dict) -> Tuple[torch.Tensor, torch.Tensor, dict, dict int]`

Transforms a full plan into model-ready tensors.

##### Parameters

`plan: dict`

Plan dictionary with keys.

* `months` is a list of period dicts. 
* `static` is a dict of static features.
* `label` is the integer class label. 

##### Returns

`torch.Tensor`

Dynamic sequence (`time x features`).

`torch.Tensor`

Static feature vector.

`dict`

Dynamic embedding indices.

`dict`

Statis embedding indices.

`int`

Label.

---

#### `build_loaders`

`build_loaders(examples: List, batch_size = 32) -> DataLoader`

Builds a PyTorch `DataLoader` for a set of pre-processed examples.

##### Parameters

`examples: List`

List of tuples from `preprocess_plan`.

`batch_size: int`

Number of examples per batch (default = 32).

##### Returns

`DataLoader`

Loader yielding (`sequences, static_vecs, dynamic_embs, static_embs, labels`).

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