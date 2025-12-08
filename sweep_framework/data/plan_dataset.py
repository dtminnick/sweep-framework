
# sweep_framework/data/plan_dataset.py

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from sweep_framework.data.base_dataset import BaseDataset
from sweep_framework.data.plan_preprocessor import PlanPreprocessor


def compute_stats(raw_plans, feature_schema, static_schema):
    """
    Compute normalization statistics (mean, std, min, max) for continuous features
    using only the training split.
    """
    stats = {}

    # dynamic features
    for f, spec in feature_schema.items():
        if spec.get("type") == "continuous":
            values = []
            for plan in raw_plans:
                for period in plan["months"]:
                    if f in period:
                        values.append(period[f])
            if values:
                stats[f] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }

    # static features
    for f, spec in static_schema.items():
        if spec.get("type") == "continuous":
            values = [plan["static"].get(f, 0.0) for plan in raw_plans]
            if values:
                stats[f] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }

    return stats


class PlanDataset(BaseDataset):
    """
    Dataset class for retirement plan dynamics.
    Splits raw data into train/val/test, computes normalization stats from training set,
    and preprocesses each split into tensors for RNN models.
    """

    def __init__(self, raw_data, config_path="features.yaml", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
        # Step 1: stratified split by label
        labels = [plan["label"] for plan in raw_data]
        train_raw, temp_raw, train_labels, temp_labels = train_test_split(
            raw_data, labels, stratify=labels, test_size=(1 - train_ratio), random_state=seed
        )
        val_size = val_ratio / (val_ratio + test_ratio)
        val_raw, test_raw, _, _ = train_test_split(
            temp_raw, temp_labels, stratify=temp_labels, test_size=(1 - val_size), random_state=seed
        )

        self.raw_train, self.raw_val, self.raw_test = train_raw, val_raw, test_raw

        # Step 2: load schema and compute stats from training split
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        feature_schema = cfg.get("feature_schema", {})
        static_schema = cfg.get("static_schema", {})
        stats = compute_stats(self.raw_train, feature_schema, static_schema)

        # Step 3: instantiate preprocessor with stats
        self.preprocessor = PlanPreprocessor(config_path=config_path, stats=stats)

        # Step 4: preprocess each split
        self.train_examples = [self.preprocessor.preprocess_plan(p) for p in self.raw_train]
        self.val_examples   = [self.preprocessor.preprocess_plan(p) for p in self.raw_val]
        self.test_examples  = [self.preprocessor.preprocess_plan(p) for p in self.raw_test]

    def __len__(self):
        return len(self.train_examples) + len(self.val_examples) + len(self.test_examples)

    def __getitem__(self, idx):
        # Default to training set indexing
        return self.train_examples[idx]

    def build_loaders(self, batch_size=32):
        def collate_fn(batch):
            sequences, static_vecs, labels = zip(*batch)
            return torch.stack(sequences), torch.stack(static_vecs), torch.tensor(labels)

        self.train_loader = DataLoader(self.train_examples, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
        self.val_loader   = DataLoader(self.val_examples, batch_size=batch_size, collate_fn=collate_fn)
        self.test_loader  = DataLoader(self.test_examples, batch_size=batch_size, collate_fn=collate_fn)
