
import yaml
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from sweep_framework.data.base_dataset import BaseDataset

class PlanDataset(BaseDataset):
    """
    Dataset class for retirement plan dynamics.
    Inherits normalization, preprocessing, and loader building from BaseDataset.
    Handles train/val/test splitting and stats computation.
    """

    def __init__(self, raw_data, config_path="features.yaml",
                 train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                 seed=42, verbose=False):
        super().__init__(config_path=config_path, verbose=verbose)

        # Step 1: stratified split by label
        labels = [plan["label"] for plan in raw_data]
        train_raw, temp_raw, train_labels, temp_labels = train_test_split(
            raw_data, labels, stratify=labels,
            test_size=(1 - train_ratio), random_state=seed
        )
        val_size = val_ratio / (val_ratio + test_ratio)
        val_raw, test_raw, _, _ = train_test_split(
            temp_raw, temp_labels, stratify=temp_labels,
            test_size=(1 - val_size), random_state=seed
        )

        self.raw_train, self.raw_val, self.raw_test = train_raw, val_raw, test_raw

        # Step 2: compute stats from training split
        self.compute_stats(self.raw_train)

        # Step 3: preprocess each split
        self.train_examples = [self.preprocess_plan(p) for p in self.raw_train]
        self.val_examples   = [self.preprocess_plan(p) for p in self.raw_val]
        self.test_examples  = [self.preprocess_plan(p) for p in self.raw_test]

    def __len__(self):
        return len(self.train_examples) + len(self.val_examples) + len(self.test_examples)

    def __getitem__(self, idx):
        # Default to training set indexing
        return self.train_examples[idx]

    def build_loaders(self, batch_size=32):
        # Use BaseDataset’s collate_fn logic
        self.train_loader = super().build_loaders(self.train_examples, batch_size=batch_size)
        self.val_loader   = super().build_loaders(self.val_examples, batch_size=batch_size)
        self.test_loader  = super().build_loaders(self.test_examples, batch_size=batch_size)
