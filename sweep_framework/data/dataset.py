
# sweep_framework/data/dataset.py
"""
Dataset utilities for text classification sweeps.

This module defines a Dataset class that:
- Stores examples as (text, label) pairs.
- Splits data into train/val/test sets with stratification by label.
- Wraps splits into PyTorch DataLoaders using a HuggingFace tokenizer.
"""

import random
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader


class Dataset:
    """
    Dataset wrapper for text classification.

    Attributes:
        examples (List[Tuple[str, int]]): Raw examples as (text, label).
        label_groups (Dict[int, List[Tuple[str, int]]]): Examples grouped by label.
        train, val, test (List[Tuple[str, int]]): Split lists.
        train_loader, val_loader, test_loader (DataLoader): PyTorch DataLoaders.
    """

    def __init__(self, examples: List[Tuple[str, int]]):
        """
        Args:
            examples (List[Tuple[str, int]]): List of (text, label) pairs.
        """
        self.examples = examples
        self.label_groups = self._group_by_label(examples)
        self.train: List[Tuple[str, int]] = []
        self.val: List[Tuple[str, int]] = []
        self.test: List[Tuple[str, int]] = []
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None

    def _group_by_label(self, examples: List[Tuple[str, int]]) -> Dict[int, List[Tuple[str, int]]]:
        """Group examples by label."""
        groups: Dict[int, List[Tuple[str, int]]] = {}
        for text, label in examples:
            groups.setdefault(int(label), []).append((text, int(label)))
        return groups

    def stratify_split(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed: Optional[int] = None):
        """
        Stratified split by label.

        Args:
            train_ratio (float): Proportion for training set.
            val_ratio (float): Proportion for validation set.
            test_ratio (float): Proportion for test set.
            seed (Optional[int]): Random seed for reproducibility.
        """
        if seed is not None:
            random.seed(seed)
        self.train, self.val, self.test = [], [], []
        for _, group in self.label_groups.items():
            random.shuffle(group)
            n = len(group)
            train_end = int(train_ratio * n)
            val_end = train_end + int(val_ratio * n)
            self.train += group[:train_end]
            self.val += group[train_end:val_end]
            self.test += group[val_end:]

    def build_loaders(self, tokenizer, batch_size=32, max_len=128):
        """
        Build PyTorch DataLoaders for train/val/test splits.

        Args:
            tokenizer: HuggingFace tokenizer.
            batch_size (int): Batch size.
            max_len (int): Maximum sequence length.
        """
        self.train_loader = DataLoader(_TextLabelDataset(self.train, tokenizer, max_len), batch_size=batch_size, shuffle=True)
        self.val_loader   = DataLoader(_TextLabelDataset(self.val, tokenizer, max_len), batch_size=batch_size)
        self.test_loader  = DataLoader(_TextLabelDataset(self.test, tokenizer, max_len), batch_size=batch_size)

    def get_split(self, split: str) -> List[Tuple[str, int]]:
        """Return raw examples for a given split."""
        if split == "train": return self.train
        if split == "val": return self.val
        if split == "test": return self.test
        raise ValueError(f"Unknown split: {split}")

    @classmethod
    def from_csv(cls, path: str, text_col: str = "text", label_col: str = "label") -> "Dataset":
        """
        Load dataset from a CSV file.

        Args:
            path (str): Path to CSV file.
            text_col (str): Column name for text.
            label_col (str): Column name for labels.

        Returns:
            Dataset: Initialized dataset.
        """
        import pandas as pd
        df = pd.read_csv(path)
        examples = list(zip(df[text_col].tolist(), df[label_col].astype(int).tolist()))
        return cls(examples)


class _TextLabelDataset(TorchDataset):
    """
    Internal PyTorch Dataset for tokenizing text/label pairs.

    Args:
        examples (List[Tuple[str, int]]): List of (text, label).
        tokenizer: HuggingFace tokenizer.
        max_len (int): Maximum sequence length.
    """

    def __init__(self, examples: List[Tuple[str, int]], tokenizer, max_len: int = 128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text, label = self.examples[idx]
        tok = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input": tok["input_ids"].squeeze(0),
            "target": torch.tensor(label, dtype=torch.long)
        }
