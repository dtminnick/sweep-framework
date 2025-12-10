
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from sweep_framework.data.base_dataset import BaseDataset

class PlanDataset(BaseDataset):
    """
    Dataset class for retirement plan dynamics.

    Extends BaseDataset to:
    - Perform stratified train/validation/test splits on raw plan data.
    - Compute normalization statistics from the training split.
    - Preprocess each split into model-ready tensors.
    - Build PyTorch DataLoaders for training, validation, and testing.

    Attributes:
        raw_train (List[dict]): Raw plan dictionaries for training.
        raw_val (List[dict]): Raw plan dictionaries for validation.
        raw_test (List[dict]): Raw plan dictionaries for testing.
        train_examples (List[Tuple]): Preprocessed training examples.
        val_examples (List[Tuple]): Preprocessed validation examples.
        test_examples (List[Tuple]): Preprocessed test examples.
        train_loader (DataLoader): PyTorch DataLoader for training set.
        val_loader (DataLoader): PyTorch DataLoader for validation set.
        test_loader (DataLoader): PyTorch DataLoader for test set.
    """

    def __init__(self, raw_data, config_path="features.yaml",
                 train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                 seed=42, verbose=False):
        """
        Initialize PlanDataset with stratified splitting and preprocessing.

        Args:
            raw_data (List[dict]): List of plan dictionaries, each with keys:
                - "months": list of period dicts (dynamic features).
                - "static": dict of static features.
                - "label": integer class label.
            config_path (str): Path to YAML schema config file. Default "features.yaml".
            train_ratio (float): Proportion of data for training. Default 0.7.
            val_ratio (float): Proportion of data for validation. Default 0.2.
            test_ratio (float): Proportion of data for testing. Default 0.1.
            seed (int): Random seed for reproducibility. Default 42.
            verbose (bool): If True, logs preprocessing steps. Default False.

        Notes:
            - Performs stratified splitting by label to preserve class distribution.
            - Computes normalization statistics from training split only.
            - Preprocesses raw plans into tensors for each split.
        """
        super().__init__(config_path=config_path, verbose=verbose)
        ...
    
    def __len__(self):
        """
        Return the total number of examples across train, validation, and test splits.

        Returns:
            int: Total number of preprocessed examples.
        """
        return len(self.train_examples) + len(self.val_examples) + len(self.test_examples)

    def __getitem__(self, idx):
        """
        Retrieve a single example by index from the training set.

        Args:
            idx (int): Index of the training example.

        Returns:
            Tuple: Preprocessed training example
                (dynamic_seq, static_vec, dynamic_embs, static_embs, label).
        """
        return self.train_examples[idx]

    def build_loaders(self, batch_size=32):
        """
        Build PyTorch DataLoaders for train, validation, and test splits.

        Args:
            batch_size (int): Number of examples per batch. Default 32.

        Returns:
            None: Populates self.train_loader, self.val_loader, self.test_loader.
        """
        self.train_loader = super().build_loaders(self.train_examples, batch_size=batch_size)
        self.val_loader   = super().build_loaders(self.val_examples, batch_size=batch_size)
        self.test_loader  = super().build_loaders(self.test_examples, batch_size=batch_size)
