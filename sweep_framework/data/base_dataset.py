
# sweep_framework/data/base_dataset.py
from abc import ABC, abstractmethod

class BaseDataset(ABC):
    """
    Abstract base class for datasets in the Sweep Framework.
    Defines the interface for loading, splitting, and building loaders.
    """

    def __init__(self, examples):
        self.examples = examples
        self.train_examples = []
        self.val_examples = []
        self.test_examples = []
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    @abstractmethod
    def stratify_split(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
        """Split examples into train/val/test sets."""
        pass

    @abstractmethod
    def build_loaders(self, **kwargs):
        """Build PyTorch DataLoaders for train/val/test sets."""
        pass
