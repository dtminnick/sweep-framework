
import random
from typing import List, Dict, Tuple, Optional

class Dataset:
    def __init__(self, examples: List[Tuple[str, str]]):
        """
        examples: List of (text, label) pairs
        """
        self.examples = examples
        self.label_groups = self._group_by_label(examples)
        self.train = []
        self.val = []
        self.test = []

    def _group_by_label(self, examples: List[Tuple[str, str]]) -> Dict[str, List[Tuple[str, str]]]:
        groups = {}
        for text, label in examples:
            groups.setdefault(label, []).append((text, label))
        return groups
    
    def stratify_split(self, train_ratio = 0.8, val_ratio = 0.1, test_ratio = 0.1, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

        self.train, self.val, self.test = [], [], []

        for label, group in self.label_groups.items():
            random.shuffle(group)
            n = len(group)
            train_end = int(train_ratio * n)
            val_end = train_end + int(val_ratio * n)

            self.train += group[:train_end]
            self.val += group[train_end:val_end]
            self.test += group[val_end:]

    def get_split(self, split: str) -> List[Tuple[str, str]]:
        if split == "train":
            return self.train
        elif split == "val":
            return self.val
        elif split == "test":
            return self.test
        else:
            raise ValueError(f"Unknown split: {split}")
        
    @classmethod
    def from_csv(cls, path: str, text_col: str = "text", label_col: str = "label") -> "Dataset":
        import pandas as pd
        df = pd.read_csv(path)
        examples = list(zip(df[text_col], df[label_col]))
        return cls(examples)
    