
# sweep_framework/data/plan_dataset.py
from sweep_framework.data.base_dataset import BaseDataset
from torch.utils.data import DataLoader
import torch

class PlanDataset(BaseDataset):
    """
    Dataset class for retirement plan dynamics.
    Converts numeric time-series into tensors for LSTM/GRU models.
    """

    def stratify_split(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
        # implement stratified split for plan examples
        ...

    def build_loaders(self, batch_size=32, seq_len=12):
        # assume examples are (sequence_of_features, label)
        def collate_fn(batch):
            sequences, labels = zip(*batch)
            return torch.stack(sequences), torch.tensor(labels)

        self.train_loader = DataLoader(self.train_examples, batch_size=batch_size, collate_fn=collate_fn)
        self.val_loader   = DataLoader(self.val_examples, batch_size=batch_size, collate_fn=collate_fn)
        self.test_loader  = DataLoader(self.test_examples, batch_size=batch_size, collate_fn=collate_fn)
