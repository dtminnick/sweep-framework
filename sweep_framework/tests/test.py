
from datasets import load_dataset


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from sweep_framework.model.model_run import ModelRun

dataset = load_dataset("stanfordnlp/sst2")

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class SST2TorchDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_len=128):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = self.tokenizer(
            item["sentence"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input": tokens["input_ids"].squeeze(0),
            "target": item["label"]
        }

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
train_ds = SST2TorchDataset(dataset["train"], tokenizer)
val_ds = SST2TorchDataset(dataset["validation"], tokenizer)
test_ds = SST2TorchDataset(dataset["test"], tokenizer)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)
test_loader = DataLoader(test_ds, batch_size=32)

class Dataset:
    def __init__(self, train_loader, val_loader, test_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

from sweep_framework.config.model_config import ModelConfig

config = ModelConfig(
    run_type="LSTM",
    # loss_type="focal",
    learning_rate=1e-4,
    num_epochs=1,
    patience=2,
    optimizer_type="Adam",
    # scheduler_type="linear",
    hidden_dim=256,
    dropout=0.3
)

run = ModelRun(config, Dataset(train_loader, val_loader, test_loader))
run.run()
print(run.export_summary())
