
# --- Imports ---
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# --- Step 1: Build a toy dataset ---
# Each row = one month of data for one plan.
# Columns:
#   - month: timestamp (used for ordering, not fed into the model)
#   - plan_id: unique identifier for the plan (used for grouping, not fed into the model)
#   - input_var1, input_var2: dynamic features that vary month to month
#   - static_var1, static_var2: static features constant per plan (e.g. plan size, vendor type)
#   - label: one class per plan (repeated per row for convenience)
df = pd.DataFrame({
    "month": ["2025-01","2025-02","2025-03",
              "2025-01","2025-02","2025-03",
              "2025-01","2025-02","2025-03"],
    "plan_id": [123456,123456,123456,
                654321,654321,654321,
                987654,987654,987654],
    "input_var1": [1,1,1,1,2,3,2,3,4],
    "input_var2": [1,2,3,2,3,4,1,3,4],
    "static_var1": [1,1,1, 2,2,2, 3,3,3],  # e.g. plan size
    "static_var2": [0,0,0, 1,1,1, 2,2,2],  # e.g. vendor type
    "label": [0,0,0, 1,1,1, 2,2,2]
})

# --- Step 2: Group rows into sequences per plan ---
# For each plan_id:
#   - Sort rows by month to preserve temporal order
#   - Extract dynamic features into a tensor (seq_len, num_dynamic_features)
#   - Extract static features into a vector (num_static_features)
#   - Store one label per plan
plan_sequences = []
plan_static = []
labels = []

for pid, group in df.groupby("plan_id"):
    group = group.sort_values("month")
    
    # Dynamic sequence (monthly features)
    seq = group[["input_var1", "input_var2"]].values
    plan_sequences.append(torch.tensor(seq, dtype=torch.float32))
    
    # Static vector (constant per plan, take first row)
    static = group[["static_var1", "static_var2"]].iloc[0].values
    plan_static.append(torch.tensor(static, dtype=torch.float32))
    
    # Label (one per plan)
    labels.append(group["label"].iloc[0])

labels = torch.tensor(labels)

# Debug: print each plan’s sequence tensor
for i, tensor in enumerate(plan_sequences):
    print(f"Plan {i}: shape {tensor.shape}")
    print(tensor)

# --- Step 3: Define recurrent models ---
# Three variants: vanilla RNN, LSTM, GRU.
# Each processes dynamic sequences and outputs a hidden representation.
# LSTMWithStatic additionally concatenates static features before classification.

class SimpleRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)              # out: (batch_size, seq_len, hidden_dim)
        last_hidden = out[:, -1, :]       # take last timestep
        return self.fc(last_hidden)       # (batch_size, output_dim)

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)

class LSTMWithStatic(nn.Module):
    def __init__(self, dynamic_dim, static_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(dynamic_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim + static_dim, output_dim)

    def forward(self, seq, static):
        out, (h_n, c_n) = self.lstm(seq)        # (batch_size, seq_len, hidden_dim)
        last_hidden = out[:, -1, :]             # (batch_size, hidden_dim)
        combined = torch.cat([last_hidden, static], dim=1)  # concat dynamic + static
        return self.fc(combined)

class SimpleGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, h_n = self.gru(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)

# --- Step 4: Batch sequences ---
# pad_sequence ensures all plans have equal length in the batch.
padded = pad_sequence(plan_sequences, batch_first=True)  # (batch_size, max_seq_len, dynamic_dim)
static_batch = torch.stack(plan_static)                  # (batch_size, static_dim)
print(padded.shape)

# --- Step 5: Training setup ---
model = LSTMWithStatic(dynamic_dim=2, static_dim=2, hidden_dim=8, output_dim=3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# --- Step 6: Training loop ---
# Run multiple epochs to minimize loss and improve accuracy.
for epoch in range(50):
    optimizer.zero_grad()
    logits = model(padded, static_batch)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    # Accuracy: compare predicted class vs true label
    preds = torch.argmax(logits, dim=1)
    acc = (preds == labels).float().mean().item()
    print(f"Epoch {epoch:02d}, Loss: {loss.item():.4f}, Accuracy: {acc:.2f}")

# --- Step 7: Inference on a new plan ---
# Prepare new plan’s sequence (dynamic features) and static vector.
new_plan = pd.DataFrame({
    "month": ["2025-01","2025-02","2025-03",
              "2025-01","2025-02","2025-03"],
    "plan_id": [111111,111111,111111,
                222222,222222,222222],
    "input_var1": [2,3,4,3,2,5],
    "input_var2": [5,6,7,6,7,5],
    "static_var1": [15,15,15,3,3,3],  # e.g. plan size
    "static_var2": [1,1,1,2,2,2]      # e.g. vendor type
})

# Group by plan_id just like training
seq_list = []
static_list = []
plan_ids = []

for pid, group in new_plan.groupby("plan_id"):
    group = group.sort_values("month")
    
    # Dynamic sequence
    seq = group[["input_var1","input_var2"]].values
    seq_list.append(torch.tensor(seq, dtype=torch.float32))
    
    # Static vector (constant per plan)
    static = group[["static_var1","static_var2"]].iloc[0].values
    static_list.append(torch.tensor(static, dtype=torch.float32))
    
    plan_ids.append(pid)

# Pad sequences and stack static features
padded_seq = pad_sequence(seq_list, batch_first=True)   # (batch_size, max_seq_len, dynamic_dim)
static_batch = torch.stack(static_list)                 # (batch_size, static_dim)

# Predict classes for all plans
model.eval()
with torch.no_grad():
    logits = model(padded_seq, static_batch)            # (batch_size, output_dim)
    pred_classes = torch.argmax(logits, dim=1).tolist()

# Map predictions back to plan_ids
for pid, pred in zip(plan_ids, pred_classes):
    print(f"Plan {pid} predicted class: {pred}")
