
import torch
import torch.nn as nn

# Generate sample data.

seq_len = 1        # Length of the time sequence.
batch_size = 1      # Number of sequences in a batch.
input_size = 5      # Number of features (variables) in the multivariate time series.

# If batch_size = False

data = torch.randn(seq_len, batch_size, input_size)

print(data)

# If batch_size = True

# data = torch.randn(batch_size, seq_len, input_size)

# Define the RNN model.

hidden_size = 20
num_layers = 1

rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = False)

output, h_n = rnn(data)

print("Output Shape: ", output.shape)

print("Hidden State: ", h_n.shape)