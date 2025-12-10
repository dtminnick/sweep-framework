
import torch
import torch.nn as nn

class PlanGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1,
                 num_layers=1, dropout=0.0,
                 embedding_specs=None):
        """
        input_dim: dimension of continuous + ordinal + one-hot features
        hidden_dim: GRU hidden size
        output_dim: usually 1 for binary classification/regression
        num_layers: number of GRU layers
        dropout: dropout between GRU layers
        embedding_specs: dict {feature_name: (num_categories, embedding_dim)}
        """
        super().__init__()
        # build embedding layers if specified
        self.embeddings = nn.ModuleDict()
        if embedding_specs:
            for fname, (num_cats, emb_dim) in embedding_specs.items():
                self.embeddings[fname] = nn.Embedding(num_cats, emb_dim)

        # GRU
        self.gru = nn.GRU(input_dim + self._embedding_total_dim(),
                          hidden_dim,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)

        # output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def _embedding_total_dim(self):
        return sum(emb.embedding_dim for emb in self.embeddings.values())

    def forward(self, sequences, embedding_indices=None):
        """
        sequences: tensor of shape (batch, seq_len, input_dim)
        embedding_indices: dict {feature_name: tensor(batch, seq_len)}
        """
        if self.embeddings and embedding_indices:
            emb_list = []
            for fname, emb_layer in self.embeddings.items():
                idxs = embedding_indices[fname]  # (batch, seq_len)
                emb_list.append(emb_layer(idxs))
            emb_cat = torch.cat(emb_list, dim=-1)  # (batch, seq_len, total_emb_dim)
            x = torch.cat([sequences, emb_cat], dim=-1)
        else:
            x = sequences

        out, _ = self.gru(x)
        logits = self.fc(out[:, -1, :])  # last timestep
        return logits
