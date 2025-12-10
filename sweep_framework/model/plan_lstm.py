
class PlanLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1,
                 num_layers=1, dropout=0.0,
                 embedding_specs=None):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        if embedding_specs:
            for fname, (num_cats, emb_dim) in embedding_specs.items():
                self.embeddings[fname] = nn.Embedding(num_cats, emb_dim)

        self.lstm = nn.LSTM(input_dim + self._embedding_total_dim(),
                            hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def _embedding_total_dim(self):
        return sum(emb.embedding_dim for emb in self.embeddings.values())

    def forward(self, sequences, embedding_indices=None):
        if self.embeddings and embedding_indices:
            emb_list = []
            for fname, emb_layer in self.embeddings.items():
                idxs = embedding_indices[fname]
                emb_list.append(emb_layer(idxs))
            emb_cat = torch.cat(emb_list, dim=-1)
            x = torch.cat([sequences, emb_cat], dim=-1)
        else:
            x = sequences

        out, _ = self.lstm(x)
        logits = self.fc(out[:, -1, :])
        return logits
