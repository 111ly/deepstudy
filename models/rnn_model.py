import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.3):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout)
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)                       # (batch, seq_len, embed_dim)
        output, hidden = self.rnn(embedded, hidden)        # output: (batch, seq_len, hidden_dim)
        output = self.ln(output)                           # LayerNorm
        output = self.dropout(output)                      # Dropout
        output = self.fc(output)                           # (batch, seq_len, vocab_size)
        return output, hidden