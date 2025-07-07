import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ===== 位置编码模块 =====
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # shape: (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ===== Transformer模型定义 =====
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, nhid, nlayers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        self.decoder = nn.Linear(embed_dim, vocab_size)
        self.embed_dim = embed_dim

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def forward(self, src):
        # src shape: (batch_size, seq_len)
        x = self.embedding(src) * math.sqrt(self.embed_dim)  # (batch_size, seq_len, embed_dim)
        x = self.pos_encoder(x)
        seq_len = src.size(1)
        mask = self.generate_square_subsequent_mask(seq_len).to(src.device)
        out = self.transformer_encoder(x, mask=mask)
        out = self.decoder(out)  # (batch_size, seq_len, vocab_size)
        return out


# ===== Top-K / Top-P / Temperature采样函数 =====
def sample_next_token(logits, temperature=1.0, top_k=0, top_p=0.0):
    logits = logits / temperature

    # Top-k
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        threshold = values[..., -1, None]
        logits = torch.where(logits < threshold, torch.full_like(logits, -float('inf')), logits)

    # Top-p (nucleus sampling)
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, -float('inf'))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ===== 文本生成函数 =====
def generate(model, start_token, max_len=50, temperature=1.0, top_k=0, top_p=0.0, device='cpu'):
    model.eval()
    input_ids = torch.tensor([[start_token]], dtype=torch.long, device=device)

    for _ in range(max_len):
        with torch.no_grad():
            output = model(input_ids)  # (1, seq_len, vocab_size)
            logits = output[:, -1, :]  # 最后一个时间步的输出
            next_token = sample_next_token(logits, temperature, top_k, top_p)  # (1, 1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids[0].tolist()