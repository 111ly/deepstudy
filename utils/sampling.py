import torch
import torch.nn.functional as F

def sample_next_token(logits, temperature=1.0, top_k=0, top_p=0.0):
    """采样下一个token，支持temperature、top-k和top-p三种策略"""
    logits = logits.clone()  # 避免修改原始logits
    
    if temperature > 0:
        # Temperature采样
        logits = logits / temperature
        
    if top_k > 0:
        # Top-K采样
        top_k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k)[0][..., -1, None]
        indices_to_remove = logits < threshold
        logits[indices_to_remove] = -float('Inf')
        
    if top_p > 0.0:
        # Top-P采样
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # 移除累积概率高于top_p的token
        sorted_indices_to_remove = cumulative_probs > top_p
        # 保留第一个token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # 构建要移除的索引掩码
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = -float('Inf')
    
    # 从分布中采样
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token

def generate_poem(model, word2idx, idx2word, max_len=30, temperature=1.0, top_k=0, top_p=0.0, model_type='rnn'):
    model.eval()
    device = next(model.parameters()).device
    input_idx = torch.tensor([[word2idx['<START>']]], dtype=torch.long).to(device)
    generated = []

    hidden = None
    for _ in range(max_len):
        if model_type == 'rnn':
            output, hidden = model(input_idx, hidden)
        else:
            output = model(input_idx)
        logits = output[:, -1, :]
        next_token = sample_next_token(logits, temperature, top_k, top_p)
        next_id = next_token.item()
        if next_id == word2idx['<END>']:
            break
        generated.append(idx2word.get(next_id, '<UNK>'))
        input_idx = torch.cat([input_idx, next_token.unsqueeze(0)], dim=1)
    return "".join(generated)
