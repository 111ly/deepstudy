import torch
import torch.nn as nn
from utils.preprocess import load_data, build_vocab, process_data
from models.rnn_model import RNNModel
from models.transformer_model import TransformerModel
import matplotlib.pyplot as plt
import os
import matplotlib
import torch.nn.functional as F

# 设置matplotlib支持中文字体和负号正常显示
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0):
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[..., -1, None]
        logits = torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = 0
        # 这里将 scatter 的维度改为 0，避免维度错误
        indices_to_remove = sorted_mask.scatter(0, sorted_indices, sorted_mask)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    return logits


def generate_poem(model, word2idx, idx2word, temperature=1.0, top_k=0, top_p=0.0,
                  model_type="rnn", start_word="春", max_len=30, repetition_penalty=1.2,
                  repetition_window=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    input_seq = [word2idx.get(ch, word2idx['<UNK>']) for ch in start_word]
    input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)

    result = start_word
    hidden = None

    for _ in range(max_len - len(start_word)):
        if model_type == "rnn":
            output, hidden = model(input_tensor, hidden)
        else:
            output = model(input_tensor)

        logits = output[0, -1, :] / temperature

        # 统计最近 repetition_window 个token出现次数，动态加重惩罚
        recent_tokens = input_tensor[0, -repetition_window:].tolist()
        token_counts = {}
        for token_id in recent_tokens:
            token_counts[token_id] = token_counts.get(token_id, 0) + 1

        for token_id, count in token_counts.items():
            # 指数惩罚，出现次数越多，惩罚越重
            logits[token_id] /= (repetition_penalty ** count)

        filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(filtered_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()

        next_char = idx2word.get(next_id, "<UNK>")
        result += next_char

        next_input = torch.tensor([[next_id]], dtype=torch.long).to(device)
        input_tensor = torch.cat([input_tensor, next_input], dim=1)

    return result



def compute_metrics(texts):
    all_chars = []
    all_bigrams = []
    repeat_count = 0
    total = len(texts)

    for text in texts:
        chars = list(text)
        all_chars.extend(chars)
        all_bigrams.extend(zip(chars, chars[1:]))
        if len(set(text)) < len(text):
            repeat_count += 1

    distinct1 = len(set(all_chars)) / len(all_chars) if all_chars else 0
    distinct2 = len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0
    repeat_rate = repeat_count / total if total else 0

    return {
        "Distinct-1": round(distinct1, 4),
        "Distinct-2": round(distinct2, 4),
        "Repeat Rate": round(repeat_rate, 4)
    }


def load_model(args, vocab_size):
    if args["model_type"] == "rnn":
        model = RNNModel(vocab_size=vocab_size, embed_dim=args["embed_dim"],
                         hidden_dim=args["hidden_dim"], num_layers=args["num_layers"])
        model_path = f"models/rnn_{args['poem_type']}.pth"
    else:
        model = TransformerModel(vocab_size=vocab_size, embed_dim=args["embed_dim"],
                                 nhead=args["nhead"], nhid=args["nhid"], nlayers=args["nlayers"])
        model_path = f"models/transformer_{args['poem_type']}.pth"

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def evaluate_sampling_strategies():
    args = {
        "model_type": "rnn",
        "poem_type": "七言",
        "embed_dim": 128,
        "hidden_dim": 256,
        "num_layers": 2,
        "nhead": 4,
        "nhid": 256,
        "nlayers": 2
    }

    poems = load_data("data/poems.txt")
    vocab, word2idx, idx2word = build_vocab(poems)

    model = load_model(args, vocab_size=len(vocab))

    sampling_settings = [
        {"name": "T=1.4+TopP0.98", "temperature": 1.4, "top_k": 0, "top_p": 0.98},
        {"name": "T=1.3+TopP0.95", "temperature": 1.3, "top_k": 0, "top_p": 0.95},
        {"name": "T=1.2", "temperature": 1.2, "top_k": 0, "top_p": 0.0},
        {"name": "T=1.1+TopK20", "temperature": 1.1, "top_k": 20, "top_p": 0.0},
        {"name": "T=1.0+TopK20+TopP0.95", "temperature": 1.0, "top_k": 20, "top_p": 0.95},
    ]

    results = {}

    for setting in sampling_settings:
        print(f"采样策略: {setting['name']}")
        generated_poems = []
        for _ in range(150):  # 生成150条，稍微多一点
            poem = generate_poem(
                model=model,
                word2idx=word2idx,
                idx2word=idx2word,
                temperature=setting["temperature"],
                top_k=setting["top_k"],
                top_p=setting["top_p"],
                model_type=args["model_type"],
                repetition_penalty=1.4,
                max_len=40
            )
            generated_poems.append(poem)

        metrics = compute_metrics(generated_poems)
        results[setting["name"]] = metrics
        print("质量指标:", metrics)
        print("-" * 50)

    return results


def plot_results(results):
    os.makedirs("results", exist_ok=True)
    strategies = list(results.keys())
    distinct1 = [results[s]["Distinct-1"] for s in strategies]
    distinct2 = [results[s]["Distinct-2"] for s in strategies]
    repeat_rate = [results[s]["Repeat Rate"] for s in strategies]

    x = range(len(strategies))
    bar_width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar([i - bar_width for i in x], distinct1, width=bar_width, label='Distinct-1')
    plt.bar(x, distinct2, width=bar_width, label='Distinct-2')
    plt.bar([i + bar_width for i in x], repeat_rate, width=bar_width, label='Repeat Rate')

    plt.xticks(x, strategies)
    plt.ylabel("Metric Value")
    plt.title("生成策略对比（多样性与重复率）")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("results/sampling_strategy_comparison.png")
    plt.show()


if __name__ == "__main__":
    result_metrics = evaluate_sampling_strategies()
    plot_results(result_metrics)


