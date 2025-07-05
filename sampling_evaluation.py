import torch
import torch.nn as nn
from utils.preprocess import load_data, build_vocab, process_data
from models.rnn_model import RNNModel
from models.transformer_model import TransformerModel
from main import generate_poem  # 直接复用你已有的生成函数
import matplotlib.pyplot as plt
import os
import matplotlib
import matplotlib.pyplot as plt

# 设置matplotlib支持中文字体和负号正常显示
matplotlib.rcParams['font.family'] = 'SimHei'  # 或者你系统中支持的中文字体，比如 'SimSun'
matplotlib.rcParams['axes.unicode_minus'] = False


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
    # 参数
    args = {
        "model_type": "rnn",       # 改为 "transformer" 可测试 Transformer
        "poem_type": "五言",       # 或 "五言"
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

    # 设置采样策略
    sampling_settings = [
    {"name": "T=1.2", "temperature": 1.2, "top_k": 0, "top_p": 0.0},           # 最自由，温度最高，无限制
    {"name": "T=1.0", "temperature": 1.0, "top_k": 0, "top_p": 0.0},           # 自由，普通随机采样
    {"name": "T=0.8+TopP0.9", "temperature": 0.8, "top_k": 0, "top_p": 0.9},   # 半自由，核采样控制多样性
    {"name": "T=0.7+TopK10", "temperature": 0.7, "top_k": 10, "top_p": 0.0},   # 半严谨，温度和Top-K限制采样范围
    {"name": "T=0.6+TopK5", "temperature": 0.6, "top_k": 5, "top_p": 0.0},     # 最严谨，温度低+Top-K严格限制
]


    results = {}

    for setting in sampling_settings:
        print(f"采样策略: {setting['name']}")
        generated_poems = []
        for _ in range(100):  # 生成 100 首
            poem = generate_poem(
                model=model,
                word2idx=word2idx,
                idx2word=idx2word,
                temperature=setting["temperature"],
                top_k=setting["top_k"],
                top_p=setting["top_p"],
                model_type=args["model_type"]
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
