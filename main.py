import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.preprocess import load_data, build_vocab, process_data, PoemDataset
from models.rnn_model import RNNModel
from models.transformer_model import TransformerModel
import os
import math
import matplotlib.pyplot as plt

def evaluate_ppl(model, dataloader, vocab_size, model_type="rnn"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            if model_type == "rnn":
                outputs, _ = model(inputs, None)
            else:
                outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return math.exp(avg_loss)

def sample_next_token(logits, temperature=1.0, top_k=0, top_p=0.0):
    import torch.nn.functional as F
    if temperature > 0:
        logits = logits / temperature
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = -float('Inf')
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token

def generate_poem(model, word2idx, idx2word, start_token="<START>", max_len=30, temperature=1.0, top_k=0, top_p=0.0, model_type="rnn"):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_seq = [word2idx[start_token]]
    input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)

    generated = []
    hidden = None
    for _ in range(max_len):
        if model_type == "rnn":
            output, hidden = model(input_tensor, hidden)
        else:
            output = model(input_tensor)
        logits = output[:, -1, :]
        next_token = sample_next_token(logits.squeeze(0), temperature=temperature, top_k=top_k, top_p=top_p)
        next_token_id = next_token.item()
        generated.append(next_token_id)
        if idx2word[next_token_id] == "<END>":
            break
        input_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(device)

    return "".join([idx2word[idx] for idx in generated if idx2word[idx] not in ["<PAD>", "<START>", "<END>"]])

def train_model(model, dataloader, vocab_size, epochs=10, lr=0.001, model_type="rnn"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    ppls = []
    
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            if model_type == "rnn":
                hidden = None
                outputs, _ = model(inputs, hidden)
            else:
                outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        model.eval()
        ppls.append(math.exp(avg_loss))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, PPL: {ppls[-1]:.2f}")
    
    return model, losses, ppls

def plot_training_metrics(losses, ppls, model_type="rnn"):
    epochs = list(range(1, len(losses)+1))
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label="Train Loss")
    plt.title(f"{model_type.upper()} Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, ppls, label="Perplexity", color='orange')
    plt.title(f"{model_type.upper()} Perplexity")
    plt.xlabel("Epoch")
    plt.ylabel("PPL")
    plt.grid(True)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{model_type}_training_curve.png")
    plt.show()

def main(args):
    poems = load_data("data/poems.txt")
    vocab, word2idx, idx2word = build_vocab(poems)
    max_len = 30 if args.poem_type == "五言" else 40
    processed_poems = process_data(poems, word2idx, poem_type=args.poem_type)
    dataset = PoemDataset(processed_poems, vocab, word2idx, idx2word, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    vocab_size = len(vocab)
    
    if args.model_type == "rnn":
        model = RNNModel(vocab_size=vocab_size, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
        model_path = f"models/rnn_{args.poem_type}.pth"
    else:
        model = TransformerModel(vocab_size=vocab_size, embed_dim=args.embed_dim, nhead=args.nhead, nhid=args.nhid, nlayers=args.nlayers)
        model_path = f"models/transformer_{args.poem_type}.pth"
    
    os.makedirs("models", exist_ok=True)
    
    print(f"开始训练 {args.model_type.upper()} 模型（{args.poem_type}）...")
    trained_model, losses, ppls = train_model(model, dataloader, vocab_size, epochs=args.epochs, lr=args.lr, model_type=args.model_type)
    
    torch.save(trained_model.state_dict(), model_path)
    print(f"模型已保存至: {model_path}")
    
    # 评估PPL
    ppl = evaluate_ppl(trained_model, dataloader, vocab_size, model_type=args.model_type)
    print(f"{args.model_type.upper()} 模型困惑度（PPL）: {ppl:.2f}")
    
    # 绘制训练曲线
    plot_training_metrics(losses, ppls, model_type=args.model_type)
    
    # 示例生成
    poem = generate_poem(trained_model, word2idx, idx2word, temperature=0.8, top_k=10, top_p=0.9, model_type=args.model_type)
    print("示例生成诗句:", poem)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="古诗生成模型训练")
    parser.add_argument("--model_type", type=str, default="rnn", choices=["rnn", "transformer"])
    parser.add_argument("--poem_type", type=str, default="五言", choices=["五言", "七言"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--nhid", type=int, default=256)
    parser.add_argument("--nlayers", type=int, default=2)
    args = parser.parse_args()
    main(args)
