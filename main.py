# main.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.preprocess import load_data, build_vocab, process_data, PoemDataset
from models.rnn_model import RNNModel
from models.transformer_model import TransformerModel
import os
import matplotlib.pyplot as plt  # ✅ 新增

# 设置随机种子
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def train_model(model, dataloader, vocab_size, epochs=10, lr=0.001, model_type="rnn"):
    """训练模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    loss_list = []     # ✅ 用于绘图
    ppl_list = []      # ✅ 用于绘图

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
        ppl = torch.exp(torch.tensor(avg_loss)).item()  # ✅ Perplexity = e^loss

        loss_list.append(avg_loss)
        ppl_list.append(ppl)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, PPL: {ppl:.2f}")
    
    # ✅ 绘图：loss 和 PPL 两张图放一窗口
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(loss_list, label='Loss', color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss") 
    plt.title("Training Loss")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(ppl_list, label='Perplexity', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("Training Perplexity (PPL)")
    plt.grid(True)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/loss_ppl.png")   # ✅ 保存图像
    plt.show()                            # ✅ 显示图像

    return model

def main(args):
    poems = load_data("data/poems.txt")
    vocab, word2idx, idx2word = build_vocab(poems)
    max_len = 30 if args.poem_type == "五言" else 40
    processed_poems = process_data(poems, word2idx, poem_type=args.poem_type)
    
    dataset = PoemDataset(processed_poems, vocab, word2idx, idx2word, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    vocab_size = len(vocab)
    
    if args.model_type == "rnn":
        model = RNNModel(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers
        )
        model_path = f"models/rnn_{args.poem_type}.pth"
    else:
        model = TransformerModel(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            nhead=args.nhead,
            nhid=args.nhid,
            nlayers=args.nlayers
        )
        model_path = f"models/transformer_{args.poem_type}.pth"
    
    os.makedirs("models", exist_ok=True)

    print(f"开始训练 {args.model_type.upper()} 模型（{args.poem_type}）...")
    trained_model = train_model(
        model=model,
        dataloader=dataloader,
        vocab_size=vocab_size,
        epochs=args.epochs,
        lr=args.lr,
        model_type=args.model_type
    )
    
    torch.save(trained_model.state_dict(), model_path)
    print(f"模型已保存至: {model_path}")

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
