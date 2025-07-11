# 简介
这是一个基于PyTorch的智能古诗生成系统，支持RNN和Transformer两种深度学习模型，能够按需生成五言或七言绝句。用户可通过调节温度值、Top-K/Top-P等参数控制诗句风格。
# 项目界面
![屏幕截图 2025-07-07 144630](https://github.com/user-attachments/assets/beaffc5a-227a-4d6b-b0fb-53af4c20d716)
![image](https://github.com/user-attachments/assets/41b78891-c480-4327-a125-c00ed46af3e4)



# 运⾏说明

- 直接在命令行输入命令：
   ```bash
   uvicorn app:app --reload
- 然后在浏览器访问http://localhost:8000

# 参数设置

`main.py` 用于训练，具体参数如下：

### 1. 模型类型
- `model_type`：选择模型架构（RNN 或 Transformer）  
- `poem_type`：生成古诗的类型（五言或七言绝句）

### 2. 数据配置参数
- `batch_size`：每次训练的样本数量（默认32条诗句一起处理）

### 3. 训练参数
- `epochs`：训练轮次（默认20轮）  
- `lr`：学习率（默认0.001），控制参数更新步长

### 4. 模型结构参数
#### RNN 专用参数
- `embed_dim`：汉字向量维度（默认128维）  
- `hidden_dim`：RNN隐藏状态维度（默认256维）  
- `num_layers`：堆叠的RNN层数（默认2层）

#### Transformer 专用参数
- `nhead`：多头注意力头数（默认4头）  
- `nhid`：前馈网络隐藏层维度（默认256维）  
- `nlayers`：Transformer编码器层数（默认2层）

---

### 使用示例
1. 训练五言绝句RNN模型：  
   ```bash
   python train.py --model_type rnn --poem_type 五言 --epochs 30 --hidden_dim 512

2. 训练七言绝句的Transformer模型：  
   ```bash
   python train.py --model_type transformer --poem_type 七言 --nhead 8 --nhid 512

# 依赖列表
见requirements.txt
