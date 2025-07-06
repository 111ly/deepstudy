# 运⾏说明
直接在命令行输入命令：uvicorn app:app --reload 然后在浏览器访问http://localhost:8000
# 参数设置
main.py用于训练，具体的参数设置如下：
1.模型类型：
model_type：选择模型架构（RNN 或 Transformer）
poem_type：生成古诗的类型（五言或七言绝句）
2.数据配置参数：
batch_size：每次训练输入的样本数量（默认32条诗句一起处理）
3.训练参数：
epochs：训练轮次（默认20轮）
lr：学习率（默认0.001），控制参数更新的步长
4. 模型结构参数：
(1) RNN 专用参数：
embed_dim：每个汉字用多少维的数字表示（默认128维）
hidden_dim：RNN隐藏状态的维度（默认256维）
num_layers：堆叠的RNN层数（默认2层）
(2) Transformer 专用参数：
nhead：多头注意力的头数（默认4头）
nhid：前馈神经网络的隐藏层维度（默认256维）
nlayers：Transformer编码器的层数（默认2层）
备注：针对默认的参数，可以通过在命令行设置参数：
(1) 训练一个五言绝句的RNN模型
python train.py --model_type rnn --poem_type 五言 --epochs 30 --hidden_dim 512
(2) 训练一个七言绝句的Transformer模型
python train.py --model_type transformer --poem_type 七言 --nhead 8 --nhid 512
# 依赖列表
见requirements.txt
