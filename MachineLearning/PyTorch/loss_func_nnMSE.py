from torch import nn
import torch

# 实例化一个损失函数

loss_fn = nn.MSELoss()

# 假设模型预测值为 y_pred，真实标签为 y_true
y_pred = torch.tensor([2.5])
y_true = torch.tensor([2.0])

# 计算损失
loss = loss_fn(y_pred, y_true)
print(f"损失值: {loss.item():.4f}") # (2.5 - 2.0)^2 = 0.25
