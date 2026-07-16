from torch import nn
import torch

class LinearRegressModel(nn.Module):
    """线性回归模型"""
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.linear_layer(x)
    
# 实例化一个模型
model = LinearRegressModel()

# 设定学习率
learning_rate = 0.01

# 创建一个 SGD 优化器，将模型的所有参数和学习率传入
optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)
print(optimizer)

