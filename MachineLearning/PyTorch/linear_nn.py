from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch


class LinearRegressModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个线性层 nn.Linear(in_features, out_features)
        # in_features=1: 输入的特征维度为 1
        # out_features=1: 输出的特征维度为 1
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        # 定义前向传播路径
        # 直接调用我们定义的层即可
        return self.linear_layer(x)


class ToyDataset(Dataset):
    def __init__(self, num_samples=100):
        # 创建一些符合 y = 2x + 噪声 的数据
        self.X = torch.randn(num_samples, 1) * 10
        self.y = 2 * self.X + torch.randn(num_samples, 1) * 2  # y = 2x + noise
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 实例化模型、损失函数、优化器
model = LinearRegressModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# 实例化数据加载器
dataloader = DataLoader(dataset=ToyDataset(), batch_size=10, shuffle=True)

# 训练循环：
num_epochs = 10  # 训练 10 个周期

for epoch in range(num_epochs):
    # 内层循环遍历数据加载器
    for X_batch, Y_batch in dataloader:
        # 1. 前向传播
        y_pred = model(X_batch)
        # 2. 计算损失
        loss = loss_fn(y_pred, Y_batch)

        # 3. 梯度清零
        optimizer.zero_grad()

        # 4. 反向传播
        loss.backward()

        # 5. 参数更新
        optimizer.step()

    # 在每个周期结束后，打印损失值以监控训练过程
    if (epoch + 1) % 1 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


# 查看训练后的模型参数，应该接近 y=2x 的 w=2, b=0
print("\n训练后的模型参数:")
for name, param in model.named_parameters():
    print(f"参数名称: {name}, 参数值: {param.data}")
