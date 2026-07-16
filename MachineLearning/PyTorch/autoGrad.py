import torch

w_true = 2
x: torch.Tensor = torch.tensor(1.0)
y_true = torch.tensor(2.0)

# 初始化w，并通过 requires_grad=True 告诉PyTorch需要追踪其梯度
w: torch.Tensor = torch.tensor(0.5, requires_grad=True)
learning_rate = 0.1

print(f"学习开始前， w = {w.item():.3f}")

for epoch in range(10):
    # 1. 前向传播
    y_pred = w * x
    # 2. 计算损失
    loss = (y_pred - y_true) ** 2

    # 3. 自动计算梯度
    loss.backward()

    # 4. 更新权重
    # w.grad 中存储了 loss 对 w 的梯度
    # 使用 torch.no_grad() 确保更新操作本身不被追踪
    with torch.no_grad():
        assert w.grad is not None
        w -= learning_rate * w.grad

    # 清空梯度，为下一次迭代做准备
    assert w.grad is not None
    w.grad.zero_()

    # .item() 用于从单元素张量总获取 Python 数字
    print(f"第 {epoch + 1} 轮: w = {w.item():.3f}, loss = {loss.item():.3f}")
