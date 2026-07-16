# 设定一个真实的目标 w
w_true = 2
# 输入数据 x 和真实标签 y_true
x = 1.0
y_true = 2.0

# 初始化一个随机的参数 w
w = 0.5
# 设定学习率
learning_rate = 0.1

print(f"学习开始前, w = {w:.3f}")

# 进行10次迭代学习
for epoch in range(10):
    # 1. 前向传播：计算预测值
    y_pred = w * x

    # 2. 计算损失
    loss = (y_pred - y_true) ** 2

    # 3. 手动计算梯度
    grad = 2 * x * (y_pred - y_true)

    # 4. 更新权重：向梯度的反方向迈出一步
    w = w - learning_rate * grad

    print(f"第 {epoch + 1} 轮: w = {w:.3f}, loss = {loss:.3f}")
