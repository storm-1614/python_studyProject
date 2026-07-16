import torch
import torch.nn as nn
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['Source Han Sans CN']
plt.rcParams['axes.unicode_minus'] = False

def plot_activation(fn, title):
    x = torch.linspace(-8, 8, 400)
    y = fn(x)
    plt.figure(figsize=(6, 4))
    plt.plot(x.numpy(), y.numpy())
    plt.title(title, fontsize=14)
    plt.xlabel('输入 (x)')
    plt.ylabel('输出 (f(x))')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.show()

# 绘制Sigmoid
plot_activation(nn.ReLU6(), 'ReLU 激活函数')
