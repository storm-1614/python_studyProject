from torch.utils.data import Dataset, DataLoader
import torch

# 创建自定义数据集
class ToyDataset(Dataset):
    def __init__(self, num_samples = 100):
        # 创建一些符合 y = 2x + 噪声 的数据
        self.X = torch.randn(num_samples, 1) * 10
        self.y = 2 * self.X + torch.randn(num_samples, 1) * 2 # y = 2x + noise
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx) :
        return self.X[idx], self.y[idx]


# 实例化数据集
dataset = ToyDataset()

# 创建数据加载器
# batch_size=10: 每个批次包含 10 个样本
# shuffle = True: 在每个 epoch 开始时打乱数据
dataloader = DataLoader(dataset=dataset, batch_size = 10, shuffle = True)

# 我们可以像遍历普通的可迭代对象一样遍历 dataloader
X_batch, Y_batch = next(iter(dataloader))

print(f"一个批次的数据形状: {X_batch.shape}")
print(f"一个批次的标签形状: {Y_batch.shape}")
