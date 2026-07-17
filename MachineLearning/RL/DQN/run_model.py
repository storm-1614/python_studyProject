"""直接加载已有模型进行演示，不重新训练"""
import gym
import torch
import torch.nn as nn
import numpy as np
import os

device = torch.device("cpu")


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self.to(device)

    def forward(self, x):
        return self.fc(x)


# 创建环境和网络
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

q_net = QNetwork(state_dim, action_dim)

# 加载已有模型
model_path = "./output/best_model.pth"
if not os.path.exists(model_path):
    print(f"模型文件不存在: {model_path}")
    print("请先运行 cartPole.py 训练模型")
    exit(1)

q_net.load_state_dict(torch.load(model_path, map_location=device))
q_net.eval()
print(f"模型已从 {model_path} 加载")

# 跑几个 episode 看看效果
num_test = 5
total_rewards = []

for ep in range(num_test):
    state = env.reset()
    episode_reward = 0
    done = False

    while not done:
        state_tensor = torch.FloatTensor(state).to(device)
        action = q_net(state_tensor).argmax().item()
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        state = next_state

    total_rewards.append(episode_reward)
    print(f"Episode {ep + 1}: reward = {episode_reward}")

print(f"\n平均 reward: {np.mean(total_rewards):.1f} (max=500 表示完美)")
print(f"最高 reward: {max(total_rewards)}")

# 保存 GIF 演示
import imageio
from PIL import Image

state = env.reset()
done = False
frames = []

while not done:
    frame = env.render(mode="rgb_array")
    # gym 0.21 CartPole 首帧尺寸与其他帧不同，统一 resize
    frames.append(np.array(Image.fromarray(frame).resize((600, 400))))
    state_tensor = torch.FloatTensor(state).to(device)
    action = q_net(state_tensor).argmax().item()
    state, reward, done, _ = env.step(action)

gif_path = "./output/video/dqn_cartpole_demo.gif"
os.makedirs("./output/video", exist_ok=True)
imageio.mimsave(gif_path, frames, fps=30)
print(f"演示 GIF 已保存到 {gif_path}")

env.close()