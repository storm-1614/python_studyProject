"""直接加载已有 PPO 模型进行演示，不重新训练"""
import gym
import torch
import torch.nn.functional as F
import numpy as np
import os


class PolicyNet(torch.nn.Module):
    """策略网络（Actor），与 cartPole.py 保持一致"""
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


# 超参数（需与训练时一致）
hidden_dim = 128
device = torch.device("cpu")

# 创建环境和网络
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)

# 加载已有模型
model_path = "./output/ppo_actor.pth"
if not os.path.exists(model_path):
    print(f"模型文件不存在: {model_path}")
    print("请先运行 cartPole.py 训练并保存模型")
    exit(1)

actor.load_state_dict(torch.load(model_path, map_location=device))
actor.eval()
print(f"模型已从 {model_path} 加载")

# 跑几个 episode 看看效果
num_test = 5
total_rewards = []

for ep in range(num_test):
    state = env.reset()
    episode_reward = 0
    done = False

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        # PPO 输出概率分布，取概率最大的动作作为确定性策略
        action = actor(state_tensor).argmax(dim=1).item()
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
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    action = actor(state_tensor).argmax(dim=1).item()
    state, reward, done, _ = env.step(action)

gif_path = "./output/video/ppo_cartpole_demo.gif"
os.makedirs("./output/video", exist_ok=True)
imageio.mimsave(gif_path, frames, fps=30)
print(f"演示 GIF 已保存到 {gif_path}")

env.close()
