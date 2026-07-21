import gym
import os
import random
import torch.nn as nn
import torch
import collections
import numpy as np

import imageio
from PIL import Image

device = torch.device("cpu")


# 设置随机数种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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


class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.q_net = QNetwork(state_dim, action_dim)  # 当前网络
        self.target_net = QNetwork(state_dim, action_dim)  # 目标网络
        self.target_net.load_state_dict(
            self.q_net.state_dict()
        )  # 将目标网络和当前网络初始化一致，避免网络不一致导致训练波动
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.replay_buffer = collections.deque(maxlen=10000)  # 经验回放区
        self.batch_size = 64
        self.gamma = 0.99 # 学习率
        self.epsilon = 0.1 # epsilon-贪婪函数因子
        self.update_target_freq = 100  # 目标网络更新频率
        self.step_count = 0
        self.best_reward = 0
        self.best_avg_reward = 0
        self.eval_episodes = 5  # 评估时的 episode 数量

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 2)  # CartPole 左右两个动作
        else:
            state_tensor = torch.FloatTensor(state)
            q_values = self.q_net(state_tensor)
        return q_values.cpu().detach().numpy().argmax()

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # 从缓冲区随机采样
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # 计算当前 Q 值
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # 计算目标 Q 值（使用目标网络)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # 计算损失并更新网络
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 定期更新目标网络
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            # 使用深拷贝更新目标网络参数
            self.target_net.load_state_dict(
                {k: v.clone() for k, v in self.q_net.state_dict().items()}
            )

    def save_model(self, path="./output/best_model.pth"):
        if not os.path.exists("./output"):
            os.makedirs("./output")
        torch.save(self.q_net.state_dict(), path)
        print(f"Model saved to {path}")

    def evaluate(self, env):
        """评估当前模型的性能"""
        original_epsilon = self.epsilon
        self.epsilon = 0  # 关闭探索
        total_rewards = []

        for _ in range(self.eval_episodes):
            state = env.reset()
            episode_reward = 0
            while True:
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                state = next_state
                if done or episode_reward > 2e4:
                    break
            total_rewards.append(episode_reward)

        self.epsilon = original_epsilon  # 恢复探索
        return np.mean(total_rewards)


env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim)

# 训练过程
config = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    "batch_size": agent.batch_size,
    "gamma": agent.gamma,
    "epsilon": agent.epsilon,
    "update_target_freq": agent.update_target_freq,
    "replay_buffer_size": agent.replay_buffer.maxlen,
    "learning_rate": agent.optimizer.param_groups[0]["lr"],
    "episode": 600,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
}
print(f"Starting DQN training on CartPole-v1, config: {config}")

num_episodes = config["episode"]
epsilon = config["epsilon_start"]

for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    done = False

    while not done:
        # epsilon-greedy 选择动作
        action = agent.choose_action(state)

        next_state, reward, done, _ = env.step(action)
        episode_reward += reward

        # 存储经验并训练
        agent.store_experience(state, action, reward, next_state, done)
        agent.train()
        state = next_state

    # epsilon 衰减
    epsilon = max(config["epsilon_end"], epsilon * config["epsilon_decay"])

    if (episode + 1) % 50 == 0:
        avg_reward = agent.evaluate(env)
        print(
            f"Episode {episode + 1}/{num_episodes}, "
            f"Reward: {episode_reward}, "
            f"Avg Eval Reward: {avg_reward:.2f}, "
            f"Epsilon: {epsilon:.4f}"
        )

agent.save_model("./output/best_model.pth")
env.close()
print("Training finished!")

# ====== 渲染最终结果并保存视频 ======
eval_env = gym.make("CartPole-v1")
state = eval_env.reset()
total_reward = 0
done = False

# 录制视频帧
frames = []
agent.epsilon = 0  # 关闭探索，纯利用


while not done:
    frame = eval_env.render(mode="rgb_array")
    frames.append(np.array(Image.fromarray(frame).resize((600, 400))))
    state_tensor = torch.FloatTensor(state).to(device)
    action = agent.q_net(state_tensor).argmax().item()
    state, reward, done, _ = eval_env.step(action)
    total_reward += reward

gif_path = "./output/video/dqn_cartpole_demo.gif"
os.makedirs("./output/video", exist_ok=True)
imageio.mimsave(gif_path, frames, fps=30)
print(f"演示 GIF 已保存到 {gif_path}")

eval_env.close()
