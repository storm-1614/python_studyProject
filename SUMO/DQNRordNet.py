import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import collections
import torch.nn.functional as F
import random

device = torch.device("cpu")


class Qnet(torch.nn.Module):
    """
    Q 网络
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.fc(x)


class ReplayBuffer:
    """
    经验回放池
    """

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs):
        self.buffer.append((obs, action, reward, next_obs))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs = zip(*transitions)
        return np.array(obs), action, reward, np.array(next_obs)

    def size(self):
        return len(self.buffer)


class DQN:
    def __init__(
        self,
        obs_dim,
        hidden_dim,
        action_dim,
        learning_rate,
        gamma,
        tau,  # 软更新参数
        epsilon,
        device,
    ):
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.q_net = Qnet(obs_dim, action_dim).to(device)
        self.target_q_net = Qnet(obs_dim, action_dim).to(device)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.device = device
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

    def take_actionn(self, obs):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([obs], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        obs = torch.tensor(transition_dict["obs"], dtype=torch.float).to(self.device)
        actions = (
            torch.tensor(transition_dict["actions"], dtype=torch.long)
            .view(-1, 1)
            .to(self.device)
        )
        rewards = (
            torch.tensor(transition_dict["rewards"], dtype=torch.float)
            .view(-1, 1)
            .to(device)
        )
        next_obs = torch.tensor(transition_dict["next_obs"], dtype=torch.float).to(
            device
        )
        terminated = (
            torch.tensor(transition_dict["terminated"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        truncated = (
            torch.tensor(transition_dict["truncated"], dtype=torch.float)
            .view(-1, 1)
            .to(device)
        )
        done = torch.logical_or(terminated, truncated).float().to(self.device)

        q_values = self.q_net(obs).gather(1, actions)
        max_next_q_values = self.target_q_net(next_obs).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - done)
        dqn_loss = torch.mean(torch.nn.functional.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        for target_param, q_param in zip(
            self.target_q_net.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * q_param.data + (1 - self.tau) * target_param.data
            )


class TwoPhaseIntersection(gym.Env):
    """
    最简 2 相位单路口
    """

    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(2)  # 动作：0/1
        self.observation_space = spaces.MultiDiscrete([5, 5, 2])  # 状态：(5, 5, 2)
        self.phase = 0  # 当前绿灯相位 0/1
        self.queue = [0, 0]  # 两个方向的排队分档

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # 初始化 self.np_random
        self.phase = 0
        self.queue = [0, 0]

        return self._obs(), {}

    def step(self, action):
        # 车辆到达
        for i in range(2):
            self.queue[i] = min(4, int(self.queue[i] + self.np_random.integers(0, 2)))

        # 决策相位
        if action == 1:
            self.phase = 1 - self.phase

        # 绿灯放行
        self.queue[self.phase] = max(0, self.queue[self.phase] - 1)

        # 奖励 + 终止状态
        reward = sum(self.queue) * -1
        terminated = False  # 终止
        truncated = False  # 截断
        info = {}  # 附加调试信息字典

        return self._obs(), reward, terminated, truncated, info

    def _obs(self):
        """
        return:
        queue[0] 方向 A 排队数
        queue[1] 方向 B 排队数
        phase    绿灯相位
        """
        return np.array([self.queue[0], self.queue[1], self.phase], dtype=int)


env = TwoPhaseIntersection()

lr = 2e-3
num_episodes = 501
hidden_dim = 128
gamma = 0.98
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
tau = 0.007

epsilon_min = 0.01
epsilon_decay = 0.995
epsilon_start = 0.1

# 类型检查器把 observation_space 当 Optional、action_space 当通用 Space，
# 实际上自定义环境里运行时一定非空，这里用断言收窄类型（也顺带防御）
assert env.observation_space is not None
assert env.observation_space.shape is not None
assert isinstance(env.action_space, spaces.Discrete)

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

replay_buffer = ReplayBuffer(buffer_size)
agent = DQN(obs_dim, hidden_dim, action_dim, lr, gamma, tau, epsilon_start, device)

for ep in range(num_episodes):
    obs, _ = env.reset()
    total = 0
    for _ in range(100):
        action = agent.take_actionn(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        replay_buffer.add(obs, action, reward, next_obs)  # 2 Bool Value
        obs = next_obs

        if replay_buffer.size() > minimal_size:
            b_o, b_a, b_r, b_no = replay_buffer.sample(batch_size)
            transition_dict = {
                "obs": b_o,
                "actions": b_a,
                "rewards": b_r,
                "next_obs": b_no,
                "terminated": False,
                "truncated": False,
            }
            agent.update(transition_dict)
        obs = next_obs
        total += reward
        if terminated or truncated:
            break
    agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay)
    if ep % 50 == 0:
        print(f"ep{ep} total={total:.0f} eps={agent.epsilon:.2f}")
