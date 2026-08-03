"""
Q Learning 单路口仿真
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


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
        terminated = False # 终止
        truncated = False  # 截断
        info = {} # 附加调试信息字典

        return self._obs(), reward, terminated, truncated, info

    def _obs(self):
        """
        return: 
        queue[0] 方向 A 排队数
        queue[1] 方向 B 排队数
        phase    绿灯相位
        """
        return np.array([self.queue[0], self.queue[1], self.phase], dtype=int)


class QLearning:
    def __init__(
        self, n_states=(5, 5, 2), n_actions=2, alpha=0.1, gamma=0.9, espilon=0.3
    ):
        """
        Q 表：
        把两个元组拼起来，然后生成一张 4 维全零表
        (5,5,2) + (2,) -> (5, 5, 2, 2)
        变成：[queue0, queue1, phase, action]
        """
        self.qTable = np.zeros(n_states + (n_actions,))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = espilon

    def choose_action(self, obs):
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, 2)
        else:
            action = np.argmax(self.qTable[tuple(obs)])
        return action

    def update(self, obs, action, reward, next_obs):
        td_error = (
            reward
            + self.gamma * np.max(self.qTable[tuple(next_obs)])
            - self.qTable[tuple(obs) + (action,)]
        )
        self.qTable[tuple(obs) + (action,)] += self.alpha * td_error

    def delay_espilon(self):
        self.epsilon = max(0.01, self.epsilon * 0.995)


env = TwoPhaseIntersection()
agent = QLearning()

for ep in range(5001):
    obs, _ = env.reset()
    total = 0
    for _ in range(100):
        action = agent.choose_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        agent.update(obs, action, reward, next_obs)
        obs = next_obs
        total += reward
        if terminated or truncated:
            break

    agent.delay_espilon()

    if ep % 50 == 0:
        print(f"ep{ep} total={total:.0f} eps={agent.epsilon:.2f}")
