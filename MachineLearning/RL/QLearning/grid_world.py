import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

table: NDArray = np.zeros([8, 8], dtype=float)
state: int = 8 * 8 + 8
row: int = 8
col: int = 8


class grid_world:
    def __init__(self, rows: int, cols: int, obstacle: list[tuple[int, int]]) -> None:
        """
        4 个 action: 0 ~ 3 代表上下左右
        """
        self.action_delta = {
            0: (-1, 0),  # 上: row-1
            1: (1, 0),   # 下: row+1
            2: (0, -1),  # 左: col-1
            3: (0, 1),   # 右: col+1
        }
        self.rows: int = rows
        self.cols: int = cols
        self.state: int = 0
        self.end_point: tuple[int, int] = (rows - 1, cols - 1)
        self.obstacle: list[tuple[int, int]] = obstacle

    def reset(self) -> int:
        self.state = 0
        return self.state

    def step(self, action):
        """
        返回 next_state, reward, done
        """
        row: int = self.state // self.rows
        col: int = self.state % self.cols
        dr, dc = self.action_delta[action]
        new_row, new_col = row + dr, col + dc

        if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
            next_state = new_row * self.cols + new_col
        else:
            next_state = self.state

        if (new_row, new_col) == self.end_point:
            reward = 1.0
            done = True
        elif (new_row, new_col) in self.obstacle:
            reward = -1.0
            done = True
        else:
            dirt = abs(new_row - self.end_point[0]) + abs(new_col - self.end_point[1])
            reward = -0.01 - 0.001 * dirt
            done = False

        self.state = next_state
        return next_state, reward, done

    @property
    def get_pos(self):
        row = self.state // self.rows
        col = self.state % self.cols
        return row, col

    def map(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if (i, j) != self.get_pos:
                    print(" *", end="")
                else:
                    print(" o", end="")
            print()


class QLearning:
    def __init__(
        self,
        states: int,  # 状态
        actions: int,  # 动作
        learning_rate: float,  # 学习率
        gamma: float,  # 折扣因子
        espilon: float,  # 初始探索概率
    ) -> None:
        """ """
        self.q_table: NDArray = np.zeros([states, actions])
        self.states: int = states
        self.actions: int = actions
        self.learning_tate: float = learning_rate
        self.gamma: float = gamma
        self.espilon: float = espilon

    def take_action(self, state):
        if np.random.random() < self.espilon:
            action = np.random.randint(self.actions)
        else:
            action = np.argmax(self.q_table[state])
        return action

    def update(self, s0, a0, s1, r):
        td_error = r + self.gamma * self.q_table[s1].max() - self.q_table[s0, a0]
        self.q_table[s0, a0] += self.learning_tate * td_error

    def save(self):
        return self.q_table

    def load(self, q: NDArray):
        self.q_table = q


obstacle = [(2, 2), (4, 6), (1, 6)]
env = grid_world(8, 8, obstacle)
agent = QLearning(64, 4, 0.5, 0.9, 0.5)
step = 0
episode_steps = []  # 记录每轮的步数

for i in range(2000):
    state = env.reset()
    done = False
    step = 0
    while not done:
        action = agent.take_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, next_state, reward)
        state = next_state
        step += 1
    episode_steps.append(step)
    if i % 200 == 0:
        print(f"Episode {i}: {step} steps")

print(f"最终步数: {step}")


def plot_training_curve(episode_steps: list[int], window: int = 100) -> None:
    """绘制每轮步数随训练的变化曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(episode_steps, linewidth=0.5, alpha=0.7)
    if len(episode_steps) >= window:
        ma = np.convolve(episode_steps, np.ones(window) / window, mode="valid")
        plt.plot(ma, color="red", linewidth=2,
                 label=f"{window}-episode moving average")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Q-Learning Training Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("training_curve.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_q_heatmap(
    q_table: NDArray,
    rows: int,
    cols: int,
    obstacle: list[tuple[int, int]],
) -> None:
    """绘制四个方向的分动作 Q 值热力图"""
    action_names = ["↑ Up", "↓ Down", "← Left", "→ Right"]
    q_maps = [q_table[:, a].reshape(rows, cols) for a in range(4)]
    vmin = min(m.min() for m in q_maps)
    vmax = max(m.max() for m in q_maps)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for a, (ax, name, q_map) in enumerate(zip(axes.flat, action_names, q_maps)):
        im = ax.imshow(q_map, cmap="RdYlGn", origin="upper", vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=13)
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        for i in range(rows):
            for j in range(cols):
                ax.text(j, i, f"{q_map[i, j]:.2f}", ha="center", va="center",
                        fontsize=6,
                        color="white" if q_map[i, j] < (vmin + vmax) / 2 else "black")
        for (r, c) in obstacle:
            ax.text(c, r, "X", ha="center", va="center", fontsize=8,
                    color="black", fontweight="bold")

    fig.colorbar(im, ax=axes, shrink=0.8, label="Q-value")
    fig.suptitle("Q-Values per Action", fontsize=15)
    plt.savefig("q_per_action.png", dpi=150, bbox_inches="tight")
    plt.show()


plot_training_curve(episode_steps)
plot_q_heatmap(agent.q_table, 8, 8, obstacle)
