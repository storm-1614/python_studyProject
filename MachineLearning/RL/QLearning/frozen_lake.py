import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

# 字体
plt.rcParams["font.sans-serif"] = [
    "Source Han Sans SC",
    "Noto Sans CJK SC",
    "SimHei",
]
plt.rcParams["axes.unicode_minus"] = False


class forzen_lake:
    def __init__(self, obstacle: list[tuple[int, int]]) -> None:
        """
        'S', 'F', 'F', 'F',
        'F', 'F', 'F', 'H',
        'F', 'F', 'F', 'H',
        'H', 'F', 'F', 'G'
        S 起点 G 终点 H 洞 F 冰面
        """
        self.desc = [
            [
                "S",
                "F",
                "F",
                "F",
            ],
            [
                "F",
                "F",
                "F",
                "F",
            ],
            [
                "F",
                "F",
                "F",
                "F",
            ],
            [
                "F",
                "F",
                "F",
                "G",
            ],
        ]

        for obs in obstacle:
            self.desc[obs[0]][obs[1]] = "H"

        self.action_delta = {
            0: (-1, 0),  # 左
            1: (1, 0),  # 右
            2: (0, -1),  # 上
            3: (0, 1),  # 下
        }
        self.rows = 4
        self.cols = 4
        self.states = self.rows * self.cols
        self.state = 0
        self.finish = 0

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        col = self.state % self.cols
        row = self.state // self.rows
        dc, dr = self.action_delta[action]  # (dx, dy): 第一个值=x(col), 第二个值=y(row)
        new_col, new_row = col + dc, row + dr
        done = False
        reward = 0

        if 0 <= new_col < self.cols and 0 <= new_row < self.rows:
            match self.desc[new_row][new_col]:
                case "F" | "S":
                    reward = 0.0
                    done = False
                case "G":
                    reward = 1.0
                    done = True
                    self.finish += 1
                    # self.map() 
                case "H":
                    reward = -0.5
                    done = True
            next_state = new_col * self.rows + new_row
        else:
            next_state = self.state
            reward = 0
            done = False
        self.state = next_state
        return next_state, reward, done

    @property
    def get_pos(self):
        col = self.state % self.cols
        row = self.state // self.rows
        return col, row

    def map(self):
        for i in range(self.cols):
            for j in range(self.rows):
                print(
                    f"{self.desc[i][j]}{'o' if (i, j) == self.get_pos else ' '}",
                    end=" ",
                )
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
        self.q_table: NDArray = np.zeros(
            [states, actions]
        )  # 零初始化，与真实回报（约-1.5~1.0）量级匹配
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

    def update(self, s0, a0, s1, r, done):
        if done:
            td_target = r  # 终止状态：未来价值为 0
        else:
            td_target = r + self.gamma * self.q_table[s1].max()
        td_error = td_target - self.q_table[s0, a0]
        self.q_table[s0, a0] += self.learning_tate * td_error


def plot_training_curve(episode_steps: list[int], window: int = 100) -> None:
    """绘制每轮步数随训练的变化曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(episode_steps, linewidth=0.5, alpha=0.7)
    if len(episode_steps) >= window:
        ma = np.convolve(episode_steps, np.ones(window) / window, mode="valid")
        plt.plot(ma, color="red", linewidth=2, label=f"{window}轮移动平均")
    plt.xlabel("训练轮数")
    plt.ylabel("步数")
    plt.title("Q-Learning 训练过程（冰湖问题）")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("training_curve.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_success_rate(episode_success: list[int], window: int = 10) -> None:
    """绘制每 window 轮的成功次数柱状图"""
    chunk_success = [
        sum(episode_success[i : i + window])
        for i in range(0, len(episode_success), window)
    ]
    episode_ticks = [i + 1 for i in range(0, len(episode_success), window)]

    plt.figure(figsize=(12, 5))
    plt.bar(
        episode_ticks,
        chunk_success,
        width=window * 0.8,
        align="edge",
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
        linewidth=0.3,
    )
    plt.axhline(y=window, color="green", linestyle="--", label=f"满分线 ({window})")
    plt.xlabel("训练轮数")
    plt.ylabel(f"每 {window} 轮成功次数")
    plt.title(f"Q-Learning — 每 {window} 轮成功次数（冰湖问题）")
    plt.ylim(0, window + 0.5)
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.savefig("success_rate.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_q_heatmap(
    q_table: NDArray,
    rows: int,
    cols: int,
    obstacle: list[tuple[int, int]],
) -> None:
    """绘制四个方向的分动作 Q 值热力图"""
    action_names = ["← 左", "→ 右", "↑ 上", "↓ 下"]
    q_maps = [q_table[:, a].reshape(rows, cols, order="F") for a in range(4)]
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
                ax.text(
                    j,
                    i,
                    f"{q_map[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if q_map[i, j] < (vmin + vmax) / 2 else "black",
                )
        for r, c in obstacle:
            ax.text(
                c,
                r,
                "X",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                fontweight="bold",
            )

    fig.colorbar(im, ax=axes, shrink=0.8, label="Q 值")
    fig.suptitle("各动作 Q 值热力图", fontsize=15)
    plt.savefig("q_per_action.png", dpi=150, bbox_inches="tight")
    plt.show()


obstacle = [(1, 3), (2, 3), (3, 0)]
env = forzen_lake(obstacle)
agent = QLearning(16, 4, 0.5, 0.9, 1.0)
epsilon_decay = 0.997
epsilon_min = 0.01

episode_steps = []  # 记录每轮的步数
episode_success = []  # 记录每轮是否成功 (1=到达G, 0=掉洞/超步数)
success_window = 10  # 每多少轮统计一次成功次数

for i in range(2000):
    state = env.reset()
    done = False
    step = 0
    while not done:
        action = agent.take_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, next_state, reward, done)
        state = next_state
        step += 1
    episode_steps.append(step)
    episode_success.append(1 if reward == 1.0 else 0)
    agent.espilon = max(epsilon_min, agent.espilon * epsilon_decay)
    if (i + 1) % success_window == 0:
        recent_success = sum(episode_success[-success_window:])
        print(
            f"Episode {i + 1 - success_window + 1}~{i + 1}: "
            f"成功 {recent_success}/{success_window} 次, "
            f"最近步数 {step} 步, "
            f"epsilon={agent.espilon:.3f}"
        )
    elif i % 200 == 0:
        print(f"Episode {i}: {step} steps, epsilon={agent.espilon:.3f}")

print(f"总成功次数: {env.finish} / 2000")

# ==================== 训练后演示 ====================
print("\n" + "=" * 40)
print("训练完毕，演示：")
print("=" * 40)
agent.espilon = 0.0  # 纯利用，不探索
state = env.reset()
done = False
step = 0
env.map()
while not done:
    action = agent.take_action(state)
    next_state, reward, done = env.step(action)
    print(
        f"\nStep {step + 1}: 动作={['←左', '→右', '↑上', '↓下'][action]}, "
        f"奖励={reward}, 状态 {state} → {next_state}"
    )
    env.map()
    state = next_state
    step += 1
    if step > 20:  # 安全上限
        print("超过20步，终止")
        break
print(f"\n结果: {'到达终点！' if reward == 1.0 else '失败'} 共 {step} 步")

plot_training_curve(episode_steps)
plot_success_rate(episode_success)
plot_q_heatmap(agent.q_table, 4, 4, obstacle)
