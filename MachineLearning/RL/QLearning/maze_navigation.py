import numpy as np
import time
import matplotlib.pyplot as plt
import platform
import subprocess

# 字体
plt.rcParams["font.sans-serif"] = [
    "Source Han Sans SC",
    "Noto Sans CJK SC",
    "SimHei",
]
plt.rcParams["axes.unicode_minus"] = False


def load_maze(path: str):
    """
    S: 入口 #: 墙 . : 路 G: 出口
    """
    with open(path, "r") as f:
        lines = [
            line.rstrip(
                "\n",
            )
            for line in f
        ]
    return lines


def clear_screen():
    cur_os = platform.system()
    if cur_os == "Windows":
        subprocess.run("cls")
    else:
        subprocess.run("clear")


class maze_navigation:
    """
    - 到达出口：+100
    - 撞墙：-10
    - 普通移动：-1（鼓励效率）。
    """

    def __init__(self):
        self.maze = load_maze("./maze.txt")
        self.rows = len(self.maze)  # 行
        self.cols = len(self.maze[0])  # 列
        self.states = self.rows * self.cols
        self.action_delta = {
            0: (0, -1),  # 上
            1: (0, 1),  # 下
            2: (-1, 0),  # 左
            3: (1, 0),  # 右
        }
        self.actions = len(self.action_delta)
        self.state = 0
        self.exit_pos = (7, 7)

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        row = self.state // self.rows
        col = self.state % self.cols
        dr, dc = self.action_delta[action][1], self.action_delta[action][0]
        new_row = row + dr
        new_col = col + dc

        if 0 <= new_col < self.cols and 0 <= new_row < self.rows:
            dot = self.maze[new_row][new_col]
            next_state = (new_row * self.cols + new_col) if dot != "#" else self.state
            if dot == "#":
                reward = -10.0
                done = False
            elif dot == "G":
                reward = 100.0
                done = True
            else:
                reward = -1.0
                done = False
        else:
            reward = -10.0
            done = False
            next_state = self.state

        self.state = next_state
        return next_state, reward, done

    @property
    def get_pos(self):
        row = self.state // self.rows
        col = self.state % self.cols
        return row, col

    def map(self):
        clear_screen()
        print("=====")
        for i in range(self.rows):
            for j in range(self.cols):
                print(
                    self.maze[i][j] if (i, j) != self.get_pos else "o",
                    end=" ",
                    sep="",
                )
            print()


class QLearning:
    def __init__(
        self,
        states: int,  # 状态个数
        actions: int,  # 动作数
        learning_rate: float,  # 学习率
        gamma: float,  # 折扣因子
        espilon: float,  # 初始探索概率
    ):
        self.q_table = np.zeros([states, actions])
        self.states = states
        self.actions = actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.espilon = espilon

    def take_action(self, state):
        if np.random.random() < self.espilon:
            action = np.random.randint(self.actions)
        else:
            action = np.argmax(self.q_table[state])
        return action

    def update(self, s0, a0, s1, r):
        td_error = r + self.gamma * self.q_table[s1].max() - self.q_table[s0, a0]
        self.q_table[s0, a0] += self.learning_rate * td_error

    def manhattan_distance_q(self, goal_row, goal_col, rows, cols):
        """用曼哈顿距离的倒数初始化 Q 表，离出口越近初值越高"""
        for i in range(rows):
            for j in range(cols):
                distance = abs(goal_row - i) + abs(goal_col - j) + 1
                for k in range(self.actions):
                    self.q_table[i * cols + j, k] = 1 / distance


def plot_q_heatmap(
    q_table: np.ndarray,
    rows: int,
    cols: int,
    maze: list[str],
    title: str = "各动作 Q 值热力图（迷宫）",
    save_path: str = "q_per_action.png",
) -> None:
    """绘制四个方向的分动作 Q 值热力图，墙用深色块特别标识"""
    action_names = ["↑ 上", "↓ 下", "← 左", "→ 右"]
    # maze_navigation 使用 row-major 索引: state = row * cols + col
    q_maps = [q_table[:, a].reshape(rows, cols) for a in range(4)]

    # 找出所有墙的位置
    walls = [(r, c) for r in range(rows) for c in range(cols) if maze[r][c] == "#"]

    # 仅用非墙状态的 Q 值计算色阶范围，避免墙的 0 值拉偏色阶
    wall_states = [r * cols + c for r, c in walls]
    non_wall_mask = np.ones(q_table.shape[0], dtype=bool)
    non_wall_mask[wall_states] = False
    valid_q = q_table[non_wall_mask]
    vmin = valid_q.min() if valid_q.size > 0 else q_table.min()
    vmax = valid_q.max() if valid_q.size > 0 else q_table.max()

    # 构建墙的布尔掩码（True = 墙，在 imshow 中会被遮盖）
    wall_mask = np.zeros((rows, cols), dtype=bool)
    for r, c in walls:
        wall_mask[r, c] = True

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for a, (ax, name, q_map) in enumerate(zip(axes.flat, action_names, q_maps)):
        masked_q = np.ma.array(q_map, mask=wall_mask)
        im = ax.imshow(masked_q, cmap="RdYlGn", origin="upper", vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=13)
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))

        for i in range(rows):
            for j in range(cols):
                if wall_mask[i, j]:
                    # 墙：深灰色填充块 + 粗体 # 标识
                    ax.add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5),
                            1,
                            1,
                            fill=True,
                            facecolor="#333333",
                            edgecolor="black",
                            linewidth=0.5,
                        )
                    )
                    ax.text(
                        j,
                        i,
                        "#",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="white",
                        fontweight="bold",
                    )
                else:
                    ax.text(
                        j,
                        i,
                        f"{q_map[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if q_map[i, j] < (vmin + vmax) / 2 else "black",
                    )

    fig.colorbar(im, ax=axes, shrink=0.8, label="Q 值")
    fig.suptitle(title, fontsize=15)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_max_q_heatmap(
    q_table: np.ndarray,
    rows: int,
    cols: int,
    maze: list[str],
    title: str = "各位置最大 Q 值（迷宫）",
    save_path: str = "max_q_heatmap.png",
) -> None:
    """绘制每个位置最大 Q 值（最优动作价值）的热力图，墙用深色块特别标识"""
    # maze_navigation 使用 row-major 索引: state = row * cols + col
    max_q = q_table.max(axis=1).reshape(rows, cols)

    # 找出所有墙的位置
    walls = [(r, c) for r in range(rows) for c in range(cols) if maze[r][c] == "#"]
    wall_mask = np.zeros((rows, cols), dtype=bool)
    for r, c in walls:
        wall_mask[r, c] = True

    # 仅用非墙状态的 Q 值计算色阶范围
    valid_values = max_q.flatten()[~wall_mask.flatten()]
    vmin = valid_values.min() if valid_values.size > 0 else max_q.min()
    vmax = valid_values.max() if valid_values.size > 0 else max_q.max()

    masked_q = np.ma.array(max_q, mask=wall_mask)

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(masked_q, cmap="RdYlGn", origin="upper", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=15)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))

    for i in range(rows):
        for j in range(cols):
            if wall_mask[i, j]:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=True,
                        facecolor="#333333",
                        edgecolor="black",
                        linewidth=0.5,
                    )
                )
                ax.text(
                    j,
                    i,
                    "#",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="white",
                    fontweight="bold",
                )
            else:
                val = max_q[i, j]
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if val < (vmin + vmax) / 2 else "black",
                )

    fig.colorbar(im, ax=ax, shrink=0.8, label="max Q 值")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_curve(
    series: dict[str, list[int]],
    window: int = 100,
    title: str = "Q-Learning 训练过程（迷宫问题）",
    save_path: str = "training_curve.png",
    max_episodes: int | None = None,
) -> None:
    """绘制每轮步数随训练的变化曲线，支持多条曲线对比

    Parameters
    ----------
    series : dict[str, list[int]]
        标签 → 步数列表，每条曲线一个键值对
    window : int
        移动平均窗口大小
    title : str
        图表标题
    save_path : str
        保存路径
    max_episodes : int | None
        仅绘制前 N 轮；为 None 则绘制全部
    """
    plt.figure(figsize=(12, 5))
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
    for idx, (label, steps) in enumerate(series.items()):
        episodes = steps[:max_episodes] if max_episodes else steps
        color = colors[idx % len(colors)]
        plt.plot(
            range(1, len(episodes) + 1),
            episodes,
            linewidth=0.8,
            alpha=0.4,
            color=color,
        )
        if len(episodes) >= window:
            ma = np.convolve(episodes, np.ones(window) / window, mode="valid")
            x_ma = range(window, len(episodes) + 1)
            plt.plot(
                x_ma,
                ma,
                color=color,
                linewidth=2,
                label=f"{label}（{window}轮MA）",
            )
        else:
            # 数据不足 window 时仍显示图例
            plt.plot([], [], color=color, linewidth=2, label=label)
    plt.xlabel("训练轮数")
    plt.ylabel("步数")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def train_one_run(env: maze_navigation, agent: QLearning, episodes: int, final_exec:bool) -> list[int]:
    """训练单个 agent，返回每轮的步数列表"""
    espilon_decay = 0.9
    espilon_end = 0.1
    step_list = []
    for i in range(episodes):
        state = env.reset()
        done = False
        step = 0
        while not done:
            action = agent.take_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, next_state, reward)
            state = next_state
            step += 1
            if (final_exec and i > episodes - 2):
                time.sleep(0.1)
                env.map()
        step_list.append(step)
        agent.espilon = max(espilon_end, agent.espilon * espilon_decay)
    return step_list


np.random.seed(42)

env = maze_navigation()

# 无初始化（全零 Q 表）
agent_zero = QLearning(64, 4, 0.5, 0.9, 0.5)
steps_zero = train_one_run(env, agent_zero, 2000, False)

# 曼哈顿距离初始化
env2 = maze_navigation()
agent_manhattan = QLearning(64, 4, 0.5, 0.9, 0.5)
agent_manhattan.manhattan_distance_q(*env2.exit_pos, env2.rows, env2.cols)
steps_manhattan = train_one_run(env2, agent_manhattan, 2000, True)

# 对比前 20 轮收敛曲线
plot_training_curve(
    {"无初始化（全零）": steps_zero, "曼哈顿距离初始化": steps_manhattan},
    window=5,
    title="有无曼哈顿距离初始化的收敛性对比（前 20 轮）",
    save_path="training_curve_comparison.png",
    max_episodes=20,
)

# 全 2000 轮对比
plot_training_curve(
    {"无初始化（全零）": steps_zero, "曼哈顿距离初始化": steps_manhattan},
    window=100,
    title="有无曼哈顿距离初始化的收敛性对比（全 2000 轮）",
    save_path="training_curve_comparison_full.png",
)

# 无初始化 — max Q 热力图
plot_max_q_heatmap(
    agent_zero.q_table, env.rows, env.cols, env.maze,
    title="最大 Q 值 — 无初始化",
    save_path="max_q_zero.png",
)

# 曼哈顿初始化 — max Q 热力图
plot_max_q_heatmap(
    agent_manhattan.q_table, env2.rows, env2.cols, env2.maze,
    title="最大 Q 值 — 曼哈顿距离初始化",
    save_path="max_q_manhattan.png",
)

# 无初始化 — 分动作 Q 热力图
plot_q_heatmap(
    agent_zero.q_table, env.rows, env.cols, env.maze,
    title="各动作 Q 值 — 无初始化",
    save_path="q_per_action_zero.png",
)

# 曼哈顿初始化 — 分动作 Q 热力图
plot_q_heatmap(
    agent_manhattan.q_table, env2.rows, env2.cols, env2.maze,
    title="各动作 Q 值 — 曼哈顿距离初始化",
    save_path="q_per_action_manhattan.png",
)
