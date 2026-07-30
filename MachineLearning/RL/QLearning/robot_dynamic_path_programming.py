import numpy as np
import time
import platform
import subprocess
import matplotlib.pyplot as plt
from mealpy.swarm_based.SSA import OriginalSSA

# 字体
plt.rcParams["font.sans-serif"] = [
    "Source Han Sans SC",
    "Noto Sans CJK SC",
    "SimHei",
]
plt.rcParams["axes.unicode_minus"] = False

map = """
S...#....
.#.......
.........
...#..#..
...#....G
"""


def clear_screen():
    cur_os = platform.system()
    if cur_os == "Windows":
        subprocess.run("cls")
    else:
        subprocess.run("clear")


class robot_dynamic_path_programming_v1:
    def __init__(self, map, begin_pos, end_pos, dynamic_row):
        self.maze = self.load_maze(map)
        self.rows = len(self.maze)
        self.cols = len(self.maze[0])
        self.begin_pos = begin_pos  # (x, y)
        self.end_pos = end_pos  # (x, y)
        self.dynamic_row = dynamic_row  # y
        self.states = self.rows * self.cols
        self.action_delta = {
            0: (0, -1),  # 上
            1: (0, 1),  # 下
            2: (-1, 0),  # 左
            3: (1, 0),  # 右
            4: (0, 0),  # 不动
        }
        self.obstacle_pos = [0, self.dynamic_row]  # (x, y)
        self.obstacle_direction = 1  # 1 表示向右，-1 表示向左
        self.state = self.pos_to_state(*self.begin_pos)

    def load_maze(self, map: str):
        lines = map.strip().splitlines()
        return [list(line) for line in lines]

    def reset(self):
        self.state = self.pos_to_state(*self.begin_pos)
        return self.state

    def step(self, action):
        x, y = self.get_pos
        dx, dy = self.action_delta[action]
        new_x, new_y = x + dx, y + dy

        # 动态障碍物横向来回移动：到达边界后反向
        obstacle_x, obstacle_y = self.obstacle_pos
        self.maze[obstacle_y][obstacle_x] = "."

        next_obstacle_x = obstacle_x + self.obstacle_direction
        if next_obstacle_x < 0 or next_obstacle_x >= self.cols:
            self.obstacle_direction *= -1
            next_obstacle_x = obstacle_x + self.obstacle_direction

        self.obstacle_pos[0] = next_obstacle_x
        obstacle_x, obstacle_y = self.obstacle_pos
        self.maze[obstacle_y][obstacle_x] = "#"

        if 0 <= new_x < self.cols and 0 <= new_y < self.rows:
            match self.maze[new_y][new_x]:
                case "." | "S":
                    reward = -1.0  # TODO: 这里应该要用曼哈顿距离的倒数来加快收敛
                    done = False
                    next_state = self.pos_to_state(new_x, new_y)
                case "#":  # 撞到障碍物
                    reward = -10.0
                    done = True
                    next_state = self.pos_to_state(new_x, new_y)
                case "G":
                    reward = 50.0
                    done = True
                    next_state = self.pos_to_state(new_x, new_y)
        else:  # 撞墙
            reward = -10
            done = False
            next_state = self.state
        self.state = next_state
        return next_state, reward, done

    @property
    def get_pos(self):
        y = self.state // self.cols
        x = self.state % self.cols
        return x, y

    def pos_to_state(self, x, y):
        return y * self.cols + x

    def print_map(self):
        clear_screen()
        x, y = self.get_pos
        for row in range(self.rows):
            for col in range(self.cols):
                print("o" if (col, row) == (x, y) else self.maze[row][col], end="")
            print()


class QLearning:
    def __init__(self, states, actions, learning_rate, gamma, espilon):
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


def plot_training_curve(episode_steps: list[int], window: int = 100) -> None:
    """绘制每轮步数随训练的变化曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(episode_steps, linewidth=0.5, alpha=0.7)
    if len(episode_steps) >= window:
        ma = np.convolve(episode_steps, np.ones(window) / window, mode="valid")
        plt.plot(ma, color="red", linewidth=2, label=f"{window}轮移动平均")
    plt.xlabel("训练轮数")
    plt.ylabel("步数")
    plt.title("Q-Learning 训练过程（动态障碍物）")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("training_curve.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_success_bar(successes: list[bool], group_size: int = 10) -> None:
    """绘制每 group_size 轮的成功次数柱状图"""
    success_counts = [
        sum(successes[i : i + group_size]) for i in range(0, len(successes), group_size)
    ]
    episode_ranges = [
        f"{i + 1}-{min(i + group_size, len(successes))}"
        for i in range(0, len(successes), group_size)
    ]

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(success_counts)), success_counts, color="#4C78A8")
    plt.xlabel("训练轮区间")
    plt.ylabel("成功次数")
    plt.title(f"每 {group_size} 轮成功次数")
    plt.ylim(0, group_size)

    tick_step = max(1, len(episode_ranges) // 20)
    tick_positions = list(range(0, len(episode_ranges), tick_step))
    plt.xticks(
        tick_positions,
        [episode_ranges[i] for i in tick_positions],
        rotation=45,
        ha="right",
    )

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("success_count_bar.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_q_heatmap(q_table: np.ndarray, env: robot_dynamic_path_programming_v1) -> None:
    """绘制每个动作的 Q 值热力图"""
    action_names = ["↑ 上", "↓ 下", "← 左", "→ 右", "· 不动"]
    q_maps = [
        q_table[:, a].reshape(env.rows, env.cols) for a in range(len(action_names))
    ]
    vmin = min(m.min() for m in q_maps)
    vmax = max(m.max() for m in q_maps)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, name, q_map in zip(axes.flat, action_names, q_maps):
        im = ax.imshow(q_map, cmap="RdYlGn", origin="upper", vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=13)
        ax.set_xticks(range(env.cols))
        ax.set_yticks(range(env.rows))
        ax.axhline(
            env.dynamic_row - 0.5,
            color="#1f77b4",
            linestyle="--",
            linewidth=1.2,
        )
        ax.axhline(
            env.dynamic_row + 0.5,
            color="#1f77b4",
            linestyle="--",
            linewidth=1.2,
        )

        for row in range(env.rows):
            for col in range(env.cols):
                ax.text(
                    col,
                    row,
                    f"{q_map[row, col]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if q_map[row, col] < (vmin + vmax) / 2 else "black",
                )

                cell = env.maze[row][col]
                if cell == "#":
                    ax.text(
                        col,
                        row,
                        "#",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color="black",
                        fontweight="bold",
                    )
                elif cell in {"S", "G"}:
                    ax.text(
                        col,
                        row,
                        cell,
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color="black",
                        fontweight="bold",
                    )

    axes.flat[-1].axis("off")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Q 值")
    fig.suptitle("各动作 Q 值热力图（动态障碍物）", fontsize=15)
    plt.savefig("q_per_action.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_max_q_heatmap(
    q_table: np.ndarray, env: robot_dynamic_path_programming_v1
) -> None:
    """绘制每个位置的最大 Q 值热力图"""
    max_q = q_table.max(axis=1).reshape(env.rows, env.cols)
    vmin = max_q.min()
    vmax = max_q.max()

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    im = ax.imshow(max_q, cmap="RdYlGn", origin="upper", vmin=vmin, vmax=vmax)
    ax.set_title("各位置最大 Q 值热力图（动态障碍物）", fontsize=15)
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.axhline(
        env.dynamic_row - 0.5,
        color="#1f77b4",
        linestyle="--",
        linewidth=1.2,
    )
    ax.axhline(
        env.dynamic_row + 0.5,
        color="#1f77b4",
        linestyle="--",
        linewidth=1.2,
    )

    for row in range(env.rows):
        for col in range(env.cols):
            ax.text(
                col,
                row,
                f"{max_q[row, col]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if max_q[row, col] < (vmin + vmax) / 2 else "black",
            )

            cell = env.maze[row][col]
            if cell == "#" and row == env.dynamic_row:
                ax.text(
                    col,
                    row,
                    "动",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#1f77b4",
                    fontweight="bold",
                )
            elif cell == "#":
                ax.text(
                    col,
                    row,
                    "#",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="black",
                    fontweight="bold",
                )
            elif cell in {"S", "G"}:
                ax.text(
                    col,
                    row,
                    cell,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="black",
                    fontweight="bold",
                )

    fig.colorbar(im, ax=ax, shrink=0.8, label="max Q 值")
    plt.savefig("max_q_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()


env = robot_dynamic_path_programming_v1(map, (0, 0), (8, 4), 2)
agent = QLearning(env.rows * env.cols, len(env.action_delta), 0.5, 0.9, 0.5)
espilon_delay = 0.9
espilon_min = 0.1
steps_list = []
success_list = []
for i in range(2000):
    step = 0
    done = False
    success = False
    state = env.reset()
    while not done:
        action = agent.take_action(state)
        next_state, reward, done = env.step(action)
        if done and next_state == env.pos_to_state(*env.end_pos):
            success = True
        agent.update(state, action, next_state, reward)
        state = next_state
        step += 1
        if i > 1998:
            time.sleep(0.1)
            env.print_map()
    agent.espilon = max(espilon_min, agent.espilon * espilon_delay)
    steps_list.append(step)
    success_list.append(success)


plot_training_curve(steps_list)
plot_success_bar(success_list)
plot_q_heatmap(agent.q_table, env)
plot_max_q_heatmap(agent.q_table, env)
