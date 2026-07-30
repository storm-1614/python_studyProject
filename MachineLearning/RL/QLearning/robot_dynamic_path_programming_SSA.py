import numpy as np
import time
import platform
import subprocess
import matplotlib.pyplot as plt
from mealpy import FloatVar
from mealpy.swarm_based.SSA import OriginalSSA

# 字体
plt.rcParams["font.sans-serif"] = [
    "Source Han Sans SC",
    "Noto Sans CJK SC",
    "SimHei",
]
plt.rcParams["axes.unicode_minus"] = False

map = """
S...#...#
.#....#..
.........
...#.....
.....#...
.#.....#.
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
            reward = -1.0
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


def run_episode(
    env: robot_dynamic_path_programming_v1,
    agent: QLearning,
    train: bool = True,
    render: bool = False,
    max_steps: int = 200,
) -> tuple[int, bool, float]:
    """运行一轮路径规划，返回步数、是否到达终点、累计奖励"""
    step = 0
    done = False
    success = False
    total_reward = 0.0
    state = env.reset()

    while not done and step < max_steps:
        action = agent.take_action(state)
        next_state, reward, done = env.step(action)
        if done and next_state == env.pos_to_state(*env.end_pos):
            success = True
        if train:
            agent.update(state, action, next_state, reward)
        state = next_state
        total_reward += reward
        step += 1

        if render:
            time.sleep(0.1)
            env.print_map()

    return step, success, total_reward


def plot_training_curve(
    episode_steps: list[int],
    param_history: list[tuple[int, float, float, float]] | None = None,
    window: int = 100,
) -> None:
    """绘制每轮步数随训练的变化曲线，叠加 SSA 调参时刻的 learning_rate / gamma"""
    fig, ax1 = plt.subplots(figsize=(14, 5))

    ax1.plot(episode_steps, linewidth=0.5, alpha=0.7, label="每轮步数")
    if len(episode_steps) >= window:
        ma = np.convolve(episode_steps, np.ones(window) / window, mode="valid")
        ax1.plot(ma, color="red", linewidth=2, label=f"{window}轮移动平均")
    ax1.set_xlabel("训练轮数")
    ax1.set_ylabel("步数", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, alpha=0.3)

    if param_history:
        episodes, lrs, gammas, _scores = zip(*param_history)
        # 绘制 lr 阶梯线
        all_episodes = [0]
        all_lrs = [lrs[0]]
        all_gammas = [gammas[0]]
        for i, ep in enumerate(episodes):
            all_episodes.append(ep)
            all_lrs.append(lrs[i])
            all_gammas.append(gammas[i])
            if i + 1 < len(episodes):
                all_episodes.append(episodes[i + 1])
                all_lrs.append(lrs[i])
                all_gammas.append(gammas[i])
            else:
                all_episodes.append(len(episode_steps))
                all_lrs.append(lrs[i])
                all_gammas.append(gammas[i])

        ax2 = ax1.twinx()
        ax2.step(all_episodes, all_lrs, where="post", color="#ff7f0e",
                  linewidth=2, label="Learning Rate")
        ax2.step(all_episodes, all_gammas, where="post", color="#2ca02c",
                  linewidth=2, label="Gamma", linestyle="--")
        ax2.set_ylabel("超参数值", color="#333")
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(axis="y")

    lines1, labels1 = ax1.get_legend_handles_labels()
    if param_history:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        ax1.legend(loc="upper right")

    plt.title("Q-Learning 训练过程（动态障碍物 + SSA 调参）")
    fig.tight_layout()
    plt.savefig("training_curve.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_param_evolution(
    param_history: list[tuple[int, float, float, float]],
) -> None:
    """绘制 SSA 调参过程中 learning_rate, gamma 和 fitness score 的变化"""
    if not param_history:
        print("param_history 为空，跳过参数演化图")
        return

    episodes, lrs, gammas, scores = zip(*param_history)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, constrained_layout=True
    )

    # 上图：learning_rate 和 gamma 阶梯线
    ax1.step(episodes, lrs, where="post", color="#ff7f0e", linewidth=2,
             label="Learning Rate")
    ax1.step(episodes, gammas, where="post", color="#2ca02c", linewidth=2,
             linestyle="--", label="Gamma")
    ax1.set_ylabel("超参数值")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("SSA 调参 — Learning Rate 与 Gamma 变化")

    # 下图：SSA 目标函数得分（越小越好）
    ax2.step(episodes, scores, where="post", color="#d62728", linewidth=2,
             label="SSA Fitness Score")
    ax2.set_xlabel("训练轮数")
    ax2.set_ylabel("Fitness Score")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("SSA 每次搜索的最优得分")

    plt.savefig("param_evolution.png", dpi=150, bbox_inches="tight")
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


def find_pos(maze: list[list[str]], target: str) -> tuple[int, int]:
    """在地图中查找目标字符的位置 (x, y)"""
    for y, row in enumerate(maze):
        for x, cell in enumerate(row):
            if cell == target:
                return x, y
    raise ValueError(f"地图中未找到 '{target}'")


# 从地图自动解析起点/终点
maze_data = [list(line) for line in map.strip().splitlines()]
begin_pos = find_pos(maze_data, "S")
end_pos = find_pos(maze_data, "G")

env = robot_dynamic_path_programming_v1(map, begin_pos, end_pos, 2)
main_agent = QLearning(
    env.rows * env.cols,
    len(env.action_delta),
    learning_rate=0.5,
    gamma=0.9,
    espilon=0.5,
)

# SSA 参数
adjust_interval = 200  # 每 200 轮调整一次
ssa_epochs = 5
ssa_pop = 10
eval_episodes_for_ssa = 30
max_steps_per_episode = 200

# lr, gamma 上下界
ub = [0.9, 0.999]
lb = [0.01, 0.8]


def search_best_params_by_ssa(agent: QLearning) -> tuple[float, float, float]:
    """使用 SSA 搜索当前阶段更合适的 learning_rate 和 gamma"""

    def objective(solution):
        learning_rate, gamma = solution
        eval_env = robot_dynamic_path_programming_v1(
            map, env.begin_pos, env.end_pos, env.dynamic_row
        )
        eval_agent = QLearning(
            eval_env.rows * eval_env.cols,
            len(eval_env.action_delta),
            learning_rate=float(learning_rate),
            gamma=float(gamma),
            espilon=agent.espilon,
        )
        eval_agent.q_table = agent.q_table.copy()

        total_steps = 0
        total_reward = 0.0
        success_count = 0
        for _ in range(eval_episodes_for_ssa):
            step, success, reward = run_episode(
                eval_env,
                eval_agent,
                train=True,
                render=False,
                max_steps=max_steps_per_episode,
            )
            total_steps += step
            total_reward += reward
            success_count += int(success)

        avg_steps = total_steps / eval_episodes_for_ssa
        avg_reward = total_reward / eval_episodes_for_ssa
        failure_rate = 1 - success_count / eval_episodes_for_ssa
        return avg_steps + failure_rate * max_steps_per_episode - 0.01 * avg_reward # SSA 目标函数

    problem = { # 定义优化问题
        "bounds": FloatVar(lb=lb, ub=ub, name="lr_gamma"), # 两个决策变量的搜索范围
        "minmax": "min", # 最小化问题
        "obj_func": objective, # 目标函数 输入是 [lr, gamma] 输出的 fitness 分数
        "log_to": None, # 不输出 SSA 内部迭代日志
    }

    # 初始化并求解
    model = OriginalSSA(epoch=ssa_epochs, pop_size=ssa_pop)
    best = model.solve(problem)

    # 提取结果
    best_learning_rate, best_gamma = best.solution
    return float(best_learning_rate), float(best_gamma), float(best.target.fitness)

total_episodes = 2000
espilon_delay = 0.9
espilon_min = 0.1

steps_list = []
success_list = []
param_history = []
for episode in range(total_episodes):
    if episode > 0 and episode % adjust_interval == 0:
        best_lr, best_gamma, best_score = search_best_params_by_ssa(main_agent)
        main_agent.learning_rate = best_lr
        main_agent.gamma = best_gamma
        param_history.append((episode, best_lr, best_gamma, best_score))
        print(
            f"第 {episode} 轮 SSA 调参："
            f"learning_rate={best_lr:.4f}, gamma={best_gamma:.4f}, score={best_score:.2f}"
        )

    for i in range(3):
        step, success, _ = run_episode(
            env,
            main_agent,
            train=True,
            render=episode > total_episodes - 2,
            max_steps=max_steps_per_episode,
        )
    main_agent.espilon = max(espilon_min, main_agent.espilon * espilon_delay)
    steps_list.append(step)
    success_list.append(success)


plot_training_curve(steps_list, param_history)
plot_success_bar(success_list)
plot_param_evolution(param_history)
plot_q_heatmap(main_agent.q_table, env)
plot_max_q_heatmap(main_agent.q_table, env)
