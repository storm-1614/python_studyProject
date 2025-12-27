import numpy as np
import pandas as pd
import time
import os

np.random.seed(2)  # reproducible


N_STATES = 6  # the length of the 1 dimensional world
ACTIONS = ["left", "right"]  # available actions  # 行动
EPSILON = 0.9  # greedy police
ALPHA = 0.1  # 学习率 过大波动会很大
GAMMA = 0.9  #  对未来预期 看重未来 1 ，现实 0 折扣因子，衡量未来奖励有多重要
MAX_EPISODES = 13  # maximum episodes
FRESH_TIME = 0.3  # fresh time for one move


def build_q_table(n_states, actions):
    """
    Q 表
    位置 比如 6 种状态，Q 表的索引
    """
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),  # q_table initial values
        columns=actions,  # actions's name
    )
    # print(table)    # show table
    return table


def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]  # 将该位置的左右权数全部取出
    if (np.random.uniform() > EPSILON) or ((state_actions == 1).all()):  # 非理性状态
        action_name = np.random.choice(ACTIONS)
    else:  # 理性状态
        action_name = state_actions.idxmax()  # 取最大值作为 action 名字   idxmax() 也可以有一定随机选择的可能
        # 返回方向（id)
    return action_name


def get_env_feedback(S, A):
    """
    下一步的位置和奖励
    """
    if A == "right":  # move right
        if S == N_STATES - 2:  # terminate
            S_ = "terminal"
            R = 1  # 奖励
        else:
            S_ = S + 1
            R = 0
    else:  # move left
        R = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    return S_, R


def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ["-"] * (N_STATES - 1) + ["T"]  # '---------T' our environment
    if S == "terminal":
        interaction = "Episode %s: total_steps = %s" % (episode + 1, step_counter)
        print("\r{}".format(interaction), end="")
        time.sleep(2)
        print("\r                                ", end="")
    else:
        env_list[S] = "o"
        interaction = "".join(env_list)
        print("\r{}".format(interaction), end="")
        time.sleep(FRESH_TIME)


def rl():
    """
    预期收益 q_predict = 选择方向的 Q table 值
    实际收益 q_target = 奖励 + 未来最大收益
    0.1 冒险 0.9 现实
    """
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0  # 位置
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:
            A = choose_action(S, q_table)
            q_predict = q_table.loc[S, A]
            S_, R = get_env_feedback(S, A)  # S_ 下一步的收益
            if S_ != "terminal":
                q_target = (
                    R + GAMMA * q_table.iloc[S_, :].max()
                )  # 实际收益=奖励+未来预期最大收益（非终点）
            else:
                q_target = R  # next state is terminal
                is_terminated = True  # terminate this episode

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  #  更新 q 表
            S = S_  #  状态：往前走一步

            update_env(S, episode, step_counter + 1)  # 绘制新的游戏界面
            step_counter += 1
            print("")
            print(q_table)
            time.sleep(0.5)
            # input("按下任意键后继续（需按回车）...")
            os.system("clear")
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print("\r\nQ-table:\n")
    print(q_table)
