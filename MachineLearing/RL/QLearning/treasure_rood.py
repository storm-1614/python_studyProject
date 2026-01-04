import numpy as np
import pandas as pd
import time
import os

np.random.seed(2)

LEFT_STATE = 3  # 左侧长度
RIGHT_STATE = 3  # 右侧长度
UP_STATE = 3  # 上部长度
DOWN_STATE = 3  # 下部长度
EPSILON = 0.9  # 理性值
ALPHA = 0.1  # 学习率
GAMMA = 0.9  # 对未来的预期
ACTIONS = ["left", "right", "up", "down"]  # 可用方向

MAX_EPISODES = 13  # 最大回合数
FRESH_TIME = 0.1  # 刷新间隔
nextAction = [""] * (LEFT_STATE + RIGHT_STATE + UP_STATE + DOWN_STATE + 1)
AWARD_DOT = [6, 7, 12]  # 奖励列表
# TODO: 未来可以实现一个惩罚表
"""
    迷宫图：

   7
   8
   9
0123456
   10
   11
   12
"""


def buildActionList(nextAction, left, right, up, down):
    """
    创建动作列表
    动作列表用于表示该块可以有的动作
    """
    for i in range(left + right + 1):
        nextAction[i] = ["left", "right"]
    nextAction[left] = ["left", "right", "up", "down"]
    for i in range(left + right + 1, left + right + up + down):
        nextAction[i] = ["up", "down"]


def buildQTable(nState, actions):
    """
    创建 Q 表
    需要所有位置为列
    所有可能动作为行
    """
    table = pd.DataFrame(np.zeros((nState, len(actions))), columns=actions)
    return table


def chooseAction(state, qTable):
    """
    选择下一步的方向
    """
    stateActions = qTable.iloc[state, :]  # 将该位置的方向权数全部取出
    if np.random.uniform() > EPSILON or (stateActions == 0).all():
        """
        非理性状态
        """
        actionName = np.random.choice(nextAction[state])
    else:
        """
        理性状态
        """
        actionName = stateActions.idxmax()

    return actionName


def getEnvFeedback(pos, action):
    """
    这个函数通过选择的 action 对其进行计算，判断下一步的位置和奖励情况
    有两个返回值：
        下一步位置，奖励
    """
    if action == "right":
        if pos == LEFT_STATE + RIGHT_STATE: 
            nextPos = pos
        else:
            nextPos = pos + 1
        if nextPos in AWARD_DOT:
            award = 1
        else:
            award = 0

    elif action == "left":
        if pos == 0:
            nextPos = pos
        else:
            nextPos = pos - 1

        if nextPos in AWARD_DOT:
            award = 1
        else:
            award = 0

    elif action == "up":
        if pos == LEFT_STATE:  # 当前为在 3 点处
            nextPos = LEFT_STATE + RIGHT_STATE + UP_STATE  # 跳到 9 处
        elif pos == LEFT_STATE + RIGHT_STATE + UP_STATE + 1:  # 在 10 处
            nextPos = LEFT_STATE  # 跳到 3 处
        elif pos == LEFT_STATE + RIGHT_STATE + 1:  # 在 7 处
            nextPos = pos
        else:
            nextPos = pos - 1

        if nextPos in AWARD_DOT:
            award = 1
        else:
            award = 0

    elif action == "down":
        if pos == LEFT_STATE:  # 在 3 处
            nextPos = LEFT_STATE + RIGHT_STATE + UP_STATE + 1
        elif pos == LEFT_STATE + UP_STATE + RIGHT_STATE:  # 在 9 处
            nextPos = LEFT_STATE
        elif pos == LEFT_STATE + RIGHT_STATE + UP_STATE + DOWN_STATE:  # 在 12 处
            nextPos = pos
        else:
            nextPos = pos + 1

        if nextPos in AWARD_DOT:
            award = 1
        else:
            award = 0

    else:
        nextPos = pos
        award = 0

    if award == 1:
        # 判断是否为奖励处
        nextPos = "terminal"
    return nextPos, award


def updateEnv(S, episode, stepCounter):
    """
    用于绘制迷宫图
    以及在获得奖励后进行结算
    envList 这个字符串包含了所需要绘制的符号
            其中用 o 代表当前位置 T 代表奖励
    同时在底部还需要打印位置
    """
    envList = ["-"] * (LEFT_STATE + RIGHT_STATE + UP_STATE + DOWN_STATE + 1)
    for i in AWARD_DOT:
        # 将奖励位置绘制为 T
        envList[i] = "T"
    if S == "terminal":
        """
        如果遇到奖励也就是该回合结束，进行回合结算
        """
        interaction = "Episode %s: total_steps = %s" % (episode + 1, stepCounter)
        print("\r{}".format(interaction), end="")
        time.sleep(1)
        print("\r                                ", end="")

    else:
        envList[S] = "o" # 将当前位置符号改为 o
        line = "".join(envList[0 : LEFT_STATE + RIGHT_STATE + 1])
        for i in range(UP_STATE): # 打印上支路
            print(" " * LEFT_STATE, end="")
            print("{}".format(envList[LEFT_STATE + RIGHT_STATE + 1 + i]))
        print("{}".format(line))  # 打印左右支路
        for i in range(DOWN_STATE): # 打印下支路
            print(" " * LEFT_STATE, end="")
            print("{}".format(envList[LEFT_STATE + RIGHT_STATE + UP_STATE + 1 + i]))

    print(S)
    time.sleep(FRESH_TIME)


def main():
    """
    主函数
    首先初始化了两个列表
    """
    buildActionList(nextAction, LEFT_STATE, RIGHT_STATE, UP_STATE, DOWN_STATE)
    qTable = buildQTable(LEFT_STATE + RIGHT_STATE + UP_STATE + DOWN_STATE + 1, ACTIONS)

    for episode in range(MAX_EPISODES):
        stepCounter = 0  # 步数计数器
        pos = 0  # 所处的位置
        isTerminated = False  # 是否结束
        updateEnv(pos, episode, stepCounter)  #绘制初始迷宫图

        while not isTerminated:
            action = chooseAction(pos, qTable) # 获得下一步的方向
            qPredict = qTable.loc[pos, action] # 计算预期收益
            nextPos, award = getEnvFeedback(pos, action) # 获得下一步的位置和奖励情况
            """
            接下来的判断语句是获得实际收益
            是否有奖励对实际收益的影响是不一样的
            """
            if nextPos != "terminal":
                qTarget = award + GAMMA * qTable.iloc[nextPos, :].max()
            else:
                qTarget = award
                isTerminated = True

            qTable.loc[pos, action] += ALPHA * (qTarget - qPredict) # 计算收益并更新 q 表
            pos = nextPos # 更新位置

            updateEnv(pos, episode, stepCounter) # 更新地图
            stepCounter += 1 # 计数器加一
            print("")
            print(qTable)
            time.sleep(FRESH_TIME)

            # 清屏操作
            # 依据不同操作系统使用不同命令
            if os.name == "nt":
                os.system("cls")
            else:
                os.system("clear")

    return qTable


if __name__ == "__main__":
    qTable = main()
    print("\r\nQ-table:\n")
    print(qTable)
