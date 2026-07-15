import copy


class CliffWalkingEnv:
    """悬崖漫步环境"""

    def __init__(self, ncol=12, nrow=4) -> None:
        self.ncol = ncol  # 定义网格世界的列
        self.nrow = nrow  # 定义网络世界的行

        # 转移矩阵P[state][action] = [(p, next_state, reward, done)] 包含下一个状态和奖励
        self.P = self.createP()

    def createP(self) -> list:
        # 初始化
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]
        # 4种动作, change[0]:上,change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    # 位置在悬崖或者目标状态，因为无法继续交互，任何动作奖励都为 0
                    if i == self.nrow - 1 and j > 0: # 如果已经在悬崖或终点
                        # 任何动作都留在原地，奖励 0,done = True（游戏结束
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0, True)]
                        continue

                    # 否则正常移动
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False
                    if next_y == self.nrow - 1 and next_x > 0: # 走到最后一行到非起点位置
                        done = True
                        if next_x != self.ncol - 1: # 不是终点就是悬崖
                            reward = -100 # 掉下悬崖
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P


class PolicyIteration:
    """策略迭代算法"""

    def __init__(self, env: CliffWalkingEnv, theta, gamma) -> None:
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow  # 初始化价值为 0
        self.pi = [
            [0.25, 0.25, 0.25, 0.25] for i in range(self.env.ncol * self.env.nrow)
        ]  # 初始化为均匀随机策略
        self.theta = theta  # 策略评估收敛阈值
        self.gamma = gamma  # 折扣因子

    def policy_evaluation(self):  # 策略评估
        cnt = 1  # 计数器
        while 1: # 不停迭代直到收敛
            max_diff = 0 # 记录本轮 V 值最大变化量
            new_v = [0] * self.env.ncol * self.env.nrow # 新一轮 V 值表
            for s in range(self.env.ncol * self.env.nrow): # 遍历每一个格子
                qsa_list = []  # 开始计算状态 s 下的所有 Q(s, a) 价值
                for a in range(4): # 遍历 4 个动作（上下左右）
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        # 关键：如果 done = True（终点/悬崖），下一步 V = 0，没有未来
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                        # 本章环境比较特殊，奖励和下一个状态有关，所以需要和状态转移概率相乘
                    qsa_list.append(self.pi[s][a] * qsa) # 用策略概率加权
                new_v[s] = sum(qsa_list)  # 状态价值函数和动作价值函数之间的关系
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta: # V 值几乎不变了-> 收敛
                break  # 满足收敛条件，退出评估迭代
            cnt += 1
        print("策略评估进行 %d 轮后完成" % cnt)

    def policy_improvement(self):
        """
        策略提升——贪心
        对每个格子，看看四个方向哪个后续价值最大，就走哪个方向（概率1.0，其他方向 0）
        """
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa) # Q(s, 上) Q(s, 下) Q(s, 左) Q(s, 右)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)  # 计算有几个动作得到了最大的 Q 值
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list] # 让这些动作均分概率
        print("策略提升完成")
        return self.pi

    def policy_iteration(self):  # 策略迭代
        while 1:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)  # 将列表进行深拷贝，方便接下来进行比较
            new_pi = self.policy_improvement()
            if old_pi == new_pi:
                break


def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("状态价值：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 为了输出美观，保持输出 6 个字符
            print("%6.6s" % ("%.3f" % agent.v[i * agent.env.ncol + j]), end=" ")
        print()

    print("策略：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 一些特殊的状态，例如悬崖漫步中的悬崖
            if (i * agent.env.ncol + j) in disaster:
                print("****", end=" ")
            elif (i * agent.env.ncol + j) in end:  # 目标状态
                print("EEEE", end=" ")
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str = ""
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else "o"
                print(pi_str, end=" ")
        print()


env = CliffWalkingEnv()
action_meaning = ["^", "v", "<", ">"]
theta = 0.001
gamma = 0.9
agent = PolicyIteration(env, theta, gamma)
agent.policy_iteration()
print_agent(agent, action_meaning, list(range(37, 47)), [47])
