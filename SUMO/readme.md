# 从简单两相位模拟到 SUMO 研究学习
## 简单两相位单路口仿真
两相位的单路口仿真核心不在于如何仿真。实际是搭建出初步的环境框架，在此基础上接入 RL 算法。  
目前主流的 RL 学习环境是基于 OpenAI 的 gym 迭代来的 Farama-Foundation/Gymnasium。比如最经典的倒立摆就是 Gymnasium 下的一个环境。Gymnasium 还提供了 `Gymnasium.Env` 这个基类来自定义环境，由此我写了一篇博客来讲关于自定义环境：[https://storm1614.top/p/gymnasium-spaces-library](https://storm1614.top/p/gymnasium-spaces-library/)  

这里的环境没有可视化，仅分南北向和东西向累计两方向各自随机而来的等候车流，车辆只能直行，不过这也足以。核心数据如下：  

``` python
self.phase = 0  # 当前绿灯相位 0/1
self.queue = [0, 0]  # 两个方向的排队分档
```

设置最大单向车流为 4 辆，再多意义不大，而且 Q 表还要再往外扩展。车流增加在 `step()` 按如下操作：  
``` python
for i in range(2):
    self.queue[i] = min(4, int(self.queue[i] + self.np_random.integers(0, 2)))
```
这样，也大致可以知道 `self.queue` 意义了。  

step 函数流程图如下，上述随机添加 queue 为车辆到达部分。  

![](./res/qlearning_step_flow_chart.png)

按照 `Gymnasium API`，step 输入动作输出各种状态，其中比较重要的是观测值，也就是环境当前的各种状态。需要为环境定义当前提供给智能题的信息这里返回的是一个元组，用`_obs()` 函数进行额外的封装：  

``` python
def _obs(self):
    """
    return: 
    queue[0] 方向 A 排队数
    queue[1] 方向 B 排队数
    phase    绿灯相位
    """
    return np.array([self.queue[0], self.queue[1], self.phase], dtype=int)
```

代码在 `./qlearningRordNet.py` 。  

Q Learning 类实际很普通，无非就是选择动作、更新 Q 表两件事。因为观测值包含两向排队数和绿灯相位，所以 q 表定义的是一个四维表，不过也不复杂，索引直接按观测值 `obs` 元组进行索引即可，初始化利用 python 的元组的组合来创建。  
``` python
self.qTable = np.zeros(n_states + (n_actions,))
```
一个 state(obs) 是 3 元素的元组，这里写作 state(状态) 惯性使然，当成观测值即可。n_actions 为整形。  
可以看作：(5, 5, 2) + (2,) 相加为元组：(5, 5, 2, 2) 传给 `np.zero` 就创建出来 4 维的 `NPArray` 数据结构了。  

后续很多需要对一个观测值进行索引，直接传观测值元组即可找到匹配的两种动作的 Q 值，如下：  
```  python
action = np.argmax(self.qTable[tuple(obs)])
```

还有 espilon 收敛之类就不多展开，讲了比较多的代码细节，接下来放上训练结果。  
这里 `total` 对每次所有方向的排队数取负数进行累加，累计每 50 步放出来。`total` 越少说明训练效果越好：  

![](./res/qlearning_total.png)

收敛结果一般般吧，不过此实验主要是把框架搭起来。  

## 引入 DQN
QLearning 的收敛结果不佳，所以就引入深度强化学习算法 DQN。需要用 pyTorch 库建立神经网络。这也导致计算量暴增。我们的环境类保持不变不做修改，仅将 QLearning 换成 DQN。这里的 DQN 也做了比较好的创新，对目标网络的更新从硬更新，一段时间深拷贝一次变成追影子式的动态更新。这在 [#3296f3f](https://github.com/storm-1614/python_studyProject/commit/3296f3fda21f9ec3afa27cd996ac8464edd9913f) 中做单独的 commit 更好的体现。  

软拷贝到目标网络更新代码为：  
``` python
for target_param, q_param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
    target_param.data.copy_(self.tau * q_param.data + (1 - self.tau) * target_param.data)
```
这里的 tau 是一个很小的数值，这里为  0.007。  

除此之外，就是常规的双 Q 网络，经验回放之类可以看代码。为了和最新的 Gymnasium 框架做融合还是费了不少功夫，特别是 `torch.tensor` 张量的处理还是略显吃力。  
因为有经验回放池，加上参数也很多，给智能体 `update` 时用了字典进行传入，加上 DQN 的 update 本身就比较复杂代码写得略有些凌乱。  

``` python
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
```

DQN 的收敛效果显然更好，基本上可以收敛到最优的结果。  

![](./res/dqn_total.png)
