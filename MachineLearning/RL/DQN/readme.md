# CartPole 代码笔记

## 动作选择
```python
def choose_action(self, state):
    if np.random.rand() < self.epsilon:
        return np.random.randint(0, 2)  # CartPole 左右两个动作
    else:
        state_tensor = torch.FloatTensor(state)
        q_values = self.q_net(state_tensor)
    return q_values.cpu().detach().numpy().argmax()
```

这里 q_values 返回两个元素的数组，q_values[0]  是向左推小车的期望累积回报，q_values[1] 是向右推小车的期望累积回报。  

``` python 
q_values        # PyTorch 张量，shape=(2,) 两个动作的 Q 值    
    .cpu()      # 确保数据在 CPU 上
    .detach()   # 从计算图中剥离，不追踪梯度
    .numpy()    # 转为 NumPy 数组
    .argmax()   # 返回最大值的索引，即选中的动作
```

也就是说：**从神经网络输出的两个 Q 值中，挑出值更大的那个动作的编号（0=左，1=右）**。`.cpu().detach().numpy()` 是从 PyTorch 推理后转 NumPy 的标准模板写法。  


## 训练
``` python
def train(self):
    if len(self.replay_buffer) < self.batch_size:
        return

    # 从缓冲区随机采样
    batch = random.sample(self.replay_buffer, self.batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(np.array(states))
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(np.array(next_states))
    dones = torch.FloatTensor(dones)

    # 计算当前 Q 值
    current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()

    # 计算目标 Q 值（使用目标网络)
    with torch.no_grad():
        next_q = self.target_net(next_states).max(1)[0]
        target_q = rewards + self.gamma * next_q * (1 - dones)

    # 计算损失并更新网络
    loss = nn.MSELoss()(current_q, target_q)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    # 定期更新目标网络
    self.step_count += 1
    if self.step_count % self.update_target_freq == 0:
        # 使用深拷贝更新目标网络参数
        self.target_net.load_state_dict(
            {k: v.clone() for k, v in self.q_net.state_dict().items()}
        )
```

随机采样：打破时间相关性，提高数据利用率。  
顺序采样会：
- 过拟合近期经验：值学会在当前小区域做决策，忘了之前学到的
- 梯度更新方向单一：连续相似样本的梯度方向基本一致，参数更新像是在走同一条窄路，容易震荡
- 违反 SGD 的前提假设：随机梯度下降要求样本独立同分布，连续经验显然不是  

使用随机采样，就像是：随机翻页抽查，同时巩固所有章节的知识点。还能复用数据，在 buffer 里反复随机采样，每条经验可以被学习很多次，同等交互次数下训练效率高出很多。  

--- 
``` python
# 计算当前 Q 值
current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
```

这里 `self.q_net(states)` 会输出采样动作的 Q 值（2 * 64) 的表。  

`.gather(1, actions.unsqueeze(1))` 沿着列方向，按 action 索引取值。  
`actions.unsqueeze(1) = [[1], [0], [1]]` 每行的列索引。  

`.squeeze()` 压掉多余的维度。  
例子：
``` python
self.q_net(states)               # shape: (3, 2) — 每行是 [Q_左, Q_右]
[[1.2, 3.5],                     #   状态0：左=1.2, 右=3.5
 [2.1, 1.8],                     #   状态1：左=2.1, 右=1.8
 [0.9, 4.2]]                     #   状态2：左=0.9, 右=4.2

.gather(1, actions.unsqueeze(1)) # shape: (3, 1) — 沿着列方向，按 action 索引取值
# actions.unsqueeze(1) = [[1], [0], [1]]  每行的列索引
# → 第0行取列1=3.5, 第1行取列0=2.1, 第2行取列1=4.2
[[3.5],
 [2.1],
 [4.2]]

.squeeze()                        # shape: (3,) — 压掉多余的维度
[3.5, 2.1, 4.2]                  # 就是实际执行的三个动作各自的 Q 值
```

DQN 的损失函数是 TD 误差：  

$Loss = MSE(Q(s, a), r + \gamma \cdot max_{a'}Q_{target}(s', a'))$  

target_q：用目标网络算 `next_states` 所有动作都最大 Q 值，应该达到的目标。  
current_q：用当前网络算 `states`，但只需实际执行动作 `a` 的 Q 值。因为只有这个动作是被执行的，只需要修正它。  

`gather` 的作用就是从 [Q_左, Q_右]，中精准挑出那个被执行动作的 Q 值，其余动作不参与本轮更新。  

```python
# 计算目标 Q 值（使用目标网络)
with torch.no_grad():
    next_q = self.target_net(next_states).max(1)[0]
    target_q = rewards + self.gamma * next_q * (1 - dones)
```
`next_states` 是来自 env.step() 的返回值，代表了智能体在环境中走一步后环境告诉它的新状态。  
将这些参数传入 target_net，取出最大的那个值。  

`with torch.no_grad` 关闭梯度更新，这样不会更新 target_net。  
`.max(1)` 沿着第 1 轴方向（即列方向）取最大值，取的是每个状态的最优动作 Q 值。这里计算从下一状态出发的最优 Q 值估算。  

接下来的 `target_q` 计算的是 TD 目标，$r+\gamma \cdot Q_{next} (1-dones$。  

``` python
# 计算损失并更新网络
loss = nn.MSELoss()(current_q, target_q)
self.optimizer.zero_grad()
loss.backward()
self.optimizer.step()
```

这里计算损失，梯度清零，反向传播，参数更新。对Q网络进行更新。  

``` python
loss = nn.MSELoss()(current_q, target_q)
```
**计算损失**，对 `q_value` (当前 Q 网络对实际采取动作的预测 Q 值) 和 `q_targets` (TD 目标) 做均方误差，度量当前 Q 网络对预测值与应该接近目标的差距，这个损失就是 DQN 要最小化的东西。  

``` python
self.optimizer.zero_grad()
```

**清零梯度**，PyTorch 中的梯度默认是累计的，在每次反向传播前必须显式清零，否则会混入上一批的梯度。  

``` python
loss.backward()
```
**反向传播**,从 loss 出发，沿计算图反向传播，计算出 loss 对 Q 网络每个参数的梯度，存到 `.grad` 属性里。  

``` python
self.optimizer.step()
```
**更新参数**， Adam 优化器根据刚才计算出的梯度和学习率，更新 Q 网络对参数。使 loss 下降，只更新 q_net,不懂 target_q_net。  

也就是说，这段代码是：**拿来一批经验，算出当前预测与目标值的差距，把这个差距反向传播，用梯度下降把 Q 网络对参数往更准的方向挪一小步。**  

``` python
# 定期更新目标网络
self.step_count += 1
if self.step_count % self.update_target_freq == 0:
    # 使用深拷贝更新目标网络参数
    self.target_net.load_state_dict(
        {k: v.clone() for k, v in self.q_net.state_dict().items()}
    )
```

定期更新 target_q，直接用深拷贝把 Q网络复制到目标网络。  

