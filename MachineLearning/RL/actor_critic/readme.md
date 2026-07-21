# Actor Critic 算法
- Actor: 演员/策略网络，负责做动作
- Critic：评论家/价值网络，不操作，只打分  

Actor 管选什么动作，Critic 管评价这个局面好不好。  
Actor 在学习时需要知道，我选的这个动作，到底比平均水平好多少，这就是优势函数。  
优势计算的公式是：实际拿到的总奖励 - Critic 预测的平均分。  
实际拿到的总奖励：从当前状态开始，一路玩到最后所有奖励之和  
Critic 预测的平均分：Critic 根据当前的状态，直接打出的那个分数 V  

## 一起训练：
Actor 的目标：增大那些优势为正的动作的概率，减少优势为负的的动作的概率  
Critic 的

## 动作选择
Actor 根据概率分布采样选动作。  

```python
def take_action(self, state):
    state = torch.tensor([state], dtype=torch.float).to(self.device)
    probs = self.actor(state)
    action_dist = torch.distributions.Categorical(probs)
    action = action_dist.sample()
    return action.item()
```

## 学习
### 计算 TD 目标
``` python
td_target = rewards + self.gamma * self.critic(next_states) * (1 -
```

这就是实际能拿到的总回报估计：立即奖励 + 折现后下一个状态的 V 值。  
`(1 - done)` 是为了处理游戏结束：如果结束了，未来就没有了，只算立即奖励奖励。  

### TD 误差
``` python
td_delta = td_target - self.critic(states)
```

TD 误差是评论家的惊讶程度：  
如果 td_target 比原来预测的 V(s) 大很多，td_delta 就是正数：告诉演员刚才动作不错，比预想的好。  

### 用 TD 误差更新演员
``` python
log_probs = torch.log(self.actor(states).gather(1, actions))
actor_loss = torch.mean(-log_probs * td_delta.detach())
```

`self.actor(states)` 拿到当前状态下的新概率分布，`.gather(1, actions)` 取出实际执行那个动作对应的概率。再取 log 得到对数概率。  
`actor_loss = -log_prob  * td_delta` 用 `detach()` 固定住，不回传给 critic。  

- 如果 td_delta > 0（动作比预期好），-log_prob * 正数 会变小，优化器为了最小化 actor_loss，会把 log_prob 变大，即增大这个动作的概率。

- 如果 td_delta < 0（动作比预期差），-log_prob * 负数 会变大，优化器会把 log_prob 变小，即减小这个动作的概率。  

### 用 TD 目标训练评论家
``` python
critic_loss = torch.mean(
    F.mse_loss(self.critic(states), td_target.detach()))
```

评论家是让 $V(s)$ 尽量接近 td_target (实际回报估计),用 MSE 损失去更新，这是一个标准的回归问题。  

### 执行反向传播和参数更新
``` python
self.actor_optimizer.zero_grad()
self.critic_optimizer.zero_grad()
actor_loss.backward()  # 计算策略网络的梯度
critic_loss.backward()  # 计算价值网络的梯度
self.actor_optimizer.step()  # 更新策略网络的参数
self.critic_optimizer.step()  # 更新价值网络的参数
```


