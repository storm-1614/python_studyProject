# REINFORCE算法
让能带来高回报的动作在未来更可能被选中  
## 整体架构
```
状态 → PolicyNet(状态) → softmax → 动作概率分布 → 采样 → 动作
                                          ↓
                              玩完一局后，用实际回报 G 来更新网络
```


## 动作选择
按概率分布随机采样

```python
def take_action(self, state):  # 根据动作概率分布随机采样
    state = torch.tensor([state], dtype=torch.float).to(self.device)
    probs = self.policy_net(state)  # 得到动作概率分布
    action_dist = torch.distributions.Categorical(probs) # 构造分类分布
    action = action_dist.sample() # 按概率随机采样
    return action.item()
```

REINFORCE 学的策略，网络输出是概率分布，即使某个动作概率很低，它仍然有机会被选中。探索是策略的固有属性。  
随着训练推进，好动作的概率自然会升高 $loss = -\log{\pi}\cdot G$ 差的动作自然降低

## 训练
$G = \gammma \cdot G + Reward$ 从最后一步往前遍历，逐步累加折扣回报。  

每一步的损失：$loss = - \log{\pi(a_t|s_t)\cdot G}$  


