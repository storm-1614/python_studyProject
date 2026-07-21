# PPO (Proximal Policy Optimization) — Update 过程详解

> 对应代码：`cartPole.py` 第 66–109 行

---

## 数据流总览

```
transition_dict (一局完整轨迹)
        │
        ▼
┌───────────────────────────┐
│  算 td_target & td_delta   │  ← 用 Critic 估值
│  td_target = r + γV(s')    │
│  td_delta = td_tgt - V(s)  │
└─────────┬─────────────────┘
          │
          ▼
┌───────────────────────────┐
│  算 GAE advantage          │  ← 倒着递推，让每一步看到远期回报
│  A_t = δ_t + γλ·A_{t+1}   │
└─────────┬─────────────────┘
          │
          ▼
┌───────────────────────────┐
│  记录 old_log_probs        │  ← 快照，detach() 切断梯度
│  log π_old(a|s)            │
└─────────┬─────────────────┘
          │
          ▼
     ┌────────────────────────────────┐
     │  for epoch in 1..10:           │  ← 同一批数据反复训练
     │                                │
     │  log_probs = log π_new(a|s)    │
     │  ratio = exp(log_p - old_log_p)│  ← π_new / π_old
     │  surr1 = ratio × A             │  ← 不约束
     │  surr2 = clamp(ratio) × A      │  ← 约束在 [0.8,1.2]
     │  loss = -min(surr1, surr2)     │  ← 永远取悲观的
     │         + MSE(V(s), td_target) │
     │  backward() + step()           │
     └────────────────────────────────┘
          │
          ▼
      丢弃数据，用更新后的策略重新采集
```

---

## 进入 update 之前：transition_dict 的数据结构

一局 CartPole 跑了 $T$ 步后，`transition_dict` 长这样：

```python
transition_dict = {
    "states":      [s₀,  s₁,  s₂,  ..., s_{T-1}]   # 每步的状态 (4维)
    "actions":     [a₀,  a₁,  a₂,  ..., a_{T-1}]   # 每步做的动作 (0或1)
    "rewards":     [r₀,  r₁,  r₂,  ..., r_{T-1}]   # 每步的奖励 (都是1)
    "next_states": [s₁,  s₂,  s₃,  ..., s_T]       # 每步的下一状态
    "dones":       [F,   F,   F,   ..., T  ]       # 最后一步是True
}
```

---

## 阶段 1：数据搬上 GPU（第 67–87 行）

```python
states = torch.tensor(transition_dict["states"], dtype=torch.float).to(self.device)
# 形状: [T, 4] — T 个状态，每个 4 维

actions = torch.tensor(transition_dict["actions"]).view(-1, 1).to(self.device)
# 形状: [T, 1] — T 个动作，每个是标量但保持 2D

rewards = torch.tensor(transition_dict["rewards"], dtype=torch.float).view(-1, 1).to(self.device)
# 形状: [T, 1]

next_states = torch.tensor(transition_dict["next_states"], dtype=torch.float).to(self.device)
# 形状: [T, 4]

dones = torch.tensor(transition_dict["dones"], dtype=torch.float).view(-1, 1).to(self.device)
# 形状: [T, 1] — True→1.0, False→0.0
```

`.view(-1, 1)` 的作用：把一维向量 `[T]` 变成二维列向量 `[T, 1]`，保证广播运算维度对齐。

---

## 阶段 2：算优势函数（第 88–92 行）

### 2.1 TD 目标（第 88 行）

```python
td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
```

| 项 | 形状 | 含义 |
|:--|:--|:--|
| `rewards` | `[T, 1]` | 即时奖励 $r_t$ |
| `self.critic(next_states)` | `[T, 1]` | Critic 对下一状态的估计值 $V(s_{t+1})$ |
| `self.gamma` | 标量 `0.98` | 折扣因子 $\gamma$ |
| `(1 - dones)` | `[T, 1]` | 最后一步是 $0$，其余是 $1$ |

数学形式：

$$V_t^{\text{target}} = r_t + \gamma \cdot V(s_{t+1}) \cdot (1 - \text{done}_t)$$

**具体例子**：

- 非最后一步（`done=0`）：

  $$V_t^{\text{target}} = 1.0 + 0.98 \times V(s_{t+1})$$

- 最后一步（`done=1`）：

  $$V_{T-1}^{\text{target}} = 1.0 + 0.98 \times V(s_T) \times 0 = 1.0$$

  最后一步没有"未来"，所以目标就是最后的即时奖励。

### 2.2 TD 误差（第 89 行）

```python
td_delta = td_target - self.critic(states)
# 形状: [T, 1]
```

$$\delta_t = \underbrace{(r_t + \gamma V(s_{t+1}))}_{\text{实际发生}} - \underbrace{V(s_t)}_{\text{预期}}$$

**例子**：Critic 预测当前状态值 50 分，实际拿 1 分 + 下个状态值 52 分：

$$\delta_t = (1 + 0.98 \times 52) - 50 = 51.96 - 50 = +1.96$$

- $\delta > 0$：做得比预期好
- $\delta < 0$：做得比预期差
- $\delta \approx 0$：中规中矩

### 2.3 GAE 优势（第 90–92 行）

```python
advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
# 形状: [T]
```

底层实现（`rl_utils.py`）：

```python
def compute_advantage(gamma, lmbda, td_delta):
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:                    # 从最后一步倒着往前
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
```

数学递推式（从末尾倒着算）：

$$\begin{aligned}
A_{T-1} &= \delta_{T-1} + \gamma\lambda \cdot 0 \\[4pt]
A_{T-2} &= \delta_{T-2} + \gamma\lambda \cdot A_{T-1} \\[4pt]
&\vdots \\[4pt]
A_0 &= \delta_0 + \gamma\lambda \cdot A_1
\end{aligned}$$

等价于指数加权展开式：

$$A_t^{\text{GAE}} = \sum_{k=0}^{\infty} (\gamma\lambda)^k \cdot \delta_{t+k}$$

**数值例子**（$\gamma\lambda = 0.98 \times 0.95 \approx 0.931$）：

```
假设 td_delta = [+2, -1, +5]（3步）

第3步: A₂ = 5 + 0.931 × 0    = 5.000
第2步: A₁ = -1 + 0.931 × 5.00 = 3.655   ← 自己的 δ 是 -1，但 GAE 是 +3.655！
第1步: A₀ = +2 + 0.931 × 3.66 = 5.404
```

**第 2 步演示了 GAE 的核心价值**：自己的 TD 误差是 -1（这一步亏了），但因为后续第 3 步大赚 +5，GAE 给这一步算出了 +3.655 的正优势。短期亏但长期赚的动作，仍然会被标记为"好动作"。

---

## 阶段 3：记录旧策略的对数概率（第 93 行）

```python
old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
```

逐步拆解：

**Step 1**：`self.actor(states)` → 形状 `[T, 2]`

对 $T$ 个状态，PolicyNet 输出每个动作的概率：

```
s₀ → [0.3, 0.7]    # 30% 向左，70% 向右
s₁ → [0.6, 0.4]    # 60% 向左，40% 向右
...
```

**Step 2**：`.gather(1, actions)` → 形状 `[T, 1]`

`actions = [[0], [1], ...]`，在 dim=1（动作维度）上按索引取值：

```
s₀, 实际选了动作 0 → 取概率 0.3
s₁, 实际选了动作 1 → 取概率 0.4
```

**Step 3**：`torch.log(...)` → 取对数

$$\log \pi_{\theta_{\text{old}}}(a_t \mid s_t)$$

**Step 4**：`.detach()` — **切断梯度，至关重要**

`detach()` 让 `old_log_probs` 从计算图中摘下来，在后续 10 轮训练中保持固定——它是"更新前"的快照，必须作为不变的参照基准。

---

## 阶段 4：PPO 多轮裁剪更新（第 95–109 行）⚠️ 核心

```python
for _ in range(self.epochs):    # epochs=10，同一批数据训练10轮
```

### 4.1 新策略的对数概率（第 96 行）

```python
log_probs = torch.log(self.actor(states).gather(1, actions))
```

和 `old_log_probs` 的计算完全一样，但**没有 `detach()`**——`self.actor` 参与计算图，梯度将流经这里。

第 1 轮时 `log_probs = old_log_probs`；随着参数更新，后续轮次中 `log_probs` 逐渐偏离 `old_log_probs`。

### 4.2 重要性采样比率（第 97 行）

```python
ratio = torch.exp(log_probs - old_log_probs)
```

数学推导：

$$r_t(\theta) = \exp(\log \pi_\theta - \log \pi_{\theta_{\text{old}}}) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

**含义**：

| $r_t$ | 含义 |
|:--|:--|
| 1.0 | 新策略和旧策略对这个动作的态度一样 |
| 1.5 | 新策略比旧策略更喜欢这个动作（概率变为 1.5 倍） |
| 0.5 | 新策略更不喜欢它（概率减半） |

训练轮次的推进中 `ratio` 的变化：

```
第 1 轮: ratio ≈ 1.0（还没开始变）
第 3 轮: ratio 可能在 0.7~1.3
第 10 轮: 如果没有裁剪，ratio 可能飙到 5.0+
```

### 4.3 无约束目标（第 98 行）

```python
surr1 = ratio * advantage
```

$$\mathcal{L}^{\text{unclipped}}_t = r_t(\theta) \cdot A_t$$

**问题**：当 $r_t$ 在多次训练后变得很大（如 5.0），用过期数据指导大变的策略，重要性采样的修正不再可靠，可能导致策略崩溃。

### 4.4 裁剪目标（第 99 行）

```python
surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
```

`eps = 0.2` → ratio 被钳制在 `[0.8, 1.2]`：

```python
原始 ratio:  [1.0,  1.3,  0.5,  2.0,  0.9]
clamp 后:    [1.0,  1.2,  0.8,  1.2,  0.9]
             不变  截断  截断  截断  不变
```

$$\mathcal{L}^{\text{clipped}}_t = \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon) \cdot A_t$$

### 4.5 PPO-Clip 损失（第 100 行）

```python
actor_loss = torch.mean(-torch.min(surr1, surr2))
```

**这是整个 PPO 算法的数学核心：**

$$\boxed{\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\Big(r_t(\theta) A_t,\; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \cdot A_t\Big)\Big]}$$

前面加负号是因为 PyTorch 做的是**梯度下降**，而我们要最大化期望回报。

**四种情况逐行分析：**

#### Case 1：好动作 ($A > 0$)，ratio 过大 ($r > 1+\varepsilon$)

```
A = +5, r = 1.5
surr1 = 1.5 × 5 = 7.5
surr2 = 1.2 × 5 = 6.0
min(surr1, surr2) = 6.0  → 取 surr2
```

**解读**：这是个好动作，策略想大幅增加它的概率。PPO 把 ratio 截在 1.2，**阻止策略变得过于贪婪**——好动作也别膨胀太猛。

#### Case 2：好动作 ($A > 0$)，ratio 正常

```
A = +5, r = 1.0
surr1 = 1.0 × 5 = 5.0
surr2 = 1.0 × 5 = 5.0
min(surr1, surr2) = 5.0  → 两者相等
```

**解读**：在安全区内，正常鼓励。

#### Case 3：坏动作 ($A < 0$)，ratio 过小 ($r < 1-\varepsilon$)

```
A = -5, r = 0.5
surr1 = 0.5 × (-5) = -2.5
surr2 = 0.8 × (-5) = -4.0
min(surr1, surr2) = -4.0  → 取 surr2（更负的）
```

**解读**：这是个坏动作，策略想把它踩死（ratio 很小说明概率大幅下降）。但 PPO 把 ratio 截在 0.8，**防止惩罚过度**——坏动作也别一棍子打死。

#### Case 4：坏动作 ($A < 0$)，ratio 过大 ($r > 1+\varepsilon$)

```
A = -5, r = 1.5
surr1 = 1.5 × (-5) = -7.5
surr2 = 1.2 × (-5) = -6.0
min(surr1, surr2) = -7.5  → 取 surr1（更负的）
```

**解读**：策略反而更喜欢一个坏动作，`min` 取 surr1 确保足够惩罚，把它拉回正轨。

#### 汇总表

| 优势 $A$ | ratio 方向 | 谁更保守 | 效果 |
|:--|:--|:--|:--|
| $A > 0$ | $r > 1.2$（过于喜欢） | surr2（裁剪） | 🛑 刹车：别膨胀太猛 |
| $A > 0$ | $r \in [0.8, 1.2]$ | 相等 | ✅ 正常鼓励 |
| $A < 0$ | $r < 0.8$（过于厌恶） | surr2（裁剪） | 🛑 刹车：别再踩了 |
| $A < 0$ | $r \in [0.8, 1.2]$ | 相等 | ✅ 正常惩罚 |
| $A < 0$ | $r > 1.2$（反而喜欢） | surr1（不裁） | ⚡ 加大惩罚，纠正错误 |

**核心直觉**：$\min(surr1, surr2)$ 总是选择**更新幅度更小、更保守**的那一侧。它创造了一个不对称的安全区：ratio 偏离安全区时不会获得额外收益，但可以自由缩回安全区。

### 4.6 Critic 损失（第 101–103 行）

```python
critic_loss = torch.mean(
    F.mse_loss(self.critic(states), td_target.detach())
)
```

$$\mathcal{L}_{\text{critic}} = \frac{1}{T}\sum_{t=0}^{T-1}\big(V_\phi(s_t) - V_t^{\text{target}}\big)^2$$

让 Critic 网络 $V_\phi$ 去拟合 TD 目标。`.detach()` 阻断 `td_target` 的梯度回传到 Actor。

### 4.7 梯度更新（第 104–109 行）

```python
self.actor_optimizer.zero_grad()    # 清空 Actor 梯度
self.critic_optimizer.zero_grad()   # 清空 Critic 梯度
actor_loss.backward()               # 沿 Actor 计算图反向传播
critic_loss.backward()              # 沿 Critic 计算图反向传播
self.actor_optimizer.step()         # 更新 Actor 参数
self.critic_optimizer.step()        # 更新 Critic 参数
```

**注意**：PyTorch 默认**累加梯度**，每轮训练前必须 `zero_grad()` 清零，否则梯度会不断叠加。

---

## 完整数学公式汇总

### TD 目标与误差

$$V_t^{\text{target}} = r_t + \gamma V_\phi(s_{t+1}) (1 - \text{done}_t)$$

$$\delta_t = V_t^{\text{target}} - V_\phi(s_t)$$

### GAE 优势函数

$$A_t^{\text{GAE}} = \sum_{k=0}^{\infty} (\gamma\lambda)^k \cdot \delta_{t+k}$$

递推形式：

$$A_t = \delta_t + \gamma\lambda \cdot A_{t+1}$$

### PPO-Clip 目标函数

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

$$\boxed{\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\Big(r_t(\theta) A_t,\; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \cdot A_t\Big)\Big]}$$

### Critic 损失

$$\mathcal{L}^{\text{VF}}(\phi) = \mathbb{E}_t\Big[\big(V_\phi(s_t) - V_t^{\text{target}}\big)^2\Big]$$

---

## 超参数一览

| 参数 | 值 | 含义 |
|:--|:--|:--|
| `gamma` | 0.98 | 折扣因子，越接近 1 越看重远期回报 |
| `lmbda` | 0.95 | GAE 的 λ：越接近 1 越偏蒙特卡洛（无偏但高方差），越接近 0 越偏单步 TD（有偏但低方差） |
| `epochs` | 10 | 同一批轨迹数据重复训练的轮数 |
| `eps` | 0.2 | PPO 裁剪范围 $[0.8, 1.2]$：越小越保守 |
| `actor_lr` | 1e-3 | Actor 学习率 |
| `critic_lr` | 1e-2 | Critic 学习率（通常比 Actor 大） |

---

## 与 Actor-Critic 的关键区别

| 维度 | Actor-Critic | PPO |
|:--|:--|:--|
| **优势估计** | 单步 TD 误差 $\delta_t$ | GAE 多步加权 $A_t^{\text{GAE}}$ |
| **更新次数** | 每轮数据只用 1 次 | 每轮数据反复用 `epochs=10` 次 |
| **更新安全机制** | 无 | PPO-Clip: $\min(rA, \text{clip}(r)A)$ |
| **采样效率** | 低（数据用一次就扔） | 较高（同一批数据重复利用） |