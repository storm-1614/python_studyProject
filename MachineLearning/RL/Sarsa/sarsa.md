# Sarsa 算法
基于悬崖行走。  
## 建模——MDP
悬崖行走是一个马尔可夫决策过程，由五元组定义：  

$\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma\rangle$  

- $\mathcal{S}$：状态空间（$ 4 \times 12 = 48$  个格子)  
- $\mathcal{A}$：动作空间（上下左右，共 4 个）
- $\mathcal{P}(s'\mid s, a)$：状态转移概率（此处为确定性转移）  
- $\mathcal{R}(s, a)$：奖励函数（普通步 -1，掉悬崖 -100）  
- $\gamma$ 折扣因子（代码是 0.9）

## 核心目标——动作价值函数
Sarsa 学习的是动作价值函数 $Q^{\pi}(s, a)$，即在状态 s 采取动作 a 后，遵循策略 $\pi$ 所能获得的期望累计折扣回报：

$Q^{pi}(s, a) = \mathbb{E}_{\pi}[\sum^{\infty}_{k=0} \gamma^{k} R_{t+k+1} \mid S_t = s, A_t = a]$  

## 更新规则推导
Sarsa 基于时序差分学习(TD Learning)，用一步实际采样来近似期望。  

### Belman 期望方程（理论目标）
$Q^{\pi}(s, a) = \mathbb{E}_{s', a'} [R + \gamma \cdot Q^{\pi}(s', a') \mid s, a]$  

其中 $a' \sim \pi(\cdot \mid s')$ 即从策略 $\pi$ 中采样下一动作。  

### TD 目标（一次采样的近似）
用单步实际采样 $(s, a, r, s', a')$ 替代期望：
$TD\ Target = r + \gamma \cdot Q(s', a')$

### TD 误差

$\delta_t = \underbrace{r + \gamma \cdot Q(s_{t+1}, a_{t+1})}_{\text{TD Target}} - \underbrace{Q(s_t, a_t)}_{\text{Current\ Estimate}}$

- $\delta_t > 0$：实际结果优于预期 -> 上调 $Q$  
- $\delta_t < 0$：实际结果差于预期 -> 下调 $Q$  

### 增量更新
以学习率 $\alpha$ 控制步长，向 TD Target 方向移动：
$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \cdot \delta_t$  

展开即：

$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha[r + \gamma \cdot Q(s_{t+1}, a_{t+1} - Q(s_t, a_t)]$  

## $\epsilon$ - 贪婪策略

行为策略与目标策略相同 (on-policy 的根本)：

$
\pi (a \mid s) = 
\begin{cases}
1-\epsilon + \frac{\epsilon}{\mathcal{\left| A \right|}} & if a = arg\ max_{a'} Q(s, a')\\
\frac{\epsilon}{\left| \mathcal{A} \right|} & otherwise
\end{cases}
$

这里 $\left| \mathcal{A} \right| - 4$，所以随机动作概率各为 $\frac{\epsilon}{4}$ 最优动作概率为 $1-\epsilon + \frac{\epsilon}{4}$  



