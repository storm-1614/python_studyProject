# Double DQN 算法
普通 DQN 算法通常会导致对 Q 值的过高估计。传统 DQN 优化的 TD 误差目标为：

$$
r+\gamma \cdot max_{a'}Q_{w^{-}}(s', a')
$$

其中 $max_{w^-}(s', a')$ 由目标网络（参数为 $w^-$ 计算得出，我们还可以将其写成如下形式：  

$$
Q_{w^-}(s', argmax_{a'}Q_{w^-}(s', a'))
$$

$max$ 操作首先选取状态 $s'$ 下的最优动作 $a^* = arg max_a'Q_{w^-}(s', a')$，接着计算该动作对应的价值 $Q_{w^-}(s', a^*)$。当这两部分采用同一套 Q 值计算时，每次得到的都是神经网络估算的所有动作价值中的最大值。考虑到通过神经网络估算的值本身在某些时候会产生正向或负向的误差，在 DQN 的更新方式下神经网络会将正向误差累积。  
这样，DQN 会对 Q 值有过高的估计，造成 DQN 无法有效工作。  

为了解决，Double DNQ 利用两个独立训练的神经网络估算 $max_{a'}Q_*(s', a')$。将原有的 $max_{a'}Q_{w^-}(s', a')$ 更改为 $Q_{w^-}(s', argmax_{a'}Q_w(s', a'))$ 即利用一套神经网络 $Q_w$ 的输出选取价值最大的动作，但在使用该动作的价值时，用另一套神经网络 $Q^-_w$ 计算该动作的价值。这样即使其中一套神经网络的某个动作存在比较严重的过高估计问题，由于另一套神经网络的存在，这个动作最终使用的 $Q$ 值不会存在很大的过高估计问题。  

传统 DQN 算法，$max_{a'}Q_{w^-}(s', a')$ 的计算只用到了其中的目标网络，可以直接将训练网络作为 Double DQN 算法中的第一套神经网络来选取动作，将目标网络作为第二套神经网络计算 $Q$ 值，这便是 Double DQN 的主要思想。  

$$
r + \gamma \cdot Q_{w^-}(s', arg max_{a'}Q_w(s', a'))
$$

``` python
if self.dqn_type == "DoubleDQN":  # DQN 与 Double DQN 的区别
    max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
    max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
else:
    max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
```
