import pulp

# 求解下列动态规划问题：
#
# $$
# \begin{align}
# \max \quad & z = 2x_{1} + 3x_{2} - 5x{3}\\
# \text{s.t.}\quad &
# \begin{cases}
# x_{1} + x_{2} + x_{3} = 7\\
# 2x_{1} - 5 x_{2} + x_{3} \geq 10\\
# x_{1} + 3 x_{2} + x_{3} \leq 12\\
# x_{1}, x_{2}, x_{3} \geq0
# \end{cases}
# \end{align}
# $$

"""
创建一个线性规划问题对象
- 名称： LPProbDemo1
- 优化方向：最大化
"""
MyProbLP = pulp.LpProblem("LPProbDemo1", sense=pulp.LpMaximize)

"""
定义三个连续变量，且满足 0 <= xi <= 7
- lowBound 和 upBound 定义上下限
- cat="Continuous" 表示连续变量
"""
x1 = pulp.LpVariable("x1", lowBound=0, cat="Continuous")
x2 = pulp.LpVariable("x2", lowBound=0, cat="Continuous")
x3 = pulp.LpVariable("x3", lowBound=0, cat="Continuous")

"""
通过 += 添加函数到模型中
第一次对 `LpProblem 使用 `+=` 且没有比较符号时，即为目标函数
"""
MyProbLP += 2 * x1 + 3 * x2 - 5 * x3   # 添加目标函数
MyProbLP += x1 + x2 + x3 == 7          # 添加等式约束
MyProbLP += 2 * x1 - 5 * x2 + x3 >= 10 # 添加不等式约束（下界约束）
MyProbLP += x1 + 3 * x2 + x3 <= 12     # 添加不等式约束（上界约束）

MyProbLP.solve()  # 调用默认求解器进行求解
print("Status:", pulp.LpStatus[MyProbLP.status]) # 打印求解状态
for v in MyProbLP.variables():    # 遍历模型变量并打印各自的最优解值
    print(v.name, "=", v.varValue)
print("F(x) = ", pulp.value(MyProbLP.objective))  # 计算并打印目标函数的最优值
