import pulp
import matplotlib.pyplot as plt

n = 4

# 银行存款 s0 的参数
r0 = 0.05  # 年收益率 3%
p0 = 0  # 交易费率
s0 = 0  # 风险损失率

# 4 个投资项目
data = [
    {"r": 0.28, "p": 0.010, "q": 0.025,},
    {"r": 0.21, "p": 0.020, "q": 0.015,},
    {"r": 0.23, "p": 0.045, "q": 0.055,},
    {"r": 0.25, "p": 0.065, "q": 0.026,},
]

M = 1.0

a = 0.05


def solve_for_risk_limit(risk_limit: float) -> tuple[str, list[float], float, float]:
    """
    在给定风险上限下求解最优投资方案
    参数：
    risk_limit: float
        风险上限 a，表示最大允许风险损失占总资金的比例。

    Returns
    -------
    status : str
        求解状态（"Optimal" 表示最优解, "Infeasible" 表示无可行解等）。
    values : list[float]
        决策变量 [x0, x1, x2, x3, x4]，依次为银行存款和 4 个投资项目的资金分配额。
    net_return : float
        最大净收益，即目标函数值 r0*x0 + Σ(r_i - p_i)*x_i。
    risk_ratio : float
        组合实际风险比例，即 Σ(q_i * x_i) / M。
    """
    # 创建最大化问题
    model = pulp.LpProblem("LpProblem1", pulp.LpMaximize)

    # 决策变量 x_i
    # x_0：银行存款资金,x1~x4 四个投资项目的资金
    x = [pulp.LpVariable(f"x{i}", lowBound=0, cat="Continuous") for i in range(n + 1)]

    objective = r0 * x[0] + pulp.lpSum(
        (data[i]["r"] - data[i]["p"]) * x[i + 1] for i in range(n)
    )
    model += objective

    # 约束的：风险约束(对每个投资项目 i = 1..n)
    # q_i * x_i <= a * m
    # 这里 a 是 risk_limit
    for i in range(n):
        model += data[i]["q"] * x[i + 1] <= risk_limit * M

    # 约束2：资金总量约束
    # x0 + sum_{i=1}^{n} (1+p_i) * x_i = M
    model += x[0] + pulp.lpSum((1 + data[i]["p"]) * x[i + 1] for i in range(n)) == M

    # 求解
    solver = pulp.PULP_CBC_CMD(msg=True)
    model.solve(solver)

    status = pulp.LpStatus[model.status]
    values = [v.varValue if v.varValue is not None else 0.0 for v in x]
    net_return = pulp.value(objective) if status == "Optimal" else 0
    total_risk = sum(data[i]["q"] * values[i + 1] for i in range(n))
    risk_ratio = total_risk / M if M else 0.0
    return status, values, net_return, risk_ratio


status, values, net_return, risk_ratio = solve_for_risk_limit(a)

# 输出结果
print("求解状态:", status)
if status == "Optimal":
    print("\n最优投资方案 (资金分配):")
    print(f"银行存款 s0 : {values[0]:.2f} 元")
    for i in range(n):
        print(f"投资 s{i + 1}    : {values[i + 1]:.2f} 元")
    print(f"\n最大总收益 (净): {net_return:.6f}")
    print(f"组合风险比例(总损失/M): {risk_ratio:.6f}")
    # 验证总资金
    total = values[0] + sum((1 + data[i]["p"]) * values[i + 1] for i in range(n))
    print(f"实际使用总资金: {total:.6f} (应等于 {M})")
else:
    print("未找到最优解，请检查约束或参数 a 是否过小。")


a_values = [i * 0.001 for i in range(50)]  # 风险上限序列
returns = []  # 存储最优收益(Q)

for risk_limit in a_values:
    status, _, net_return, _ = solve_for_risk_limit(risk_limit)
    if status == "Optimal":
        returns.append(net_return)
    else:
        returns.append(0.0)

plt.figure(figsize=(7, 4.5))  # 创建画布并设置大小
plt.plot(a_values, returns, marker="*", linestyle="None")  # 绘制散点图
plt.title("Risk-Return")  # 图标题
plt.xlabel("a")  # x 轴标签
plt.ylabel("Q")  # y 轴标签
plt.grid(True, linestyle="--", alpha=0.4)  # 添加网格线
plt.tight_layout()  # 自动调整边距
plt.show()  # 显示图像
