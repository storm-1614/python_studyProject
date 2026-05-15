import pulp

# $$
# \begin{align}
# \min \quad & z = 2x_{1} + 3x_{2} + x_{3},\tag{1}\\
# \text{s.t.} \quad &
# \begin{cases}
# x_{1} + 4 x_{2} + 2x_{3} \geq8,\\
# 3x_{1}+2x_{2} \geq6,\\
# x_{1},x_{2},x_{3} \geq 0\\
# \end{cases}\tag{2}
# \end{align}
# $$


MyProbLP = pulp.LpProblem("LPProb1", sense=pulp.LpMinimize)

x1 = pulp.LpVariable("x1", lowBound=0, cat="Continuous")
x2 = pulp.LpVariable("x2", lowBound=0, cat="Continuous")
x3 = pulp.LpVariable("x3", lowBound=0, cat="Continuous")

MyProbLP += 2 * x1 + 3 * x2 + x3

MyProbLP += x1 + 4 * x2 + 2 * x3 >= 8
MyProbLP += 3 * x1 + 2 * x2 >= 6

MyProbLP.solve()
print("Status:", pulp.LpStatus[MyProbLP.status])
for v in MyProbLP.variables():
    print(v.name, "=", v.varValue)
print("F(x) = ", pulp.value(MyProbLP.objective))
