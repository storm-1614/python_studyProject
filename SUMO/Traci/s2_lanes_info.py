"""
获得各向路口排队信息
"""

import traci
import matplotlib.pyplot as plt

# 图表中文字体：首选思源黑体，次选黑体
plt.rcParams["font.sans-serif"] = ["Source Han Sans SC", "SimHei", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

traci.start(["sumo", "-c", "cross.sumocfg", "--no-step-log"])

all_lanes: tuple = traci.trafficlight.getControlledLanes("J1")

INLETS = {
    "北": all_lanes[0:3],
    "东": all_lanes[3:6],
    "南": all_lanes[6:9],
    "西": all_lanes[9:12],
}
history = []
while traci.simulation.getMinExpectedNumber() > 0:  # type: ignore[operator]
    traci.simulationStep()

    t: int = traci.simulation.getTime()  # type: ignore[assignment]

    queue = {
        name: sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)  # type: ignore[operator]
        for name, lanes in INLETS.items()
    }

    history.append((t, *queue.values()))

traci.close()

ts = [h[0] for h in history]
qs = list(zip(*[h[1:] for h in history]))

names = ["北N", "东E", "南S", "西W"]

for i, q in enumerate(qs):
    plt.plot(ts, q, label=names[i])

plt.legend()
plt.xlabel("时间(s)")
plt.ylabel("排队车数")
plt.grid()

plt.show()
