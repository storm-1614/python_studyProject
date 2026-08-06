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
history_queue = [(0.0, 0)]
while traci.simulation.getMinExpectedNumber() > 0:  # type: ignore[operator]
    traci.simulationStep()

    t: int = traci.simulation.getTime()  # type: ignore[assignment]

    queue = {
        name: sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)  # type: ignore[operator]
        for name, lanes in INLETS.items()
    }
    q = queue["南"] + queue["北"] + queue["东"] + queue["西"]

    history.append((t, *queue.values()))
    history_queue.append((t, history_queue[-1][1] + q))

traci.close()
ts = [h[0] for h in history_queue]
qs = [q[1] for q in history_queue]

plt.figure(figsize=(10, 5))
plt.plot(ts, qs, "b-")
plt.xlabel("时间(s)")
plt.ylabel("排队车辆数")
plt.title("交叉口总排队车辆数随时间变化")

plt.grid(True)
plt.show()
