"""
绿 33 秒，黄 2 秒
实现 Traci 信号灯相位切换控制
代码用状态机模型进行控制
"""

import traci

traci.start(["sumo", "-c", "cross.sumocfg", "--seed", "42"])

TLS_ID = "J1"
NS_GREEN = "GGGrrrGGGrrr"
NS_YELLOW = "yyyrrryyyrrr"
EW_GREEN = "rrrGGGrrrGGG"
EW_YELLOW = "rrryyyrrryyy"
PHASES = [(NS_GREEN, 30), (NS_YELLOW, 2), (EW_GREEN, 30), (EW_YELLOW, 2)]
hold = PHASES[0][1]
idx = 0
while traci.simulation.getMinExpectedNumber() > 0:  # type: ignore[operator]
    state, dur = PHASES[idx]
    traci.trafficlight.setRedYellowGreenState(TLS_ID, state)
    traci.simulationStep()
    hold -= 1

    if hold <= 0:
        idx = (idx + 1) % len(PHASES)
        hold = PHASES[idx][1]

traci.close()

import xml.etree.ElementTree as ET

tree = ET.parse("tripinfo.xml")
trips = list(tree.getroot().iter("tripinfo"))

if trips:
    n = len(trips)
    loss = sum(float(t.get("timeLoss", 0)) for t in trips)
    waitTime = sum(float(t.get("waitingTime", 0)) for t in trips)
    wait = sum(int(t.get("waitingCount", 0)) for t in trips)

    print(f"车辆总数(吞吐) : {n} 辆")
    print(f"平均延误 timeLoss : {loss / n:.2f} 秒")
    print(f"平均静止等待      : {waitTime / n:.2f} 秒")
    print(f"总停车次数        : {wait} 次")
    print(f"平均停车次数/辆   : {wait / n:.2f} 次")
