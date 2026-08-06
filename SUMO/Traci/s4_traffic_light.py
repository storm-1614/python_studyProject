"""
绿 33 秒，黄 2 秒
实现 Traci 信号灯相位切换控制
代码用状态机模型进行控制
"""

import traci

traci.start(["sumo-gui", "-c", "cross.sumocfg"])

TLS_ID = "J1"
NS_GREEN = "GGGrrrGGGrrr"
NS_YELLOW = "yyyrrryyyrrr"
EW_GREEN = "rrrGGGrrrGGG"
EW_YELLOW = "rrryyyrrryyy"
PHASES = [(NS_GREEN, 33), (NS_YELLOW, 2), (EW_GREEN, 33), (EW_YELLOW, 2)]
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
