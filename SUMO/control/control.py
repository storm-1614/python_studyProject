import traci

traci.start(["sumo", "-c", "my.sumocfg"])

NW_GREEN = "GGGrrrGGGrrr"
ES_GREEN = "rrrGGGrrrGGG"


def yellow_of(state: str) -> str:
    return "".join(["y" if c == "G" else c for c in state])


TLS_ID = "J1"
NW_YELLOW = yellow_of(NW_GREEN)
ES_YELLOW = yellow_of(ES_GREEN)

GREEN_TIME = 40
YELLOW_TIME = 3
PHASES = [
    (NW_GREEN, GREEN_TIME),
    (NW_YELLOW, YELLOW_TIME),
    (ES_GREEN, GREEN_TIME),
    (ES_YELLOW, YELLOW_TIME),
]


traci.trafficlight.setRedYellowGreenState(TLS_ID, yellow_of(NW_GREEN))
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
