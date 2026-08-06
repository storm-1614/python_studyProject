import traci
import matplotlib.pyplot as plt

# 图表中文字体：首选思源黑体，次选黑体
plt.rcParams["font.sans-serif"] = ["Source Han Sans SC", "SimHei", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

traci.start(["sumo-gui", "-c", "cross.sumocfg"])

TLS_ID = "J1"
GREEN = {"NS": "GGGrrrGGGrrr", "EW": "rrrGGGrrrGGG"}


def yellow_of(green_state):
    """
    把当前绿灯相位改黄
    """
    tl = ""
    for c in green_state:
        if c == "G":
            tl += "y"
        else:
            tl += c
    return tl


GREEN_DUR = 30
YELLOW_DUR = 3

MAX_GREEN = 60

cur = "NS"
hold = GREEN_DUR
phase_time = 0
is_yellow = False

all_lanes = traci.trafficlight.getControlledLanes(TLS_ID)
INLETS = {
    "北": all_lanes[0:3],
    "东": all_lanes[3:6],
    "南": all_lanes[6:9],
    "西": all_lanes[9:12],
}

history_queue = [(0.0, 0)]
while traci.simulation.getMinExpectedNumber() > 0:  # type: ignore[assignment]
    t: float = traci.simulation.getTime()  # type: ignore[assignment]
    # 读当前两向排队情况
    q = {
        name: sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)  # type: ignore[assignment]
        for name, lanes in INLETS.items()
    }

    ns, ew = q["南"] + q["北"], q["东"] + q["西"]
    history_queue.append((t, history_queue[-1][1] + ns + ew))

    # 设置当前相位灯
    state = yellow_of(GREEN[cur]) if is_yellow else GREEN[cur]
    traci.trafficlight.setRedYellowGreenState(TLS_ID, state)

    traci.simulationStep()

    hold -= 1
    phase_time += 1

    if hold <= 0:
        if not is_yellow:  # 绿灯完进黄
            if (
                cur == "NS" and ns + 2 > ew or cur == "EW" and ew + 2 > ns
            ) and phase_time < MAX_GREEN:
                print("续绿")
                hold = GREEN_DUR
                continue

            is_yellow = True
            hold = YELLOW_DUR
            phase_time = 0
        else:  # 黄灯走完
            is_yellow = False
            hold = GREEN_DUR
            phase_time = 0
            cur = "EW" if cur == "NS" else "NS"

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
