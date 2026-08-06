"""
固定配时 vs 自适应配时：交叉口总排队车辆数随时间变化对比

调用方式：分别跑一遍固定配时(s2)和自适应配时(s5)两种仿真，
各收集"累计总排队车辆数-时间"序列，放到同一张图上对比。

deepseek v4 flash 根据 s2 和 s5 生成的。我懒得写了……
"""

import traci
import matplotlib.pyplot as plt
from pathlib import Path

# 图表中文字体：首选思源黑体，次选黑体
plt.rcParams["font.sans-serif"] = ["Source Han Sans SC", "SimHei", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

TLS_ID = "J1"
GREEN = {"NS": "GGGrrrGGGrrr", "EW": "rrrGGGrrrGGG"}

# ---- 固定配时参数（对应 s2 默认信号方案）----
FIX_DUR = 33          # 每相位绿灯时长(s)

# ---- 自适应配时参数（对应 s5）----
ADAPT_GREEN = 30      # 基础绿灯时长(s)
ADAPT_YELLOW = 3      # 黄灯时长(s)
MAX_GREEN = 60        # 最大绿灯续绿上限(s)


def yellow_of(green_state: str) -> str:
    """把当前绿灯相位改黄"""
    return "".join("y" if c == "G" else c for c in green_state)


def get_INLETS() -> dict:
    all_lanes = traci.trafficlight.getControlledLanes(TLS_ID)
    return {
        "北": all_lanes[0:3],
        "东": all_lanes[3:6],
        "南": all_lanes[6:9],
        "西": all_lanes[9:12],
    }


def simulate_adaptive() -> list:
    """
    自适应配时仿真（s5 逻辑），返回 [(t, 累计排队数), ...]
    """
    traci.start(["sumo", "-c", "cross.sumocfg", "--no-step-log"])
    INLETS = get_INLETS()

    cur = "NS"
    hold = ADAPT_GREEN
    phase_time = 0
    is_yellow = False

    history = [(0.0, 0)]
    while traci.simulation.getMinExpectedNumber() > 0:  # type: ignore[operator]
        t: float = traci.simulation.getTime()  # type: ignore[assignment]
        q = {
            name: sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)  # type: ignore[operator]
            for name, lanes in INLETS.items()
        }
        ns, ew = q["南"] + q["北"], q["东"] + q["西"]
        history.append((t, history[-1][1] + ns + ew))

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
                    hold = ADAPT_GREEN
                    continue
                is_yellow = True
                hold = ADAPT_YELLOW
                phase_time = 0
            else:  # 黄灯走完
                is_yellow = False
                hold = ADAPT_GREEN
                phase_time = 0
                cur = "EW" if cur == "NS" else "NS"

    traci.close()
    return history


def simulate_fixed() -> list:
    """
    固定配时仿真（s2 逻辑，配时依据 cross.net.xml 固定方案），
    返回 [(t, 累计排队数), ...]
    """
    traci.start(["sumo", "-c", "cross.sumocfg", "--no-step-log"])
    INLETS = get_INLETS()

    history = [(0.0, 0)]
    while traci.simulation.getMinExpectedNumber() > 0:  # type: ignore[operator]
        t: int = traci.simulation.getTime()  # type: ignore[assignment]
        queue = {
            name: sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)  # type: ignore[operator]
            for name, lanes in INLETS.items()
        }
        q = queue["南"] + queue["北"] + queue["东"] + queue["西"]
        history.append((t, history[-1][1] + q))
        traci.simulationStep()

    traci.close()
    return history


def plot_compare(fixed, adaptive):
    ts_f, qs_f = [h[0] for h in fixed], [q[1] for q in fixed]
    ts_a, qs_a = [h[0] for h in adaptive], [q[1] for q in adaptive]

    plt.figure(figsize=(10, 6))
    plt.plot(ts_f, qs_f, "b-", label="固定配时")
    plt.plot(ts_a, qs_a, "r-", label="自适应配时")
    plt.xlabel("时间(s)")
    plt.ylabel("累计排队车辆数")
    plt.title("固定配时 vs 自适应配时：交叉口总排队车辆数对比")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存图片到脚本同目录
    out_path = Path(__file__).resolve().with_name("s6_compare.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"图片已保存: {out_path}")

    plt.show()


if __name__ == "__main__":
    print("运行固定配时仿真...")
    fixed = simulate_fixed()
    print("运行自适应配时仿真...")
    adaptive = simulate_adaptive()
    plot_compare(fixed, adaptive)
