import traci

traci.start(["sumo", "-c", "cross.sumocfg"])

print("版本：", traci.getVersion())

"""
getMinExpectedNumber > 0 
将仍有车辆作为循环条件
"""
while traci.simulation.getMinExpectedNumber() > 0:  # type: ignore[operator]  # stub 返回类型过宽（int | tuple[Unknown]）
    traci.simulationStep()  # 推进一个仿真步长 (1s)

    t: float = traci.simulation.getTime()  # type: ignore[assignment]  # SUMO 实际返回 float
    if int(t) % 10 != 0:
        continue

    vehs = traci.vehicle.getIDList()
    if not vehs:
        continue

    """
    打印车流
    """
    print(f"\n仿真时刻 {t}s | 在途 {len(vehs)} 辆")

    for v in vehs:
        speed = traci.vehicle.getSpeed(v)
        lane = traci.vehicle.getLaneID(v)
        vcls = traci.vehicle.getVehicleClass(v)
        print(f"{v:<20} 车速={speed:6.2f} m/s 车道={lane:<8} 车型={vcls}")

traci.close()

