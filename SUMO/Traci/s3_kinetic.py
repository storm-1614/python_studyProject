"""
计算车辆动能
"""

import traci

traci.start(["sumo", "-c", "cross.sumocfg"])

VEHICLE_MASS = {"passenger": 1500.0, "truck": 30000}


def kinetic_energy(speed_mps: float, vclass: str) -> float:
    mass = VEHICLE_MASS.get(vclass, VEHICLE_MASS["passenger"])
    return 0.5 * mass * speed_mps**2


while traci.simulation.getMinExpectedNumber() > 0:  # type: ignore[assignment]
    traci.simulationStep()

    t: float = traci.simulation.getTime()  # type: ignore[assignment]
    if int(t) % 10 != 0:
        continue

    vehs = traci.vehicle.getIDList()

    if not vehs:
        continue

    print(f"\n仿真时刻 {t}s | 在途 {len(vehs)} 辆")

    for v in vehs:
        speed = traci.vehicle.getSpeed(v)
        lane = traci.vehicle.getLaneID(v)
        vcls = traci.vehicle.getVehicleClass(v)
        ek = kinetic_energy(speed, vcls)  # type:ignore[assignment]
        print(
            f"{v:<20} 车速={speed:6.2f} m/s 车道={lane:<8} 动能:{ek:6.2f} 车型={vcls}"
        )
