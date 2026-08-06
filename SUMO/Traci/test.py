import traci

traci.start(["sumo-gui", "-c", "cross.sumocfg", "--start"])

print(traci.getVersion())


# 跑一小段仿真,验证能取到真实数据
traci.simulationStep(200)  # 推进 200 秒
veh_ids = traci.vehicle.getIDList()
print(f"仿真推进后,当前路上车辆数: {len(veh_ids)}")
if veh_ids:
    print("车辆 ID:", veh_ids)

traci.close()
print("仿真结束")
