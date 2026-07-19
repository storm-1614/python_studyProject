#! env /bin/python

import requests
import datetime
import pandas
import os
import sys

from security.account import *  # 导入账户密码


xlsx_data = pandas.read_excel(
    "./res/China_SURF_Station.xlsx",
    header=None,
    names=[
        "code",
        "provide",
        "stationName",
        "type",
        "longitude",
        "latitude",
        "altitude1",
        "altitude2",
    ],  # pyright: ignore[reportArgumentType]
    skiprows=1,
)

if len(sys.argv) < 2:
    station_name = "同安"
else:
    station_name = sys.argv[1]

station_code = xlsx_data[xlsx_data["stationName"] == station_name]["code"].values[0]  # pyright: ignore[reportAttributeAccessIssue]


def get_datetime():
    datetime_now = datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=20)
    if (datetime_now.hour % 3) != 0:
        datetime_now = datetime_now - datetime.timedelta(hours=(datetime_now.hour % 3))
    return datetime_now.strftime("%Y%m%d%H")


dataCode = get_datetime()

dataCode_begin = dataCode + "0000"
dataCode_end = dataCode + "0000"

base = "http://api.data.cma.cn:8090/api"
params = {
    "userId": f"{account}",
    "pwd": f"{password}",
    "dataFormat": "json",
    "interfaceId": "getSurfEleByTimeRangeAndStaID",
    "dataCode": "SURF_CHN_MUL_HOR_3H",
    "timeRange": f"[{dataCode_begin},{dataCode_end}]",
    "staIDs": f"{station_code}",
    "elements": "Station_Id_C,Year,Mon,Day,Hour,TEM,PRS,PRS_Sea,RHU,PRE_3h,WIN_D_Avg_2mi,WIN_S_Avg_2mi,WEP_Now,VIS",
}

res = requests.get(base, params=params, timeout=10)
res.raise_for_status()
data = res.json().get("DS")[0]


class station_info:
    press: float
    press_sea: float
    temp: float
    rain_3h: float
    relative_humidity: float
    wind_direction_2min: float
    wind_speed_2min: float
    vis: float
    weather: str

    def __init__(self, json_data) -> None:
        self.press = float(json_data["PRS"])
        self.press_sea = float(json_data["PRS_Sea"])
        self.temp = json_data["TEM"]
        self.rain_3h = json_data["PRE_3h"]
        self.wind_direction_2min = json_data["WIN_D_Avg_2mi"]
        self.wind_speed_2min = json_data["WIN_S_Avg_2mi"]
        wep_now_value = float(json_data["WEP_Now"])
        self.relative_humidity = float(json_data["RHU"])
        self.vis = float(json_data["VIS"])
        self.weather_now(wep_now_value)

    def weather_now(self, value):
        """
        判断现在天气观测值。接收一个整数，目前需要将其全部枚举出来。
        """
        match value:
            case 0.0:
                self.weather = "未观测或观测不到云的发展"
            case 1.0:
                self.weather = "从总体上看，云在消散或未发展起来"
            case 2.0:
                self.weather = "总的看来天空状态无变化"
            case 3.0:
                self.weather = "从总体上看，云在形成或发展"
            case 4.0:
                self.weather = "烟雾使能见度降低。如草原或森林火灾，工业排烟或火山灰"
            case 5.0:
                self.weather = "霾"
            case 6.0:
                self.weather = "在空气中悬浮大范围的尘土，这些尘土不是由观测时测站或附近的风吹起的。"
            case 7.0:
                self.weather = "观测时在测站或附近有风吹起的尘或沙，但无发展成熟的尘旋或沙旋，而且看不到尘暴或沙暴，或海洋站和沿海测站出现高吹飞沫"
            case 8.0:
                self.weather = "观测时或前1小时内在测站附近看到发展起来的尘旋或沙旋，但无尘暴或沙暴。"
            case 9.0:
                self.weather = "观测时看得见尘暴或沙暴，或观测前1小时内在测站出现尘暴或沙暴"
            case 10.0:
                self.weather = "薄雾"
            case 11.0:
                self.weather = "碎片状雾"
            case 12.0:
                self.weather = "或多或少连续的雾"
            case _:
                self.weather = f"未知代码：{int(value)}"


info = station_info(data)
hour = int(data["Hour"]) + 8

if hour > 24:
    hour -= 24

print(
    data["Year"],
    "年",
    data["Mon"],
    "月",
    data["Day"],
    "日",
    hour,
    "时",
    sep="",
)

print(f"{station_name}国家气象站({station_code})：")
print("气温：", info.temp, sep="")
print("气压：", info.press, sep="")
print("海平面气压：", info.press_sea, sep="")
print("相对湿度：", info.relative_humidity, sep="")
print("2分钟平均风速：", info.wind_speed_2min, " m/s", sep="")
print("2分钟平均风向：", info.wind_direction_2min, sep="")
print("3小时降水量：", info.rain_3h, sep="")
print("水平能见度：", info.vis, " m", sep="")
print(info.weather)
