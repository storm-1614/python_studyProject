#! env /bin/python3

import requests
import os
import logging
import pandas
import sys
import ui
from PyQt6.QtWidgets import QApplication, QWidget

citydata_path = "./AMap_adcode_citycode_20210406.xlsx"


def load_citycode_xlsx(path: str):
    """
    从高德提供的城市代码表格中提取数据
    """
    try:
        xlsx_data = pandas.read_excel(
            path,
            header=None,
            names=["city", "adcode", "citycode"],  # pyright: ignore[reportArgumentType]
            skiprows=2,
        )
    except FileNotFoundError:
        print("AMap_adcode_citycode_20210406.xlsx 不存在")
        exit(1)
    return xlsx_data


def init_log():
    """
    初始化 logging 模块
    """
    global logger
    logger = logging.getLogger("Logging")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_hander = logging.StreamHandler()
    console_hander.setLevel(logging.DEBUG)
    console_hander.setFormatter(formatter)

    logger.addHandler(console_hander)


class weather_info:
    """
    天气信息类
    主要用于命令行下，但是实际在图形界面每次获取天气都实例化该类
    """
    # FIX: find_city 方法只适用于命令行，而图形界面没有调用
    api_url = "https://restapi.amap.com/v3/weather/weatherInfo"
    url_parms: dict
    raw_data: dict
    now_weather_data: dict
    forecast_weather_data: dict

    def __init__(self, city_code=None) -> None:
        if city_code == None:
            city_code = self.find_city()
        self.url_parms = {"key": key, "city": city_code, "extensions":"base"}
        self.raw_data = self.get_data()
        self.now_weather_data = self.raw_data["lives"][0]
        #self.forecast_weather_data = self.raw_data["forecast"][0]

    def find_city(self) -> int:
        """
        终端下查找城市
        """
        while True:
            try:
                city_name = input("请输入城市：")
                code = city_data[city_data["city"] == city_name]["adcode"].values[0]  # pyright: ignore[reportAttributeAccessIssue]
                break
            except IndexError:
                print("城市错误")
                pass
        return code

    def get_data(self) -> dict:
        """
        获得高德开发者平台的天气数据
        """
        response = requests.get(url=self.api_url, params=self.url_parms)
        data = response.json()

        if data.get("status") != "1":
            logger.error("Data status error!")

        return data

    def print_data(self):
        """
        打印信息
        """
        print(self.data_str())

    def data_str(self):
        """
        输出的内容
        """
        result_str = (
            f"{self.now_weather_data['province']},{self.now_weather_data['city']}\n"
            f"实况：\n"
            f"天气：\t\t{self.now_weather_data['weather']}\n"
            f"气温：\t\t{self.now_weather_data['temperature']}°C\n"
            f"风向：\t\t{self.now_weather_data['winddirection']}\n"
            f"风力：\t\t{self.now_weather_data['windpower']}级\n"
            f"空气湿度：\t{self.now_weather_data['humidity']}%"
        )
        return result_str


class GuiPanel(QWidget, ui.Ui_Form):
    """
    QT5 窗口类
    """
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("高德开发平台天气API")
        self.setupUi(self)
        self.initUI()

    def initUI(self):
        """
        定义点击和回车事件
        """
        self.city_edit.returnPressed.connect(self.get_info)
        self.search_button.clicked.connect(self.get_info)

    def get_info(self):
        """
        获得信息，实例化一个 weather_info 类
        """
        self.info_text.clear()
        cityName = self.city_edit.text()
        try:
            code = city_data[city_data["city"] == cityName]["adcode"].values[0]  # pyright: ignore[reportAttributeAccessIssue]
            weather = weather_info(code)
            info = weather.data_str()
        except IndexError:
            info = "城市错误"
        self.info_text.append(info)


def cli():
    """
    调用命令行
    """
    w = weather_info()
    w.print_data()


def gui():
    """
    调用图形窗口
    """
    app = QApplication(sys.argv)
    g = GuiPanel()
    g.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    global city_data
    global key
    init_log()
    city_data = load_citycode_xlsx(citydata_path)
    print("""
    API: 高德开发平台 API
    设置系统变量 AMAP_API 为 API
          """)
    try:
        key = os.environ["AMAP_API"]
    except KeyError:
        logger.warning("AMAP_API 不存在，使用默认 API")
        key = "d854936d46fde377d8ffda03c9e4d906"
        

    if len(sys.argv) < 2:
        gui()
    elif sys.argv[1] == "-g":
        gui()
    elif sys.argv[1] == "-c":
        cli()
