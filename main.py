import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path_no2_data = "NO2_2020_01.xls"
path_traffic_data = "traffic.csv"

stations = ["München/Allach", "München/Johanneskirchen", "München/Landshuter Allee", "München/Lothstraße", "München/Stachus"]
traffic_sensor_labels = {"LHA": "Landshuter Allee", "LOT": "Lothstraße", "STA": "Stachus"}
plot_start = "12.01.2020 24:00"
plot_end = "14.01.2020 24:00"

xls = pd.ExcelFile(path_no2_data)
no2_data = xls.parse(skiprows=[0, 1])

day_start_idx = no2_data.index[no2_data["Zeitpunkt"] == plot_start].values[0]
day_end_idx = no2_data.index[no2_data["Zeitpunkt"] == plot_end].values[0]
no2_data_day = no2_data.iloc[day_start_idx:day_end_idx]

traffic_data = pd.read_csv(path_traffic_data).T
traffic_data.columns = traffic_data.iloc[1]
traffic_data = traffic_data.drop(['Sensor ID', 'Place', 'ALL DAY'])

assert(len(no2_data_day) == 2 * len(traffic_data))

fig, ax1 = plt.subplots()

for station in stations:
    no2_array = no2_data_day[station].to_numpy()
    ax1.plot(no2_array, label=station.split('/')[1], linestyle=':')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

for sensor in traffic_data:
    traffic_array = traffic_data[sensor].to_numpy()
    traffic_array_double = np.concatenate((traffic_array, traffic_array))
    ax2.plot(traffic_array_double, label=traffic_sensor_labels[sensor])

x_ticks = [t.split(" ")[1] for t in no2_data_day["Zeitpunkt"]]
plt.xticks(ticks=np.arange(len(x_ticks)), labels=x_ticks)
ax1.tick_params(axis='x', rotation=90)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
ax1.set_ylabel("NO2 [µg/m3] 1h-MW")
ax2.set_ylabel("Traffic")
plt.show()
