import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import cos, pi, exp
from geopy.distance import distance

path_no2_data = "NO2_2020_01.xls"
path_traffic_data = "traffic.csv"

stations = ["München/Allach", "München/Johanneskirchen", "München/Landshuter Allee", "München/Lothstraße", "München/Stachus"]
traffic_sensor_labels = {"LHA": "Landshuter Allee", "LOT": "Lothstraße", "STA": "Stachus"}
plot_start = "12.01.2020 24:00"
plot_end = "14.01.2020 24:00"

locations = {   # (lat, lon)
    "München/Allach": (48.18165, 11.46444),
    "München/Johanneskirchen": (48.17319, 11.64804),
    "München/Landshuter Allee": (48.14955, 11.53653),
    "München/Lothstraße": (48.15455, 11.55466),
    "München/Stachus": (48.13732, 11.56481)
}
r_earth = 6378000  # earth radius


def plot_48h(no2_day, traffic):
    assert (len(no2_day) == 2 * len(traffic))

    fig, ax1 = plt.subplots()
    for station in stations:
        no2_array = no2_day[station].to_numpy()
        ax1.plot(no2_array, label=station.split('/')[1], linestyle=':')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    for sensor in traffic:
        traffic_array = traffic[sensor].to_numpy()
        traffic_array_double = np.concatenate((traffic_array, traffic_array))
        ax2.plot(traffic_array_double, label=traffic_sensor_labels[sensor])

    plt.figure()
    x_ticks = [t.split(" ")[1] for t in no2_day["Zeitpunkt"]]
    plt.xticks(ticks=np.arange(len(x_ticks)), labels=x_ticks)
    ax1.tick_params(axis='x', rotation=90)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.set_ylabel("NO2 [µg/m3] 1h-MW")
    ax2.set_ylabel("Traffic")


def plot_map(measurements):
    print("Measurements")
    for loc in locations:
        print(loc, measurements[loc])

    grid_size = 1000  # grid cells are quadratic, length of a side in meters
    grid_dim = 20  # number of grid cells in both dimensions
    plot_points = []
    concentration_map = np.zeros((grid_dim, grid_dim))

    # reference point is the most west and north point from Landshuter Allee
    # this reference point is the center of the upper left cell (0,0) of grid
    ref_point_adjust = (grid_dim/2 - 0.5) * grid_size
    reference_point = project_point(locations["München/Landshuter Allee"], -ref_point_adjust, ref_point_adjust)
    for ix in range(0, grid_dim):
        for iy in range(0, grid_dim):
            # looping left to right and top to bottom
            new_point_coords = project_point(reference_point, ix * grid_size, (-1) * iy * grid_size)
            concentration_map[iy, ix] = concentration_map_value(new_point_coords, measurements)
            plot_points.append(new_point_coords)

    plt.figure()
    plt.imshow(concentration_map)
    plt.colorbar()
    plt.axis('off')

    plt.figure()
    lat = [p[0] for p in plot_points]
    lon = [p[1] for p in plot_points]
    plt.scatter(lon, lat)

    lat_sensors = [p[0] for p in locations.values()]
    lon_sensors = [p[1] for p in locations.values()]
    plt.scatter(lon_sensors, lat_sensors, c='r')

    plt.axes().set_aspect('equal')
    plt.xlabel('lon')
    plt.ylabel('lat')


def concentration_map_value(coords, measurements):
    """Interpolation function"""
    for sensor_str, sensor_coords in locations.items():
        if distance(coords, sensor_coords).meters < 700:
            return measurements[sensor_str]

    num, den = 0, 0
    for sensor_str, sensor_coords in locations.items():
        w = 1/distance(coords, sensor_coords).meters
        num += w * measurements[sensor_str]
        den += w
    return num/den


def project_point(ref_point, dx, dy):
    """ Return coordinates that are (dx, dy) away from a reference point
    :param ref_point: (lat, lon) of reference point
    :param dx: Projection in x direction in meters
    :param dy: Projection in y direction in meters
    :return: (lat, lon) of new point
    """
    ref_lat, ref_lon = ref_point
    new_lat = ref_lat + (dy / r_earth) * (180 / pi)
    new_lon = ref_lon + (dx / r_earth) * (180 / pi) / cos(ref_lat * pi / 180)
    return new_lat, new_lon


if __name__ == "__main__":

    xls = pd.ExcelFile(path_no2_data)
    no2_data = xls.parse(skiprows=[0, 1])

    day_start_idx = no2_data.index[no2_data["Zeitpunkt"] == plot_start].values[0]
    day_end_idx = no2_data.index[no2_data["Zeitpunkt"] == plot_end].values[0]
    no2_data_day = no2_data.iloc[day_start_idx:day_end_idx]
    no2_data_hour = no2_data.iloc[100]

    traffic_data = pd.read_csv(path_traffic_data).T
    traffic_data.columns = traffic_data.iloc[1]
    traffic_data = traffic_data.drop(['Sensor ID', 'Place', 'ALL DAY'])

    plot_48h(no2_data_day, traffic_data)
    plot_map(no2_data_hour)
    plt.show()
