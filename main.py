import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import cos, pi
from geopy.distance import distance

# libries plotly code, don't forget to pip
from plotly.offline import plot
import plotly.express as px
import plotly.graph_objects as go

path_no2_data = "NO2_2020_01.xls"
path_traffic_data = "traffic.csv"

stations = ["München/Allach", "München/Johanneskirchen", "München/Landshuter Allee", "München/Lothstraße", "München/Stachus"]
traffic_sensor_labels = {"LHA": "Landshuter Allee", "LOT": "Lothstraße", "STA": "Stachus"}
locations = {   # (lat, lon)
    "München/Allach": (48.18165, 11.46444),
    "München/Johanneskirchen": (48.17319, 11.64804),
    "München/Landshuter Allee": (48.14955, 11.53653),
    "München/Lothstraße": (48.15455, 11.55466),
    "München/Stachus": (48.13732, 11.56481)
}

location_colors = {
    "Allach": 'lightgrey',
    "Johanneskirchen": 'slategrey',
    "Landshuter Allee": 'darkblue',
    "Lothstraße": 'chocolate',
    "Stachus": 'seagreen',
}
plot_start = "12.01.2020 24:00"
plot_end = "14.01.2020 24:00"
plot_title = "NO2 and Traffic in Munich (12-13.01.2020)"
map_center = (48.137079, 11.576006)  # marienplatz

r_earth = 6378000  # earth radius [m]
c_mass_density_to_mixing_ratio = 0.52293  # µg/m3 --> ppb


def plot_48h(no2_day, traffic):
    assert (len(no2_day) == 2 * len(traffic))

    f, (ax1, ax2) = plt.subplots(2, 1)
    for station in stations:
        no2_array = no2_day[station].to_numpy()
        location_name = station.split('/')[1]
        ax1.plot(no2_array, label=location_name, color=location_colors[location_name])

    for sensor in traffic:
        traffic_array = traffic[sensor].to_numpy()
        traffic_array_double = np.concatenate((traffic_array, traffic_array))
        location_name = traffic_sensor_labels[sensor]
        ax2.plot(traffic_array_double, label=location_name, color=location_colors[location_name], linestyle=":")

    x_ticks_labels_all = [t.split(" ")[1] for t in no2_day["Zeitpunkt"]]
    x_ticks_labels = []
    for l in x_ticks_labels_all:
        hour = int(l.split(":")[0])
        label = l if (hour % 3 == 0) else ''
        x_ticks_labels.append(label)
    ax1.set_xticks(np.arange(len(x_ticks_labels)))
    ax2.set_xticks(np.arange(len(x_ticks_labels)))
    ax1.set_xticklabels('' * len(x_ticks_labels))
    ax2.set_xticklabels(x_ticks_labels, rotation='vertical')

    plt.suptitle(plot_title)
    ax1.set_title("NO2")
    ax2.set_title("Traffic")
    ax1.set_ylabel("NO2 [ppb]")
    ax2.set_ylabel("Cars / h (?)")
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    ax1.grid(True)
    ax2.grid(True)


def generate_interpolation_df(measurements):
    """Creates and returns dataframe with interpolated concentration map"""
    print("Measurements")
    for loc in locations:
        print(loc, measurements[loc])

    grid_size = 1000  # grid cells are quadratic, length of a side in meters
    grid_dim = 20  # number of grid cells in both dimensions

    # reference point is the most west and north point from Landshuter Allee
    # this reference point is the center of the upper left cell (0,0) of grid
    ref_point_adjust = (grid_dim/2 - 0.5) * grid_size
    reference_point = project_point(map_center, -ref_point_adjust, ref_point_adjust)
    lat_arr, lon_arr, concentration_arr = [], [], []
    for ix in range(0, grid_dim):
        for iy in range(0, grid_dim):
            # looping left to right and top to bottom
            new_point_coords = project_point(reference_point, ix * grid_size, (-1) * iy * grid_size)
            new_point_cocentration = concentration_map_value(new_point_coords, measurements)
            lat_arr.append(new_point_coords[0])
            lon_arr.append(new_point_coords[1])
            concentration_arr.append(new_point_cocentration)

    return pd.DataFrame(list(zip(lat_arr, lon_arr, concentration_arr)), columns=["lat", "lon", "concentration"])


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


def plotly_map_px(muc):
    # plot map with plotly,express (less versatile than plotly.graph_objects)
    muc["size"] = 1.5
    muc["size"][0]=3
    fig = px.scatter_mapbox(muc, lat="lat", lon="lon", color = "concentration[uq/m3]", zoom=10, width=900,height=800,opacity = 0.7, size="size", color_continuous_scale="jet")
    #color_continuous_scale='Inferno'color_continuous_scale='jet'
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":100,"t":10,"l":10,"b":100})
    # go.Figure(fig)
    # fig.update_traces(marker=dict(
    #                         size=12,
    #                         line=dict(width=2,
    #                                   color='DarkSlateGrey'),
    #                         marker_symbol = 'square',
    #                               ),
                          
    #               selector=dict(mode='markers&text'))
    fig.show()
    plot(fig)


def plotly_map_go(concentration_df):
    locations_df = pd.DataFrame.from_dict(locations, orient='index', columns=["lat", "lon"]).reset_index()
    custom_colorscale = [(0, "green"), (0.25, "lightgreen"), (0.35, "yellow"), (0.45, "red"), (1, "maroon")]
    
    fig = go.Figure()

    fig.add_trace(go.Scattermapbox(
            lat=concentration_df.lat,
            lon=concentration_df.lon,
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=25,
                color=concentration_df['concentration'],
                opacity=0.6,
                colorscale=custom_colorscale,
                showscale=True,
            ),
        ))
    # Unfortunatley not possible to change marker symbol, due to a bug of scattermapbox
    fig.add_trace(go.Scattermapbox(
            lat=locations_df.lat,
            lon=locations_df.lon,
            mode='markers',
            marker=dict(size=15, color='blue'),
        ))

    fig.update_layout(
        title="NO Concentration in Munich [ppb]",
        autosize=True,
        width=1200, height=1000,
        hovermode='closest',
        showlegend=False,
        mapbox=dict(
            accesstoken="pk.eyJ1IjoiZ2EyNmthbiIsImEiOiJja2pqdWsyOHIxbjh1MnlsbzFmNWJmd24wIn0.tgTabMFNJsFFt9U2wEvLdw",
            bearing=0,
            center=go.layout.mapbox.Center(
                    lat=map_center[0],
                    lon=map_center[1],),
            pitch=0,
            zoom=10.8,
            style="outdoors"),
        )
    plot(fig)


if __name__ == "__main__":
    # read excel file
    xls = pd.ExcelFile(path_no2_data)
    no2_data = xls.parse(skiprows=[0, 1])
    no2_data = no2_data[["Zeitpunkt"] + stations]

    # convert to ppb
    no2_data[stations] = no2_data[stations].apply(pd.to_numeric, errors='coerce', downcast='float')
    no2_data[stations] = no2_data[stations].multiply(c_mass_density_to_mixing_ratio)

    # reduce to day
    day_start_idx = no2_data.index[no2_data["Zeitpunkt"] == plot_start].values[0]
    day_end_idx = no2_data.index[no2_data["Zeitpunkt"] == plot_end].values[0]
    no2_data_day = no2_data.iloc[day_start_idx:day_end_idx]
    no2_data_day_mean = no2_data_day.mean()

    traffic_data = pd.read_csv(path_traffic_data).T
    traffic_data.columns = traffic_data.iloc[1]
    traffic_data = traffic_data.drop(['Sensor ID', 'Place', 'ALL DAY'])
    plot_48h(no2_data_day, traffic_data)
    plt.show()

    concentration_map_df = generate_interpolation_df(no2_data_day_mean)
    plotly_map_go(concentration_map_df)
