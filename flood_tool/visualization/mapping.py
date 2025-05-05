import folium
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

all = ['plot_circle',
       'plot_multi_circles']


def plot_circle(lat, lon, radius, map=None, **kwargs):
    '''
    Plot a circle on a map (creating a new folium map instance if necessary).

    Parameters
    ----------

    lat: float
        latitude of circle to plot (degrees)
    lon: float
        longitude of circle to plot (degrees)
    radius: float
        radius of circle to plot (m)
    map: folium.Map
        existing map object

    Returns
    -------

    Folium map object

    Examples
    --------

    >>> import folium
    >>> plot_circle(52.79, -2.95, 1e3, map=None) # doctest: +SKIP
    '''

    if not map:
        map = folium.Map(location=[lat, lon], control_scale=True)
        

    folium.Circle(
        location=[lat, lon],
        radius=radius,
        fill=True,
        fillOpacity=0.6,
        **kwargs,
    ).add_to(map)

    return map

def plot_multi_circles(df, lats, lons, label, map=None, **kwargs):
    '''
    Plot multiple circles on a map (creating a new folium map instance if necessary).

    Parameters
    ----------
    df: pandas.DataFrame

    lats: name of the column in the dataframe
        latitudes of circles to plot (degrees)
    lons: name of the column in the dataframe
        longitudes of circles to plot (degrees)
    
    label: list of strings put different colors on the circles

    map: folium.Map
        existing map object

    Returns
    -------

    Folium map object

    Examples
    --------
    >>> import folium
    >>> import pandas as pd
    >>> df = pd.DataFrame({'latitude': [52.79, 52.78], 'longitude': [-2.95, -2.94], 'riskLabel': [1, 2]})
    >>> plot_multi_circles(df, 'latitude', 'longitude', 'riskLabel', map=None) # doctest: +SKIP
 

    '''

    if not map:
        map = folium.Map(location=[df[lats][0], df[lons][0]], control_scale=True)

    norm = mcolors.Normalize(vmin=df[label].min(), vmax=df[label].max())
    cmap = plt.cm.coolwarm  # 
    df = df.sort_values(by=label)

    for _, row in df.iterrows():
        color = mcolors.to_hex(cmap(norm(row[label]))) 
        folium.Circle(
            location=[row[lats], row[lons]],
            radius=3,  
            color=color,  
            fill=True,  
            fill_color=color,  
            fill_opacity=0.3  
        ).add_to(map)


    return map

def update_map(labelled_postcodes,R_min_filter, R_max_filter):
    m = folium.Map(location=[labelled_postcodes["latitude"].mean(), labelled_postcodes["longitude"].mean()], zoom_start=6)
    norm = mcolors.Normalize(vmin=labelled_postcodes['riskLabel'].min(), vmax=labelled_postcodes['riskLabel'].max())
    cmap = plt.cm.coolwarm
    for idx, row in labelled_postcodes.iterrows():
        if R_min_filter <= row['riskLabel'] <= R_max_filter:
            color = mcolors.to_hex(cmap(norm(row['riskLabel']))) 
            popup_text = f"Postcode: {row['postcode']}, Risk-value: {row['riskLabel']},localAuthority: {row['localAuthority']},medianPrice: {row['medianPrice']}"
            folium.Circle(
                location=(row["latitude"], row["longitude"]),
                radius=5,
                fill_opacity=0.3, 
                color=color,
                fill=True,
                fill_color=color,
                popup=folium.Popup(popup_text, parse_html=True)
            ).add_to(m)
    

    return m


