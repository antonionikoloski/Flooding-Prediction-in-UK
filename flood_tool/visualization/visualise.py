import folium
import sys
sys.path.append('..')
import flood_tool as ft
import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib
import math
import numpy as np
from io import BytesIO
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap

__all__ = ['processing_day',
           'typical_wet_join',
           'postcode_nearest_station',
           'plot_map',
           'plot_houseprice_map_latlon_fig',
           'plot_potential_risk_latlon_fig',
           'river_risk',
           'potential_map'
           ]

DEFAULT_POSTCODE_FILE = os.path.join(
    os.path.dirname(__file__), 'resources', 'postcodes_labelled.csv'
)

DEFAULT_STATION_FILE = os.path.join(
    os.path.dirname(__file__), 'resources', 'stations.csv'
)

DEFAULT_TYPICAL_FILE = os.path.join(
    os.path.dirname(__file__), 'example_data', 'typical_day.csv'
)

DEFAULT_WET_FILE = os.path.join(
    os.path.dirname(__file__), 'example_data', 'wet_day.csv'
)


def processing_day(path):
    '''Preprocessing data for mapping
    
    Parameters
    ----------
    path:str
        path of data file
    
        Returns
    -------
    dataframe
        data processed
    '''
    
    
    # read the file
    df=pd.read_csv(path)

    # transfer value to numeric
    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    #drop unuseful columns
    df = df.drop(['dateTime', 'qualifier'], axis=1)
    df = df.dropna(subset=['value'])
    
    #Filter data with consistent units
    
    condition = (df['parameter'] == 'rainfall') & (df['value'] < 0)
    df.loc[condition, 'value'] = 0
    df_filtered = df[((df['parameter'] == 'level') & (df['unitName'].isin(['mAOD', 'mASD']))) |
                 ((df['parameter'] == 'rainfall') & (df['unitName'].isin(['mm', 'm'])))]
    
    return df_filtered


def typical_wet_join(typical_day, wet_day,station):
    '''merge data of typical_day, wet_day and stations
    
    Parameters
    ----------
    typical_day:dataframe
        typical day data
    wet_day:dataframe
        wet day data
    station:
        station data
        Returns
    -------
    dataframe
        data merged 
    '''

    join_data = pd.concat([typical_day, wet_day], ignore_index=True)

    grouped = join_data.groupby(['stationReference', 'parameter', 'unitName'])['value'].agg(['mean', 'max', 'min'])
    grouped=grouped.reset_index()

    merged_data = pd.merge(station, grouped, 
                       on='stationReference', 
                       how='left')  
    return join_data, grouped,merged_data


def postcode_nearest_station(station, postcode):

    '''Find nearst station of each postcode
    
    Parameters
    ----------
    station:
        station data
    postcode:
        postcode data
        Returns
    -------
    dataframe
        post code with their nearest station 
    '''

    # create KNN model
    knn_model = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    
    # get station 和 postcode's coordinate
    station_coords = station[['easting', 'northing']].values
    postcode_coords = postcode[['easting', 'northing']].values

    # use station's lat and long fit the model
    knn_model.fit(station_coords)

    # calculate each postcode's distance and index to the nearest station
    distances, indices = knn_model.kneighbors(postcode_coords)
    print(distances.max())


    # add the nearest station's information to te postcode
    postcode['nearest_station'] = station.iloc[indices[:, 0]]['stationReference'].values
    postcode['distance_to_station'] = distances.flatten()  # 转为一维数组

    return postcode

def plot_map(
        postcode_file=DEFAULT_POSTCODE_FILE,
        station_file= DEFAULT_STATION_FILE,
        typical_file=DEFAULT_TYPICAL_FILE,
        wet_file=DEFAULT_WET_FILE):
    
    '''plot the interactive map
    
    Parameters
    ----------
    path of files
    
    Returns
    -------
    folium.Map

    '''
    

    postcode=pd.read_csv(postcode_file)


    postcode['latitude'], postcode['longitude'] = ft.get_gps_lat_long_from_easting_northing(
    postcode['easting'], postcode['northing'], rads=False, dms=False
    )
    m = folium.Map(location=[postcode["latitude"].mean(), postcode["longitude"].mean()], zoom_start=6)

    station=pd.read_csv(station_file)
    station=station.drop_duplicates()
    station=station.drop(columns=['maxOnRecord','minOnRecord','typicalRangeHigh','typicalRangeLow'])
    station=station.dropna()

    typical_day=processing_day(typical_file)
    wet_day = processing_day(wet_file)

    wet_day['value_str'] = wet_day['value'].astype(str)
    wet_day['value_length'] = wet_day['value_str'].str.len()
    wet_day['value_length'].value_counts()

    join_data,grouped,merge_data=typical_wet_join(typical_day, wet_day,station)

    merge_data=merge_data.dropna()


    merge_data['easting'], merge_data['northing'] = ft.geo.get_easting_northing_from_gps_lat_long(
    merge_data['latitude'], merge_data['longitude'])

    result_df = postcode_nearest_station(merge_data, postcode)
    mixed = pd.merge(result_df,merge_data, 
                       left_on='nearest_station', right_on='stationReference',
                       how='left')  

    mixed=mixed.dropna()
    mixed=mixed.reset_index()
    mixed=mixed.drop(columns=['index'])

    rainfall=mixed[mixed['unitName']=='mm']
    masd=mixed[mixed['unitName']=='mASD']
    maod=mixed[mixed['unitName']=='mAOD']

    norm1 = mcolors.Normalize(vmin=mixed['riskLabel'].min(), vmax=mixed['riskLabel'].max())
    norm2 = mcolors.Normalize(vmin=rainfall['mean'].min(), vmax=rainfall['mean'].max())
    norm3=  mcolors.Normalize(vmin=masd['mean'].min(), vmax=masd['mean'].max())
    norm4=  mcolors.Normalize(vmin=maod['mean'].min(), vmax=maod['mean'].max())

    cmap1 = plt.cm.Reds  # 
    cmap2= plt.cm.Blues
    cmap3= plt.cm.Greens
    cmap4 = LinearSegmentedColormap.from_list("yellow", ["white", "yellow"])

    layer1  = folium.FeatureGroup(name='risk label')
    for _, row in mixed.iterrows():
        color = mcolors.to_hex(cmap1(norm1(row['riskLabel']))) 
        popup_text = f"Postcode: {row['postcode']}, Risk: {row['riskLabel']}"
        folium.CircleMarker(
            location=[row['latitude_x'], row['longitude_x']],
            radius=0.1,  
            color=color,  
            fill=True,  
            fill_color=color,  
            popup=popup_text,
            fill_opacity=0.3  
        ).add_to(layer1)

    label = mixed['unitName'].unique()
    for i in label:
        if i == 'mm':
            name = 'rainfall'

        elif i == 'mASD':
            name = 'river'
        elif i == 'mAOD':
            name = 'tidal'
        else:
            name = '' 
        layer2 = folium.FeatureGroup(name=f'{name}: {i}')
        for _, row in mixed[mixed['unitName'] == i].iterrows():
            if name =='rainfall':
                color = mcolors.to_hex(cmap2(norm2(row['mean'])))
            elif name =='river':
                color = mcolors.to_hex(cmap3(norm3(row['mean'])))
            elif name =='tidal':
                color = mcolors.to_hex(cmap4(norm4(row['mean'])))

            popup_text = f"Postcode: {row['postcode']}, Risk: {row['riskLabel']},  {name}: {row['unitName']}, {name}_mean:{round((row['mean']),2)} "
            marker = folium.CircleMarker(
            location=[row['latitude_x'], row['longitude_x']], 
            radius=0.1, 
            color=color, fill=True, fill_color=color, popup=popup_text
                                    )
            marker.add_to(layer2)
        layer2.add_to(m)

    layer1.add_to(m)

        #add legend
    legend_html = """
    <div style="
        position: fixed;
        bottom: 50px; left: 50px; width: 200px; height: 120px;
        background-color: white;
        border:2px solid grey; z-index:9999; font-size:14px;
        padding: 10px;">
        <b>Legend</b><br>
        <i style="background:red; width:10px; height:10px; float:left; margin-right:5px;"></i> Risk<br>
        <i style="background:blue; width:10px; height:10px; float:left; margin-right:5px;"></i> Rainfall<br>
        <i style="background:green; width:10px; height:10px; float:left; margin-right:5px;"></i> River<br>
        <i style="background:yellow; width:10px; height:10px; float:left; margin-right:5px;"></i> Tidal<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)

    return m


def plot_houseprice_map_latlon_fig(risk_data,coordinate=['longitude', 'latitude'], dx=0.01,figsize=(10, 10), save_to_bytes=False):
    '''plot a raster map of houseprice
    
    Parameters
    ----------
    risk data:
        data used to plot,in this scene postcode data
    
    

    '''
    fig, ax = plt.subplots(figsize=figsize)

    risk_data['latitude'], risk_data['longitude'] = ft.get_gps_lat_long_from_easting_northing(
    risk_data['easting'], risk_data['northing'], rads=False, dms=False
    )

    bbox = (
        risk_data[coordinate[0]].min() - 0.5 * dx,
        risk_data[coordinate[0]].max() + 0.5 * dx,
        risk_data[coordinate[1]].min() - 0.5 * dx,
        risk_data[coordinate[1]].max() + 0.5 * dx,
    )

    nx = (
        math.ceil((bbox[1] - bbox[0]) / dx),
        math.ceil((bbox[3] - bbox[2]) / dx),
    )

    x = np.linspace(bbox[0] + 0.5 * dx, bbox[0] + (nx[0] - 0.5) * dx, nx[0])
    y = np.linspace(bbox[2] + 0.5 * dx, bbox[2] + (nx[1] - 0.5) * dx, nx[1])

    X, Y = np.meshgrid(x, y)

    Z = np.zeros(nx, int)
    k = np.zeros(nx,int)

    for x, y, val in zip(risk_data[coordinate[0]], risk_data[coordinate[1]], risk_data['medianPrice']):
        i = math.floor((x - bbox[0]) / dx)  #find the index of the x coordinate in the grid 
        j = math.floor((y - bbox[2]) / dx)  #find the index of the y coordinate in the grid 
        if 0 <= i < nx[0] and 0 <= j < nx[1]:  #make sure the index is within the grid 
            Z[i, j] += val  #add the house price to the grid 
            k[i, j] +=1

    Z = np.ma.masked_less_equal(Z, 0)
    Z = Z/k

    mappable = ax.pcolormesh(
        X, Y, np.where(Z > 0, Z, np.nan).T, cmap = plt.cm.coolwarm, norm=matplotlib.colors.LogNorm(vmin=Z.min(), vmax=Z.max())
    )

    shapefile_path = '../flood_tool/resources/ne_10m_admin_0_countries.shp'

        

    # read shapefile
    world = gpd.read_file(shapefile_path)

    # extract UK from world map
    uk = world[world['ADMIN'] == 'United Kingdom']

    # plot UK map
    uk.plot(ax=ax, color='none', edgecolor='black', linewidth=2)

    ax.set_title('House Price of the UK')
    ax.set_xlabel(coordinate[0])
    ax.set_ylabel(coordinate[1])
    ax.axis('equal')
    # set x and y axis limits
    ax.set_xlim(bbox[0] + 0.0005, bbox[1] - 0.0005)  
    ax.set_ylim(bbox[2] + 0.0005, bbox[3] - 0.0005)  
    # add colorbar
    plt.colorbar(mappable, ax=ax, label='House Price')



    if save_to_bytes:
        
        img_bytes = BytesIO()
        plt.savefig(img_bytes, format='PNG', bbox_inches='tight', pad_inches=0)
        plt.close()  
        img_bytes.seek(0)
        return img_bytes,bbox
    else:
        
        plt.show()


def river_risk(station,wet,typical):
    st=pd.read_csv(station)
    ty=pd.read_csv(typical)
    wet=pd.read_csv(wet)
    df=pd.concat([ty, wet], ignore_index=True)
    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    river=df[df['unitName']=='mASD']
    grouped = river.groupby(['stationReference'])['value'].agg(['mean', 'max', 'min'])
    grouped=grouped.reset_index()

    merge=pd.merge(grouped,st,on='stationReference',how='left')
    merge=merge.dropna()
    merge = merge[merge['max'] <= 100]
    merge['is_anomolous'] = ((merge['max'] > merge['maxOnRecord'])|((merge['max']-merge['min'])> merge['typicalRangeHigh'])).astype(int)
    
    return merge

def plot_potential_risk_latlon_fig(risk_data,coordinate=['longitude', 'latitude'], dx=0.01,figsize=(10, 10), save_to_bytes=False):
    '''plot a raster map of potential risk area
    
    Parameters
    ----------
    risk data:
        data used to plot,in this scene postcode data
    
    

    '''
    fig, ax = plt.subplots(figsize=figsize)

   

    bbox = (
        risk_data[coordinate[0]].min() - 0.5 * dx,
        risk_data[coordinate[0]].max() + 0.5 * dx,
        risk_data[coordinate[1]].min() - 0.5 * dx,
        risk_data[coordinate[1]].max() + 0.5 * dx,
    )

    nx = (
        math.ceil((bbox[1] - bbox[0]) / dx),
        math.ceil((bbox[3] - bbox[2]) / dx),
    )

    x = np.linspace(bbox[0] + 0.5 * dx, bbox[0] + (nx[0] - 0.5) * dx, nx[0])
    y = np.linspace(bbox[2] + 0.5 * dx, bbox[2] + (nx[1] - 0.5) * dx, nx[1])

    X, Y = np.meshgrid(x, y)

    Z = np.zeros(nx, int)
    k = np.zeros(nx,int)

    for x, y, val in zip(risk_data[coordinate[0]], risk_data[coordinate[1]], risk_data['mark']):
        i = math.floor((x - bbox[0]) / dx)  #find the index of the x coordinate in the grid 
        j = math.floor((y - bbox[2]) / dx)  #find the index of the y coordinate in the grid 
        if 0 <= i < nx[0] and 0 <= j < nx[1]:  #make sure the index is within the grid 
            Z[i, j] += val  #add the house price to the grid 
            k[i, j] +=1

    Z = np.ma.masked_less_equal(Z, 0)
    Z = Z/k

    mappable = ax.pcolormesh(
        X, Y, np.where(Z > 0, Z, np.nan).T, cmap = plt.cm.coolwarm, norm=matplotlib.colors.Normalize(vmin=Z.min(), vmax=Z.max())
    )

    shapefile_path = '../flood_tool/resources/ne_10m_admin_0_countries.shp'

        

    # read shapefile
    world = gpd.read_file(shapefile_path)

    # extract UK from world map
    uk = world[world['ADMIN'] == 'United Kingdom']

    # plot UK map
    uk.plot(ax=ax, color='none', edgecolor='black', linewidth=2)

    ax.set_title('Potential risk area of the UK')
    ax.set_xlabel(coordinate[0])
    ax.set_ylabel(coordinate[1])
    ax.axis('equal')
    # set x and y axis limits
    ax.set_xlim(bbox[0] + 0.0005, bbox[1] - 0.0005)  
    ax.set_ylim(bbox[2] + 0.0005, bbox[3] - 0.0005)  
    # add colorbar
    plt.colorbar(mappable, ax=ax, label='risk grade')



    if save_to_bytes:
        
        img_bytes = BytesIO()
        plt.savefig(img_bytes, format='PNG', bbox_inches='tight', pad_inches=0)
        plt.close()  
        img_bytes.seek(0)
        return img_bytes,bbox
    else:
        
        plt.show()


def potential_map(typical_path,wet_path,station_path):
    '''preprocessing data and plot the potential risk area of UK
    
    Parameters
    ----------
    typical_path:
        typical_day data's path
    wet_path:
        wet_day data's path
    station_path:
        stations data's path

    
    

    '''
    river=ft.river_risk(station_path,typical_path,wet_path)
    rainfall=ft.get_rainfall_class(typical_path,wet_path,station_path)
    tidal=ft.tidal_risk(station_path,wet_path,typical_path)

    #get data,with station reference,coordinate,is_anomolous map
    river_l=river[['stationReference','latitude','longitude','is_anomolous']]
    river_l=river_l.rename(columns={'is_anomolous':'river_mark'})
    
    rainfall_l=rainfall[['stationReference','latitude','longitude','rainfall_class_encode_max']]
    rainfall_l=rainfall_l.rename(columns={'rainfall_class_encode_max':'rainfall_mark'})
    
    tidal_l=tidal[['stationReference','latitude','longitude','is_anomalous']]
    tidal_l=tidal_l.rename(columns={'is_anomalous':'tidal_mark'})
    
    #merge the data
    merged1 = pd.merge(river_l, rainfall_l, on='stationReference', how='outer')
    merged1['latitude']=merged1['latitude_x'].combine_first(merged1['latitude_y'])
    merged1['longitude']=merged1['longitude_x'].combine_first(merged1['longitude_y'])
    merged1=merged1.drop(columns=['latitude_x','longitude_x','latitude_y','longitude_y'])
    merged1.fillna(0, inplace=True)
    merged2=pd.merge(merged1, tidal_l, on='stationReference', how='outer')
    
    #clean the dataa
    merged2['latitude']=merged2['latitude_x'].combine_first(merged2['latitude_y'])
    merged2['longitude']=merged2['longitude_x'].combine_first(merged2['longitude_y'])
    merged2=merged2.drop(columns=['latitude_x','longitude_x','latitude_y','longitude_y'])
    merged2.fillna(0, inplace=True)
    merged2['mark']=merged2['rainfall_mark']+merged2['river_mark']+merged2['tidal_mark']
    merged2=merged2[merged2['latitude']>0]
    
    #plot
    ft.plot_potential_risk_latlon_fig(merged2,dx=0.13,figsize=[5,5])

