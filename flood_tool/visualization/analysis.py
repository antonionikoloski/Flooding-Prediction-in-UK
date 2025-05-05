'''Analysis tools for flood data.'''

import os
import math

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from io import BytesIO

import geopandas as gpd
import numpy.ma as ma

import flood_tool as ft


__all__ = ['plot_risk_map',
           'plot_postcode_density',
           'plot_risk_map_latlon',
           'plot_uk_map',
           'plot_risk_map_latlon_fig',
           "get_tidal_change",
           'calculate_tidal_threshold',
           'flag_anomalies',
           'plot_tidal_changes'
           ]

DEFAULT_POSTCODE_FILE = os.path.join(
    os.path.dirname(__file__), 'resources', 'postcodes_unlabelled.csv'
)


def plot_postcode_density(
    postcode_file=DEFAULT_POSTCODE_FILE,
    coordinate=['easting', 'northing'],
    dx=1000):
    '''Plot a postcode density map from a postcode file.'''

    pdb = pd.read_csv(postcode_file)

    bbox = (
        pdb[coordinate[0]].min() - 0.5 * dx,
        pdb[coordinate[0]].max() + 0.5 * dx,
        pdb[coordinate[1]].min() - 0.5 * dx,
        pdb[coordinate[1]].max() + 0.5 * dx,
    )

    nx = (
        math.ceil((bbox[1] - bbox[0]) / dx),
        math.ceil((bbox[3] - bbox[2]) / dx),
    )

    x = np.linspace(bbox[0] + 0.5 * dx, bbox[0] + (nx[0] - 0.5) * dx, nx[0])
    y = np.linspace(bbox[2] + 0.5 * dx, bbox[2] + (nx[1] - 0.5) * dx, nx[1])

    X, Y = np.meshgrid(x, y)

    Z = np.zeros(nx, int)

    for x, y in pdb[coordinate].values:
        Z[math.floor((x - bbox[0]) / dx), math.floor((y - bbox[2]) / dx)] += 1

    plt.pcolormesh(
        X, Y, np.where(Z > 0, Z, np.nan).T, norm=matplotlib.colors.LogNorm()
    )
    plt.axis('equal')
    plt.colorbar()


def plot_risk_map(risk_data, coordinate=['easting', 'northing'], dx=1000, figsize=(10, 10), save_to_bytes=False):
    '''Plot a risk map.'''

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

    for x, y, val in zip(risk_data[coordinate[0]], risk_data[coordinate[1]], risk_data['riskLabel']):
        i = math.floor((x - bbox[0]) / dx)  #find the index of the x coordinate in the grid 
        j = math.floor((y - bbox[2]) / dx)  #find the index of the y coordinate in the grid 
        if 0 <= i < nx[0] and 0 <= j < nx[1]:  #make sure the index is within the grid 
            Z[i, j] += val  #add the risk value to the grid 

   

        
   
    plt.pcolormesh(
        X, Y, np.where(Z > 0, Z, np.nan).T, cmap='Reds', norm=matplotlib.colors.Normalize(vmin=0, vmax=7)
    )
    plt.axis('equal')
    plt.colorbar(label='Risk')
    plt.title('Risk Map')
    plt.xlabel(coordinate[0])
    plt.ylabel(coordinate[1])

    if save_to_bytes:
        
        img_bytes = BytesIO()
        plt.savefig(img_bytes, format='PNG', bbox_inches='tight', pad_inches=0)
        plt.close()  
        img_bytes.seek(0)
        return img_bytes
    else:
        plt.show()


def plot_risk_map_latlon_fig(risk_data, coordinate=['longitude', 'latitude'], dx=0.0001,figsize=(10, 10), save_to_bytes=False):
    '''Plot a risk map.'''

    risk_data['latitude'], risk_data['longitude'] = ft.get_gps_lat_long_from_easting_northing(
    risk_data['easting'], risk_data['northing'], rads=False, dms=False
    )
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


    for x, y, val in zip(risk_data[coordinate[0]], risk_data[coordinate[1]], risk_data['riskLabel']):
        i = math.floor((x - bbox[0]) / dx)  #find the index of the x coordinate in the grid 
        j = math.floor((y - bbox[2]) / dx)  #find the index of the y coordinate in the grid 
        if 0 <= i < nx[0] and 0 <= j < nx[1]:  #make sure the index is within the grid 
            Z[i, j] += val  #add the risk value to the grid 

    Z = np.ma.masked_less_equal(Z, 0)

    mappable = ax.pcolormesh(
        X, Y, np.where(Z > 0, Z, np.nan).T, cmap = plt.cm.coolwarm, norm=matplotlib.colors.LogNorm(vmin=Z.min(), vmax=Z.max())
    )

    shapefile_path = '../flood_tool/resources/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp'

        

    # read shapefile
    world = gpd.read_file(shapefile_path)

    # extract UK from world map
    uk = world[world['ADMIN'] == 'United Kingdom']

    # plot UK map
    uk.plot(ax=ax, color='none', edgecolor='black', linewidth=2)

    ax.set_title('Risk Map of the UK')
    ax.set_xlabel(coordinate[0])
    ax.set_ylabel(coordinate[1])
    ax.axis('equal')
    # set x and y axis limits
    ax.set_xlim(bbox[0] + 0.0005, bbox[1] - 0.0005)  
    ax.set_ylim(bbox[2] + 0.0005, bbox[3] - 0.0005)  
    # add colorbar
    plt.colorbar(mappable, ax=ax, label='RiskLevel')


    # #create inset
    # axins = inset_axes(ax, width="30%", height="30%", loc='lower right')  # 右下角的 inset
    # uk.plot(ax=axins, color='lightgray', edgecolor='black', linewidth=2)
    
    # #
    # axins.set_title("United Kingdom (Inset)")
    
    # #
    # axins.set_xticks([])
    # axins.set_yticks([])



    

    if save_to_bytes:
        
        img_bytes = BytesIO()
        plt.savefig(img_bytes, format='PNG', bbox_inches='tight', pad_inches=0)
        plt.close()  
        img_bytes.seek(0)
        return img_bytes,bbox
    else:
        
        plt.show()


def plot_uk_map(ax=None):
    '''Plot a map of the UK.'''
    shapefile_path = '../flood_tool/resources/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp'

    if ax is None:
        ax = plt.gca()  # get the current axes

    # read shapefile
    world = gpd.read_file(shapefile_path)

    # extract UK from world map
    uk = world[world['ADMIN'] == 'United Kingdom']


   
    fig, ax = plt.subplots(figsize=(10, 10))

    uk.plot(ax=ax, color='white', edgecolor='black')
    ax.set_title("Map of United Kingdom")
    plt.show()

def plot_risk_map_latlon(risk_data, coordinate=['longitude', 'latitude'], dx=0.0001,figsize=(10, 10), save_to_bytes=False):
    '''Plot a risk map.'''
   

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


    for x, y, val in zip(risk_data[coordinate[0]], risk_data[coordinate[1]], risk_data['riskLabel']):
        i = math.floor((x - bbox[0]) / dx)  #find the index of the x coordinate in the grid 
        j = math.floor((y - bbox[2]) / dx)  #find the index of the y coordinate in the grid 
        if 0 <= i < nx[0] and 0 <= j < nx[1]:  #make sure the index is within the grid 
            Z[i, j] += val  #add the risk value to the grid 


    


    cmap = plt.cm.coolwarm 
    norm = matplotlib.colors.LogNorm(vmin=np.nanmin(Z), vmax=np.nanmax(Z))


    plt.pcolormesh(
        X, Y, np.where(Z > 0, Z, np.nan).T, cmap=cmap, norm=norm)

    

    
    
    # add colorbar
    plt.colorbar(label='RiskLevel')
    plt.axis('off') 


    # #create inset
    # axins = inset_axes(ax, width="30%", height="30%", loc='lower right')  # 右下角的 inset
    # uk.plot(ax=axins, color='lightgray', edgecolor='black', linewidth=2)
    
    # #
    # axins.set_title("United Kingdom (Inset)")
    
    # #
    # axins.set_xticks([])
    # axins.set_yticks([])



    

    if save_to_bytes:
        
        img_bytes = BytesIO()

        plt.savefig(img_bytes, format='PNG', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()  
        img_bytes.seek(0)
        return img_bytes,bbox
    else:
        
        plt.show()






def get_tidal_change(tidal,):
    '''
        Analyse the tidal data
        find the unusual tidal data
    '''
    tidal['dateTime'] = pd.to_datetime(tidal['dateTime'])

    # convert the value column to numeric
    tidal['value'] = pd.to_numeric(tidal['value'], errors='coerce')

    # calculate the difference between the current value and the previous value
    tidal['value_diff'] = tidal.groupby('stationReference')['value'].diff()

    #  calculate the absolute value of the difference
    tidal['value_change'] = tidal['value_diff'].abs()

    #  calculate the time difference between the current value and the previous value
    tidal = tidal.sort_values(['stationReference', 'dateTime']).reset_index(drop=True)
    return tidal

def calculate_tidal_threshold(group, k=2):
    mean_diff = group['value_diff'].mean()
    std_diff = group['value_diff'].std()
    return mean_diff + k * std_diff

def flag_anomalies(group,thresholds):
    threshold = thresholds[group.name]
    group['is_anomalous'] = np.where(group['value_diff'] > threshold, 1, 0)
    return group

def plot_tidal_changes(tidal,max_stations = 3):
    tidal = tidal.reset_index(drop=True)

    # 初始化计数器

    count = 0
    # 按 stationReference 分组并绘图
    for station, group in tidal.groupby('stationReference'):
        if count >= max_stations:  # 如果绘图数量达到限制，退出循环
            break
        plt.figure(figsize=(10, 4))
        plt.plot(group['dateTime'], group['value'], marker='o', label='Value')
        plt.bar(group['dateTime'], group['value_diff'], alpha=0.4, label='Change')
        plt.title(f"Tidal Changes at {station}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.show()
        count += 1 
