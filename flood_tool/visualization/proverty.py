'''Analysis tools for flood data.'''

import os
import math

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from io import BytesIO

import geopandas as gpd
import numpy.ma as ma
import flood_tool as ft


__all__ = ['plot_risk_map','sector_proverty','process_proverty',
           'plot_rainfall_map_fig',
           'plot_historical_flood_map_latlon_fig',
           'plot_postcode_headcount_map_latlon_fig','plot_postcode_household_map_latlon_fig']


DEFAULT_POSTCODE_FILE = os.path.join(
    os.path.dirname(__file__), 'resources', 'postcodes_unlabelled.csv'
)


def plot_postcode_density(
    postcode_file=DEFAULT_POSTCODE_FILE,
    coordinate=['easting', 'northing'],
    dx=1000,
):
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


def plot_risk_map(risk_data, coordinate=['easting', 'northing'], dx=1000):
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

    for x, y, val in risk_data[coordinate + 'risk'].values:
        Z[
            math.floor((x - bbox[0]) / dx), math.floor((y - bbox[2]) / dx)
        ] += val

    plt.pcolormesh(
        X, Y, np.where(Z > 0, Z, np.nan).T, norm=matplotlib.colors.LogNorm()
    )
    plt.axis('equal')
    plt.colorbar()



def process_proverty(sector_path,postcode_path):
    '''
    This function mainly maps postcodes to sectors, calculates the average population and household numbers for each postcode, and returns a dataframe containing postcode, latitude, longitude, sector information, and population and household data.

    Detailed Steps:
    1. Process sector data: Convert the values in the postcodeSector column to uppercase and remove consecutive spaces.
    2. Process postcode data: Extract the postcode sector by taking the first few digits of the postcode field, convert it to uppercase, and remove consecutive spaces; calculate the latitude and longitude for each postcode using easting and northing.
    3. Merge postcode and sector data: Join the postcode data with the sector data based on the postcode sector.
    4. Handle missing values: Particularly remove rows with missing values in the households column.
    5. Calculate average population and household numbers: Calculate the average headcount (population) and average number of households for each postcode.
    
    Parameters:
    sector_path (str): Path to the sector data CSV file.
    postcode_path (str): Path to the postcode data CSV file.

    Return:
    The function returns a pandas DataFrame with the following columns:
        postcode: The postcode identifier.
        latitude: The latitude of the postcode.
        longitude: The longitude of the postcode.
        easting: The eastward coordinate.
        northing: The northward coordinate.
        postcodeSector: The sector identifier of the postcode.
        headcount: The population of the sector.
        households: The number of households in the sector.
        average_headcount: The average population for each postcode (headcount / numberOfPostcodeUnits).
        average_household: The average number of households for each postcode (households / numberOfPostcodeUnits).
    '''
    # preprocessing sector data
    sector_data=pd.read_csv(sector_path)
    sector_data['postcodeSector'] = sector_data['postcodeSector'].str.strip().str.upper()
    # Observe that there are consecutive spaces that need to be removed
    sector_data['postcodeSector'] = sector_data['postcodeSector'].str.replace(r'\s+', ' ', regex=True)

    # preprocessing postcode data
    postcode=pd.read_csv(postcode_path)
    postcode['postcodeSector'] = postcode['postcode'].str[:-2]
    postcode['postcodeSector'] = postcode['postcodeSector'].str.strip().str.upper()
    postcode['latitude'], postcode['longitude'] = ft.geo.get_gps_lat_long_from_easting_northing(
    postcode['easting'], postcode['northing']
    )

    # merging postcode and sector data
    postcode_proverty = pd.merge(postcode, sector_data, on='postcodeSector', how='left')
    postcode_proverty.dropna(subset=['households'], inplace=True)

    # calculating average proverty for each postcode
    postcode_proverty['average_headcount'] = postcode_proverty['headcount'] / postcode_proverty['numberOfPostcodeUnits']
    postcode_proverty['average_household'] = postcode_proverty['households'] / postcode_proverty['numberOfPostcodeUnits']

    return postcode_proverty


def sector_proverty(postcode_proverty):
    '''
    The data is grouped according to postcodeSector, the average coordinates of each area are calculated, and this average coordinate information is merged back into the original data table.

    This function accepts a DataFrame containing information about postal regions, calculates for each postal region (postcodeSector) its corresponding average latitude, longitude, eastward and northward coordinates, and merges the results of these calculations with the original data to return a data table containing only the relevant fields (e.g., total number of households in the region, number of postal units, number of people, and average coordinates, etc.).

    Parameters.
    postcode_proverty (pandas.DataFrame): a data table containing the postal areas (postcodeSector) and the coordinates and other information associated with each postal area. Must contain the following:
        - postcodeSector: postcode sector identifier
        - latitude: latitude
        - longitude: longitude
        - easting: east coordinate
        - northing: North coordinate
        - households: number of households
        - numberOfPostcodeUnits: number of postal units
        - headcount: population

    Returns: pandas.
    pandas.DataFrame: Returns a new DataFrame containing the following:
        - postcodeSector: postcode sector identifier
        - households: number of households
        - numberOfPostcodeUnits: number of postcode units
        - headcount: population count
        - sector_mean_latitude: mean latitude
        - sector_mean_latitude: Mean longitude
        - sector_mean_easting: Mean easting coordinates
        - sector_mean_northing: Mean northward coordinates

    '''
    sector_mean_coords = postcode_proverty.groupby('postcodeSector', as_index=False)[['latitude', 'longitude','easting', 'northing']].mean()
    sector_mean_coords.rename(columns={'latitude': 'sector_mean_latitude', 'longitude': 'sector_mean_longitude','easting': 'sector_mean_easting', 'northing': 'sector_mean_northing'}, inplace=True)

# 合并回原表
    postcode_proverty = postcode_proverty.merge(sector_mean_coords, on='postcodeSector', how='left')
    sector_columns=['postcodeSector','households','numberOfPostcodeUnits','headcount','sector_mean_latitude','sector_mean_longitude','sector_mean_easting','sector_mean_northing']
    sector_proverty=postcode_proverty[sector_columns].drop_duplicates()
# sector_proverty.drop_duplicates(inplace=True)
    return sector_proverty




def plot_rainfall_map_fig(data, coordinate=['easting', 'northing'], dx=1000, figsize=(10, 10), save_to_bytes=False):
    '''Plot a rainfall map.'''

    bbox = (
       data[coordinate[0]].min() - 0.5 * dx,
       data[coordinate[0]].max() + 0.5 * dx,
       data[coordinate[1]].min() - 0.5 * dx,
       data[coordinate[1]].max() + 0.5 * dx,
    )

    nx = (
        math.ceil((bbox[1] - bbox[0]) / dx),
        math.ceil((bbox[3] - bbox[2]) / dx),
    )

    x = np.linspace(bbox[0] + 0.5 * dx, bbox[0] + (nx[0] - 0.5) * dx, nx[0])
    y = np.linspace(bbox[2] + 0.5 * dx, bbox[2] + (nx[1] - 0.5) * dx, nx[1])

    X, Y = np.meshgrid(x, y)

    Z = np.zeros(nx, int)

    for x, y, val in zip(data[coordinate[0]],data[coordinate[1]],data['rainfall_class_encode_max']):
        i = math.floor((x - bbox[0]) / dx)  #find the index of the x coordinate in the grid 
        j = math.floor((y - bbox[2]) / dx)  #find the index of the y coordinate in the grid 
        if 0 <= i < nx[0] and 0 <= j < nx[1]:  #make sure the index is within the grid 
            Z[i, j] += val  #add the risk value to the grid 

   
    plt.pcolormesh(
        X, Y, np.where(Z > 0, Z, np.nan).T, cmap='Reds', norm=matplotlib.colors.Normalize(vmin=data['rainfall_class_encode_max'].min(), vmax=data['rainfall_class_encode_max'].max())
    )
    plt.axis('equal')
    plt.colorbar(label='rainfall_class')
    plt.title('rainfall_class Map')
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


def plot_historical_flood_map_latlon_fig(risk_data, coordinate=['longitude', 'latitude'], dx=0.01,figsize=(10, 10), save_to_bytes=False):
    '''Plot a historial_flood map.'''
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
    # k= np.ones(nx,int)

    for x, y, val in zip(risk_data[coordinate[0]], risk_data[coordinate[1]], risk_data['historicallyFlooded']):
        i = math.floor((x - bbox[0]) / dx)  #find the index of the x coordinate in the grid 
        j = math.floor((y - bbox[2]) / dx)  #find the index of the y coordinate in the grid 
        if 0 <= i < nx[0] and 0 <= j < nx[1]:  #make sure the index is within the grid 
            Z[i, j] += val  #add the risk value to the grid 
            # print(Z[i, j])
    Z=np.ma.masked_less_equal(Z, 0)
    # Z = Z+k

    

    mappable = ax.pcolormesh(
        X, Y, np.where(Z > 0, Z, np.nan).T, cmap=plt.cm.coolwarm, norm=matplotlib.colors.LogNorm(vmin=Z.min(), vmax=Z.max())
    )

    shapefile_path = '../flood_tool/resources/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp'

        

    # read shapefile
    world = gpd.read_file(shapefile_path)

    # extract UK from world map
    uk = world[world['ADMIN'] == 'United Kingdom']

    # plot UK map
    uk.plot(ax=ax, color='none', edgecolor='black', linewidth=2)

    ax.set_title('historicallyFlooded Map of the UK')
    ax.set_xlabel(coordinate[0])
    ax.set_ylabel(coordinate[1])
    ax.axis('equal')
    # set x and y axis limits
    ax.set_xlim(bbox[0] + 0.0005, bbox[1] - 0.0005)  
    ax.set_ylim(bbox[2] + 0.0005, bbox[3] - 0.0005)  
    # add colorbar
    plt.colorbar(mappable, ax=ax, label='historicallyFlooded')


    if save_to_bytes:
        
        img_bytes = BytesIO()
        plt.savefig(img_bytes, format='PNG', bbox_inches='tight', pad_inches=0)
        plt.close()  
        img_bytes.seek(0)
        return img_bytes,bbox
    else:
        
        plt.show()



def plot_postcode_headcount_map_latlon_fig(risk_data, coordinate=['longitude', 'latitude'], dx=0.01,figsize=(10, 10), save_to_bytes=False):
    '''Plot a headcount map in postcode.'''
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

    for x, y, val in zip(risk_data[coordinate[0]], risk_data[coordinate[1]], risk_data['average_headcount']):
        i = math.floor((x - bbox[0]) / dx)  #find the index of the x coordinate in the grid 
        j = math.floor((y - bbox[2]) / dx)  #find the index of the y coordinate in the grid 
        if 0 <= i < nx[0] and 0 <= j < nx[1]:  #make sure the index is within the grid 
            Z[i, j] += val  #add the risk value to the grid 
    Z = np.ma.masked_less_equal(Z, 0)

    mappable = ax.pcolormesh(
        X, Y, np.where(Z > 0, Z, np.nan).T, cmap = plt.cm.Oranges, norm=matplotlib.colors.LogNorm(vmin=Z.min(), vmax=Z.max())
    )

    shapefile_path = '../flood_tool/resources/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp'

        

    # read shapefile
    world = gpd.read_file(shapefile_path)

    # extract UK from world map
    uk = world[world['ADMIN'] == 'United Kingdom']

    # plot UK map
    uk.plot(ax=ax, color='none', edgecolor='black', linewidth=2)

    ax.set_title('headcount Map of the UK')
    ax.set_xlabel(coordinate[0])
    ax.set_ylabel(coordinate[1])
    ax.axis('equal')
    # set x and y axis limits
    ax.set_xlim(bbox[0] + 0.0005, bbox[1] - 0.0005)  
    ax.set_ylim(bbox[2] + 0.0005, bbox[3] - 0.0005)  
    # add colorbar
    plt.colorbar(mappable, ax=ax, label='headcount')


    if save_to_bytes:
        
        img_bytes = BytesIO()
        plt.savefig(img_bytes, format='PNG', bbox_inches='tight', pad_inches=0)
        plt.close()  
        img_bytes.seek(0)
        return img_bytes,bbox
    else:
        
        plt.show()



def plot_postcode_household_map_latlon_fig(risk_data, coordinate=['longitude', 'latitude'], dx=0.01,figsize=(10, 10), save_to_bytes=False):
    """
    Plots a household density map for postcodes using latitude and longitude coordinates.
    use dx: 0.01,because we want to show the postcode density map in the UK, the data set is quite big, we want to plot the map quickly and clearly

    Parameters:
    risk_data (pandas.DataFrame): Data containing longitude, latitude, and 'average_household' columns.
    coordinate (list of str, optional): The column names for the longitude and latitude coordinates. Defaults to ['longitude', 'latitude'].
    dx (float, optional): The grid resolution for the map in degrees. Defaults to 0.01.
    figsize (tuple, optional): Size of the figure in inches. Defaults to (10, 10).
    save_to_bytes (bool, optional): If True, saves the plot as a PNG image in memory and returns it as bytes. Defaults to False.

    Returns:
    - If save_to_bytes is True: A tuple containing the plot as bytes and the bounding box of the map.
    - If save_to_bytes is False: Displays the plot directly and returns None.
    """
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

    for x, y, val in zip(risk_data[coordinate[0]], risk_data[coordinate[1]], risk_data['average_household']):
        i = math.floor((x - bbox[0]) / dx)  #find the index of the x coordinate in the grid 
        j = math.floor((y - bbox[2]) / dx)  #find the index of the y coordinate in the grid 
        if 0 <= i < nx[0] and 0 <= j < nx[1]:  #make sure the index is within the grid 
            Z[i, j] += val  #add the risk value to the grid 
    Z = np.ma.masked_less_equal(Z, 0)

    mappable = ax.pcolormesh(
        X, Y, np.where(Z > 0, Z, np.nan).T, cmap = plt.cm.Oranges, norm=matplotlib.colors.LogNorm(vmin=Z.min(), vmax=Z.max())
    )

    shapefile_path = '../flood_tool/resources/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp'

        

    # read shapefile
    world = gpd.read_file(shapefile_path)

    # extract UK from world map
    uk = world[world['ADMIN'] == 'United Kingdom']

    # plot UK map
    uk.plot(ax=ax, color='none', edgecolor='black', linewidth=2)

    ax.set_title('household Map of the UK')
    ax.set_xlabel(coordinate[0])
    ax.set_ylabel(coordinate[1])
    ax.axis('equal')
    # set x and y axis limits
    ax.set_xlim(bbox[0] + 0.0005, bbox[1] - 0.0005)  
    ax.set_ylim(bbox[2] + 0.0005, bbox[3] - 0.0005)  
    # add colorbar
    plt.colorbar(mappable, ax=ax, label='household')


    if save_to_bytes:
        
        img_bytes = BytesIO()
        plt.savefig(img_bytes, format='PNG', bbox_inches='tight', pad_inches=0)
        plt.close()  
        img_bytes.seek(0)
        return img_bytes,bbox
    else:
        
        plt.show()
