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


__all__ = ['read_data','preprocess_day','process_day_rainfall','station_rainfall_loc','classify_rainfall','classify_rainfall_encode','get_rainfall_class']


DEFAULT_POSTCODE_FILE = os.path.join(
    os.path.dirname(__file__), 'resources', 'postcodes_unlabelled.csv'
)


def read_data(file_path):
    data = pd.read_csv(file_path)
    return data 


import pandas as pd
def preprocess_day(path):
    '''    Preprocesses the data for a single day by cleaning and filtering it.

    This function reads the data from the given file path, cleans up outliers, drops unnecessary columns,
    and filters the dataset based on specific conditions such as valid units and non-negative rainfall values.

    Parameters:
    -----------
    path : str
        The file path of the data to be processed.

    Returns:
    --------
    pandas.DataFrame
        A cleaned and filtered DataFrame ready for further analysis.
'''

    df=read_data(path)

    # Remove outliers from the value column
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    # print(df[['parameter','qualifier']].value_counts())

    # Delete useless columns
    df = df.drop(['qualifier'], axis=1)

    # Drop rows
    df = df.dropna(subset=['value'])
    
    # Filtering data with inconsistent units
    df_filtered = df[((df['parameter'] == 'level') & (df['unitName'].isin(['mAOD', 'mASD']))) |
                 ((df['parameter'] == 'rainfall') & (df['unitName'].isin(['mm', 'm'])))]
    
    # delete rows with negative rainfall values
    df_cleaned = df_filtered[(df_filtered['parameter'] != 'rainfall') | (df_filtered['value'] >= 0)]
    df_cleaned.drop_duplicates(subset=None, keep='first', inplace=True)

    return df_cleaned

def process_day_rainfall(day_data):
    """
    Processes daily rainfall data to calculate the mean, max, and min rainfall 
    for each station. It also computes the difference between max and min values.

    Parameters:
    day_data (pandas.DataFrame): Input dataframe containing daily data, including
                                  columns 'parameter', 'stationReference', and 'value'.

    Returns:
    pandas.DataFrame: A dataframe with the following columns for each station:
                       'stationReference', 'mean', 'max', 'min', and 'max_min',
                       where 'max_min' is the difference between 'max' and 'min'.
    """

    rainfall_data=day_data[day_data['parameter']=='rainfall']
    # print(rainfall_data['stationReference'].value_counts())
    rainfall_data=rainfall_data.groupby('stationReference')['value'].agg(['mean', 'max', 'min'])
    rainfall_data.reset_index(inplace=True)
    rainfall_data['max_min']=rainfall_data['max']-rainfall_data['min']
    return rainfall_data


# ???
def wet_typical_concat(rainfall_typical,rainfall_wet):
    """
    Concatenates typical day rainfall data with wet day rainfall data, retaining
    all stations from the wet day and adding stations from the typical day that
    are not already in the wet day data.

    Parameters:
    rainfall_typical (pandas.DataFrame): DataFrame containing typical day rainfall data.
    rainfall_wet (pandas.DataFrame): DataFrame containing wet day rainfall data.

    Returns:
    pandas.DataFrame: Combined DataFrame with stations from both typical and wet days.
    """
    new_stations = rainfall_typical[~rainfall_typical['stationReference'].isin(rainfall_wet['stationReference'])]
    # Retain all stations in the wetday, add stations not in the wetday but observed on the typical day to the wetday
    station_rainfall = pd.concat([rainfall_wet, new_stations], ignore_index=True)
    return station_rainfall

def station_rainfall_loc(rainfall_data,station_loc):
    """
    Merges rainfall data with station location data by processing and enriching 
    the station location information (including converting GPS to easting/northing).

    Parameters:
    rainfall_data (pandas.DataFrame): DataFrame containing rainfall data with 'stationReference'.
    station_loc (pandas.DataFrame): DataFrame containing station location data with 'stationReference', 
                                    'latitude', and 'longitude'.

    Returns:
    pandas.DataFrame: Merged DataFrame with rainfall data and corresponding station location details.
    """

    # process the station location data
    null_columns = ["stationName","latitude","longitude"]
    station_loc = station_loc.dropna(subset=null_columns)
    station_loc=station_loc.drop(['maxOnRecord','minOnRecord','typicalRangeHigh','typicalRangeLow'], axis=1)
    station_loc['easting'], station_loc['northing'] = ft.geo.get_easting_northing_from_gps_lat_long(
        station_loc['latitude'], station_loc['longitude']
    )

    # Merge the station location data with the rainfall data
    station_rainfall = pd.merge(rainfall_data, station_loc, how='left', on='stationReference')
    return station_rainfall
# ???



# classify rainfall
def classify_rainfall(rainfall_value):
        '''rainfall         	    classifier
        less than 2 mm per hour	     slight
        2mm to 4 mm per hour	     moderate
        4mm to 50 mm per hour	      heavy
        more than 50mm per hour	      violent'''
        if rainfall_value < 2:
            return "Slight"
        elif 2 <= rainfall_value < 4:
            return "Moderate"
        elif 4 <= rainfall_value < 50:
            return "Heavy"
        else:
            return "Violent"

def classify_rainfall_encode(rainfall_class):
        if rainfall_class == "Slight":
            return 0
        elif  rainfall_class =="Moderate":
            return 1
        elif rainfall_class =="Heavy":
            return 2
        else:
            return 3


def get_rainfall_class(typical_path,wet_path,station_path):
    """
    Processes rainfall data for typical and wet days, merges with station location data, 
    and classifies the rainfall based on mean and max values.

    Parameters:
    typical_path (str): Path to the typical day rainfall data.
    wet_path (str): Path to the wet day rainfall data.
    station_path (str): Path to the station location data.

    Returns:
    pandas.DataFrame: DataFrame with station rainfall data, including classified rainfall 
                      categories based on mean and max rainfall values.
                      the final dataframe
    """

    typical_day = preprocess_day(typical_path)
    wet_day = preprocess_day(wet_path)
    rainfall_wet=process_day_rainfall(wet_day)
    rainfall_typical=process_day_rainfall(typical_day)
    station_loc=read_data(station_path)

    # ???
    wet_typical=wet_typical_concat(rainfall_typical,rainfall_wet)
    station_rainfall=station_rainfall_loc(wet_typical,station_loc)
    # ???

    # Add classifications for average and maximum rainfall for each site

    station_rainfall['rainfall_class'] = station_rainfall['mean'].apply(classify_rainfall)
    station_rainfall['rainfall_class_encode'] = station_rainfall['rainfall_class'].apply(classify_rainfall_encode)
    station_rainfall['rainfall_class_max'] = station_rainfall['max'].apply(classify_rainfall)
    station_rainfall['rainfall_class_encode_max'] = station_rainfall['rainfall_class_max'].apply(classify_rainfall_encode)
    station_rainfall.dropna(inplace=True)
    return station_rainfall