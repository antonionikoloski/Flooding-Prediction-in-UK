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

__all__ = [
           "get_tidal_change",
           'calculate_tidal_threshold',
           'flag_anomalies',
           'tidal_risk'
           ]


DEFAULT_POSTCODE_FILE = os.path.join(
    os.path.dirname(__file__), 'resources', 'postcodes_unlabelled.csv'
)

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



def tidal_risk(station,wet,typical):
    tdf = pd.read_csv(typical)
    wdf = pd.read_csv(wet)
    concat = pd.concat([wdf, tdf])
    tidal = concat[concat['unitName'] == 'mAOD']
    tidal = ft.get_tidal_change(tidal)
    tidal['value_diff_no_negative'] = tidal['value_diff'].where(tidal['value_diff'] > 0, other=np.nan)
    thresholds = tidal.groupby('stationReference').apply(lambda group: ft.calculate_tidal_threshold(group, k=2))
    tidal = tidal.reset_index(drop=True)
    tidal = tidal.groupby('stationReference').apply(lambda group: ft.flag_anomalies(group, thresholds = thresholds))

    # Plot the anomalous rises
    if 'is_anomalous' in tidal.columns:
 # Plot the anomalous rises
        anomalous_rises = tidal[tidal['is_anomalous' ]==1]
        anomalous_rises
    else:
        print("The 'is_anomalous' column does not exist in the DataFrame.")
    station = pd.read_csv('../flood_tool/resources/stations.csv')
    anomalous_rises = anomalous_rises.reset_index(drop=True)

    merged = pd.merge(station, anomalous_rises,left_on='stationReference', right_on='stationReference',how = 'right')
    merged = merged.dropna(subset=['stationName'])
    merged['is_anomalous'] = merged['is_anomalous'].astype(int)
    return merged



