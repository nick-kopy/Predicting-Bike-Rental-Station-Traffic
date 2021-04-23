# This file contains all the necessary functions for model.ipynb to run
# It mostly collects, cleans, and presents data
# Authored by Nicholas Kopystynsky

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

from geopy.distance import geodesic
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def basic_dist(row):
    '''Gives a basic euclidean trip distance in meters'''

    if row['round_trip'] == 1:
        return 0

    a = (row['start_lat'], row['start_lng'])

    b = (row['end_lat'], row['end_lng'])

    return geodesic(a, b).km * 1000

def grab_data(region=None):
    '''Loads, preps, and filters data for machine learning

    input: a set of strings, all station names

    output: pd dataframe of recent Divvy trips
      - Output is not quite AI ready or EDA ready, but right where they would branch

    '''

    # Gather one years worth of data
    filelist = []
    frames = []

    for month in [4,5,6,7,8,9,10,11,12]:
        filelist.append('data/2020{:02d}-divvy-tripdata.csv'.format(month))
    for month in [1,2,3]:
        filelist.append('data/2021{:02d}-divvy-tripdata.csv'.format(month))

    usecols = ['started_at', 'ended_at', 'start_station_name', 'end_station_name', 'member_casual', 'rideable_type',
               'start_lat', 'start_lng', 'end_lat', 'end_lng']

    for month in filelist:
        lil_df = pd.read_csv(month, usecols=usecols)

        # If you want all the data, just leave region=None
        if region is not None:
            # filter out what isn't in our region
            mask1 = (lil_df['end_station_name'].isin(region))
            mask2 = (lil_df['start_station_name'].isin(region))
            mask = mask1 | mask2

            lil_df = lil_df[mask]

        frames.append(lil_df)

    df = pd.concat(frames, ignore_index=True)

    # Only relevant missing data is lat/long, warns us if ever dropping more than 1%
    allrows = df.shape[0]

    df = df[df['start_lat'].notna()]
    df = df[df['end_lat'].notna()]
    if allrows/df.shape[0] > 1.01:
        print('NULL WARNING: more than 1% of rows null')

    df = df.reset_index(drop=True)

    # target variable is grouped by date and hour
    df['ended_at'] = pd.to_datetime(df['ended_at'])
    df['started_at'] = pd.to_datetime(df['started_at'])

    df['date'] = pd.to_datetime(df['ended_at']).dt.date
    df['hour'] = df['ended_at'].dt.hour

    # For some reason each month has a few trips from the upcoming month
    # removed to prevent data leakage
    df = df[df['date'] < pd.to_datetime('2021-04-01')]

    # daylight savings makes a few negative trip times, a quick approximate fix is okay
    df['trip_time'] = abs((df['ended_at'] - df['started_at']).dt.total_seconds())

    # All trips above 10,800 seconds (3 hrs) are on Nov 25, must be some systemic thing
    df = df[df['trip_time'] < 10800]

    # Extracting some interesting features
    df['round_trip'] = df.apply(lambda x: 1 if x['start_station_name'] == x['end_station_name'] else 0, axis=1)

    df['electric'] = df['rideable_type'].apply(lambda x: 1 if x == 'electric_bike' else 0)

    df['member'] = df['member_casual'].apply(lambda x: 1 if x == 'member' else 0)

    df['trip_dist'] = df.apply(basic_dist, axis=1)

    # only return what we need
    dropcols = ['rideable_type', 'member_casual', 'started_at', 'ended_at',
                'start_lat', 'start_lng', 'end_lat', 'end_lng']

    if region is not None:
        dropcols.append('start_station_name')
        dropcols.append('end_station_name')

    df = df.drop(columns=dropcols)

    return df

def vectorize(inputdf, return_scaler=False):
    '''Prepares data for machine learning

    input: df from grab_data

    output: 2D numpy array
      - all non-target feature columns are scaled
    '''

    out = inputdf.groupby(['date', 'hour']).agg('mean')

    out['size'] = inputdf.groupby(['date', 'hour']).size()

    dti = pd.Series(pd.date_range("2020-04-01", freq="D", periods=365)).dt.date

    idx = pd.MultiIndex.from_product([dti, np.arange(24)], names=['date', 'hour'])

    df_blank = pd.DataFrame(data = np.zeros(shape=(365*24, 6)), index=idx,
                            columns=['trip_time', 'round_trip', 'electric', 'member',
                                     'trip_dist', 'size'])

    out = pd.concat([df_blank, out]).groupby(['date', 'hour']).agg('sum')

    out = out.reset_index(drop=True)

    # If you don't want scaled data, use this return and comment out below it
    #return np.array(out)

    y = np.array(out.iloc[:, -1])

    scaler = MinMaxScaler()

    out = scaler.fit_transform(out.iloc[:, :-1])

    if return_scaler == False:
      return np.append(out, y.reshape(-1, 1), axis=1)
    else:
      return np.append(out, y.reshape(-1, 1), axis=1), scaler