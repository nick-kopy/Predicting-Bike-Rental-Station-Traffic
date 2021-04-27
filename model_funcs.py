# This file contains all the necessary functions for model.ipynb to run
# To see examples of how to use these functions, see above mentioned notebook
# Authored by Nicholas Kopystynsky

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from geopy.distance import geodesic
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

plt.style.use('ggplot')

def basic_dist(row):
    '''Gives a basic euclidean trip distance in meters'''

    if row['round_trip'] == 1:
        return 0

    a = (row['start_lat'], row['start_lng'])

    b = (row['end_lat'], row['end_lng'])

    return geodesic(a, b).km * 1000

def station_data(region, eda=False, start_end=None, exclude_within_region=False):
    '''Loads, preps, and filters data for machine learning

    input: a set of strings, all station names

    output: pd dataframe of recent Divvy trips
      - Output is not quite AI ready or EDA ready, but right where they would branch
      
    options:
      - eda: If True includes extra columns with trip related statistics. Should be excluded for modeling.
      - start_end: Pick if you want trips that start in a region or end in a region or leave blank for both.
      - exclude_within_region: If a trip started and ended within a region, excludes those trips.

    '''
    # grab a set of station names for a given region
    if region in ['downtown', 'lincoln_park', 'wicker_park', 'hyde_park', 'uptown', 'chinatown']:
        stations = get_stations(region)
    else:
        stations = set([region])

    # Gather one years worth of data
    filelist = []
    frames = []

    # change this back to the full year later
    for month in [10,11,12]: #[4,5,6,7,8,9,10,11,12]:
        filelist.append('data/2020{:02d}-divvy-tripdata.csv'.format(month))
    for month in [1,2,3]:
        filelist.append('data/2021{:02d}-divvy-tripdata.csv'.format(month))

    usecols = ['started_at', 'ended_at', 'start_station_name', 'end_station_name', 'member_casual', 'rideable_type',
               'start_lat', 'start_lng', 'end_lat', 'end_lng']

    # actually grab the data
    for month in filelist:
        lil_df = pd.read_csv(month, usecols=usecols)

        # decide weather to look at trips starting and/or ending in our selected region
        mask_end = (lil_df['end_station_name'].isin(stations))
        mask_start = (lil_df['start_station_name'].isin(stations))
        
        # want trips ending in our region, but may or may not want those starting in our region
        if start_end == 'end':
            if exclude_within_region == False:
                mask = mask_end
            elif exclude_within_region == True:
                mask = mask_end & ~mask_start
                
        # want trips starting in our region, but may or may not want those ending in our region
        elif start_end == 'start':
            if exclude_within_region == False:
                mask = mask_start
            elif exclude_within_region == True:
                mask = mask_start & ~mask_end
        
        # want all trips that started or ended in our region but may or may not want trips that did both
        else:
            if exclude_within_region == False:
                # started or ended in region
                mask = mask_start | mask_end
            elif exclude_within_region == True:
                # started xor ended in region
                mask = (mask_start & ~mask_end) | (~mask_start & mask_end)

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

    # For some reason each month has a extra few trips from the upcoming month
    # removed to prevent data leakage
    df = df[df['date'] < pd.to_datetime('2021-04-01')]

    # Future implementation can include weather data
    #weather = grab_weather()
    #df = df.merge(weather, how='left', left_on=['date', 'hour'], right_on=['date', 'hour'])

    # AI wouldn't have following aggregate features available for predictions, so they aren't included in modeling
    if eda == False:
        # instead prep for machine learning
        return vectorize(df)

    # Extracting some interesting features for EDA

    # daylight savings makes a few negative trip times, a quick approximate fix is okay
    df['trip_time'] = abs((df['ended_at'] - df['started_at']).dt.total_seconds())

    # All trips above 10,800 seconds (3 hrs) are on Nov 25, must be some systemic thing
    df = df[df['trip_time'] < 10800]

    df['round_trip'] = df.apply(lambda x: 1 if x['start_station_name'] == x['end_station_name'] else 0, axis=1)

    df['electric'] = df['rideable_type'].apply(lambda x: 1 if x == 'electric_bike' else 0)

    df['member'] = df['member_casual'].apply(lambda x: 1 if x == 'member' else 0)

    df['trip_dist'] = df.apply(basic_dist, axis=1)

    dropcols = ['rideable_type', 'member_casual', 'started_at', 'ended_at', 'start_lat', 'start_lng',
                'end_lat', 'end_lng', 'start_station_name',  'end_station_name']

    df = df.drop(columns=dropcols)

    # extract target and add to output
    out = df.groupby(['date', 'hour']).agg('mean')

    out['target'] = df.groupby(['date', 'hour']).size()

    # Some hours are missing, we want to include a row for that hour with target = 0
    dti = pd.Series(pd.date_range("2020-04-01", freq="D", periods=365)).dt.date

    idx = pd.MultiIndex.from_product([dti, np.arange(24)], names=['date', 'hour'])

    # When weather is implemented column names will need to be included below
    df_blank = pd.DataFrame(data = np.zeros(shape=(365*24, 6)),
                            index = idx,
                            columns = ['trip_time', 'round_trip', 'electric', 'member',
                                       'trip_dist', 'target'])

    out = pd.concat([df_blank, out]).groupby(['date', 'hour']).agg('sum')

    return out

def grab_weather():
    '''Future implementation: loads and preps weather data in a pandas df'''
    pass

def get_stations(region):
    '''Returns the set of station names necessary for grouping data

    possible regions: 'downtown', 'lincoln_park', 'wicker_park', 'hyde_park', 'uptown', 'chinatown'
    '''

    groups = pd.read_csv('models/station_groups.csv')

    return set(groups[groups['group'] == region].name.values)

def vectorize(inputdf):
    '''Prepares data for machine learning

    input: df from grab_data

    output: 1D numpy array
      - all non-target feature columns are scaled

    future output: 2D numpy array, scaler used
      - feature columns would all need to be scaled
    '''

    # extract target and add to output
    out = inputdf.groupby(['date', 'hour']).agg('mean')

    out['target'] = inputdf.groupby(['date', 'hour']).size()

    dropcols = ['start_lat', 'start_lng', 'end_lat', 'end_lng']

    out = out.drop(columns=dropcols)

    # Some hours are missing, we want to include a row for that hour with target = 0
    # Merging with a blank df seems to cover our bases
    dti = pd.Series(pd.date_range("2020-04-01", freq="D", periods=365)).dt.date

    idx = pd.MultiIndex.from_product([dti, np.arange(24)], names=['date', 'hour'])

    # feature columns would need to be added below
    df_blank = pd.DataFrame(data = np.zeros(shape=(365*24, 1)), index=idx,
                            columns=['target'])

    out = pd.concat([df_blank, out]).groupby(['date', 'hour']).agg('sum')

    out = out.reset_index(drop=True)

    # If data is univariate, no need for scaling
    if out.shape[1] == 1:
        return out

    # target feature should not be scaled
    y = np.array(out.iloc[:, -1])

    scaler = MinMaxScaler()

    out = scaler.fit_transform(out.iloc[:, :-1])

    return np.append(out, y.reshape(-1, 1), axis=1), scaler

class Model:
    '''
    Wrapper class for Keras GRU type recurrent neural network.

    Includes architecture and methods to streamline model training.
    '''
    def __init__(self, df, univariate=True, load_model=None):
        '''output of station_data(region, eda=False) should be passed'''
        
        self.df = np.array(df)
                
        # offset in hours from midnight (used in predictions)
        self.offset = 6
        
        if self.df.shape[1] == 1:
            self.univariate=False
        else:
            self.univariate = univariate
                
        # scale data
        self.scaler = MinMaxScaler()
        self.df = self.scaler.fit_transform(self.df)
        
        if self.univariate == True:
            self.X_train, self.X_test, self.y_train, self.y_test = self.split_and_windowize(self.df[:, -1], 120, 0.0001, univariate=self.univariate)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = self.split_and_windowize(self.df, 120, 0.0001, univariate=self.univariate)

        if load_model is not None:
            self.model = load_model
            return None    

        # Model structure
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.GRU(100, return_sequences=False))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(1, activation='relu'))
        self.model.compile(optimizer='rmsprop', loss='mse')
    
    def windowize_data(self, data, n_prev, univariate=True):
        '''Function to add a dimension of past data points to a numpy array
        
        input: 2D np array
        
        output: 3d np array (where 3D dimension is just copies of previous rows)
        
        Adapted from code by Michelle Hoogenhout:
        https://github.com/michellehoog
        '''
        n_predictions = len(data) - n_prev
        indices = np.arange(n_prev) + np.arange(n_predictions)[:, None]

        if univariate == False:
            y = data[n_prev:, -1]
            x = data[indices]
        else:
            y = data[n_prev:]
            x = data[indices, None]
        return x, y

    def split_and_windowize(self, data, n_prev, fraction_test=0.1, univariate=True):
        '''Train/test splits data with added timestep dimension
        
        Adapted from code by Michelle Hoogenhout:
        https://github.com/michellehoog
        '''
        n_predictions = len(data) - 2*n_prev

        n_test  = int(fraction_test * n_predictions)
        n_train = n_predictions - n_test   

        x_train, y_train = self.windowize_data(data[:n_train], n_prev, univariate=univariate)
        x_test, y_test = self.windowize_data(data[n_train:], n_prev, univariate=univariate)
        return x_train, x_test, y_train, y_test

    def train(self):
        '''Actually trains the model on the data'''
        self.model.fit(self.X_train, self.y_train, batch_size=16, epochs=50)
    
    def predict(self, n_out=24, offset=0):
        '''Makes a prediction
        
        offset is hours since March 27th, 2021 at 12am
        '''

        # first state of window
        window = self.X_test[0+offset, :, :].reshape([1,-1,1])

        out = []

        for _ in range(n_out):
            pred = self.model.predict(window)[0][0]

            out.append(pred)

            # add prediction as newest element to window, auto reshapes to (n+1, )
            window = np.append(window, pred)

            # delete oldest element at beginning
            window = np.delete(window, 0)

            # reshape so prediction() can use the window again
            window = window.reshape([1,-1,1])

        return np.array(out)

    def predict_plot(self, n_out=24, offset=0):
        '''Generates a plot of the prediction against the actual observations.
        
        Includes a subplot of the residuals'''

        yhat = self.predict(n_out=n_out, offset=offset)

        ytest = self.y_test[0+offset:n_out+offset]
        
        # unscale target
        yhat = self.scaler.inverse_transform(yhat.reshape(-1, 1))

        ytest = self.scaler.inverse_transform(ytest.reshape(-1, 1))

        fig, (ax1, ax2) = plt.subplots(2, figsize=(12,7), gridspec_kw={'height_ratios': [2, 1]})
        
        # xtick label if blocks
        if n_out == 24:
            ax1.set_xticks([0,6,12,18,24])
            ax2.set_xticks([0,6,12,18,24])
            if offset%24 == 0:
                ax2.set_xticklabels(['12am', '6am', '12pm', '6pm', '12am'])
            elif offset%24 == 6:
                ax2.set_xticklabels(['6am', '12pm', '6pm', '12am', '6am'])
            elif offset%24 == 12:
                ax2.set_xticklabels(['12pm', '6pm', '12am', '6am', '12pm'])
            elif offset%24 == 18:
                ax2.set_xticklabels(['6pm', '12am', '6am', '12pm', '6pm'])
        
        ax1.plot(np.arange(len(ytest)), ytest, c='darkslategrey', label='actual')
        ax1.plot(np.arange(len(ytest)), yhat, c='orangered', label='predicted')
        ax1.set_ylabel('Trips')
        ax1.title.set_text('Traffic predictions')
        ax1.set_xticklabels([])
        ax1.legend()

        ax2.plot(np.arange(len(ytest)), (ytest - yhat), c='darkslategrey')
        ax2.axhline(c='orangered')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Trip Error')
        ax2.set_ylim(-100, 100) # remove if doesn't look good
        ax2.title.set_text('Error between Actual and Predicted Traffic');
    
    def predict_score(self, n_out=24, offset=0):
        '''Returns a tuple of model and baseline RMSE scores for a window'''
        
        if self.univariate == True:
            ybase = np.ones(n_out) * self.X_train[(n_out+offset)*-1:-1-offset, 0, -1].mean()
        else:
            ybase = np.ones(n_out) * self.X_train[(n_out+offset)*-1:-1-offset,:,:].mean()

        if n_out==1:
            # this line throws runtime errors
            ybase = np.ones(n_out) * self.X_train[(n_out+offset)*-1:,:,:].mean()
        
        ybase = mean_squared_error(self.y_test[offset:offset+n_out], ybase)**0.5
        
        yhat = self.predict(n_out=n_out, offset=offset)
        
        yhat = mean_squared_error(self.y_test[0+offset:n_out+offset], yhat)**0.5
        
        #print('This model did {}% better than baseline ({})'.format(round((1-yhat/ybase)*100, 2), round(ybase, 2)))
        
        return yhat, ybase
    
    def rmse_spread(self):
        '''Gives a couple different views of RMSE to evaluate a model.
        
        Mostly used for model validation.
        '''
        
        rmse_24x1 = self.predict_score(n_out=24, offset=0)
        
        hat = []
        base = []
        for off in [0, 6, 12, 16]:
            a, b = self.predict_score(n_out=6, offset=off)
            hat.append(a)
            base.append(b)
        
        rmse_6x4 = (np.array(hat).mean(), np.array(base).mean())
        
        hat = []
        base = []
        for off in np.arange(24):
            a, b = self.predict_score(n_out=1, offset=off)
            hat.append(a)
            base.append(b)
        
        rmse_1x24 = (np.array(hat).mean(), np.array(base).mean())
        
        print('Single 24hr test: {} vs baseline {}'.format(round(rmse_24x1[0], 4), round(rmse_24x1[1], 4)))
        print('Four 6hr tests (averaged): {} vs baseline {}'.format(round(rmse_6x4[0], 4), round(rmse_6x4[1], 4)))
        print('Twenty-four 1hr tests (averaged): {} vs baseline {}'.format(round(rmse_1x24[0], 4), round(rmse_1x24[1], 4)))