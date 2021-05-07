# This file contains RNN model architecture used in model.ipynb
# To see examples of how to use this model, see said notebook
# Authored by Nicholas Kopystynsky

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

plt.style.use('ggplot')

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
        
        # window of time to look back when making a prediction (in hours)
        self.lookback = 120
        
        if self.df.shape[1] == 1:
            self.univariate=False
        else:
            self.univariate = univariate
                
        # scale data
        if self.df.shape[1] == 1:
            self.scaler = MinMaxScaler()
            self.df = self.scaler.fit_transform(self.df)
        else:
            self.scaler = MinMaxScaler()
            
            y = np.array(self.df.iloc[:, -1])
            
            X = self.scaler.fit_transform(self.df[:, :-1])
            
            self.df = np.append(X, y.reshape(-1, 1), axis=1)
        
        if self.univariate == True:
            self.X_train, self.X_test, self.y_train, self.y_test = self.split_and_windowize(self.df[:, -1], self.lookback, 0.0001, univariate=self.univariate)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = self.split_and_windowize(self.df, self.lookback, 0.0001, univariate=self.univariate)

        if load_model is not None:
            self.model = load_model
            return None    

        # Model structure, feel free to make adjustments here
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
        '''Gives a couple different views of RMSE scores to evaluate a model.
        
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