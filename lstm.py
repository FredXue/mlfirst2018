## This is a lstm model for web traffic prediction. 
## author Jiayi Gao

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from math import sqrt


# create a differenced series
def difference(dataset, interval=1):
    row,col = dataset.shape
    diff = []
    for i in range(interval, col):
        value = dataset[:,i] - dataset[:,(i-interval)]
        diff.append(value)
    return np.asarray(diff).T

# transform data to supervised with feature matrix X and predictions Y
def transform_X_Y(data, n_seq):
    X = data[:,0:-n_seq]
    Y = data[:,-n_seq:]
    return X,Y

# transform data into train and test sets for supervised learning
def prepare_data(data, n_seq):
    # extract raw values
    raw_values = data.values
    # transform data to be stationary
    diff_values = difference(raw_values, 1)
    # rescale values to -1, 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(diff_values)
    # transform into supervised learning problem X, y
    X,Y = transform_X_Y(scaled_values, n_seq)
    # split into train and test sets
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.25)
    # reshape validation data to [samples, timesteps, features]
    X_val = np.reshape(X_val,(X_val.shape[0], X_val.shape[1], 1))
    return scaler, X_train, X_val, Y_train, Y_val

# fit an LSTM network to training data
def fit_lstm(data, X_train, Y_train, X_val, Y_val, n_batch, n_epoch, n_neurons, n_seq, scaler):
    # reshape training into [samples, timesteps, features]
    n_samples,timesteps = X_train.shape
    X_train = np.reshape(X_train,(n_samples,timesteps,1))
    # design network
    model = Sequential()
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X_train.shape[1], X_train.shape[2]), stateful=True))
    model.add(Dense(Y_train.shape[1], activation='tanh'))
    adam = optimizers.Adam(clipnorm=1.)
    model.compile(loss='mean_squared_error', optimizer=adam)
    # fit network
    for i in range(n_epoch):
        print('fitting model, epoch: %d' %(i+1))
        # save the model weights after each epoch if the validation loss decreased
        checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)
        model.fit(X_train, Y_train, epochs=1, batch_size=n_batch, verbose=0, shuffle=False, validation_data=(X_val, Y_val), callbacks=[checkpointer])
        print('fitted, epoch: %d' %(i+1))
        model.reset_states()
        # compute val loss every 2 epochs of training
        if i % 2 == 0:
            rmse,smape = validation(model, data, X_val, Y_val, n_batch, scaler, n_seq)
            print('Training epoch %d: val rmse:%f, val smape:%f' %((i+1),rmse,smape))
    return model

# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
    # reshape input pattern to [samples, timesteps, features]
    # n_samples,timesteps = X.shape
    # X = np.reshape(X,(n_samples, timesteps, 1))
    # make forecast
    forecast = model.predict(X, batch_size=n_batch)
    return forecast

# invert differenced forecast
def inverse_difference(last_ob, forecast):
    # invert first forecast
    inverted = []
    inverted.append(forecast[0] + last_ob)
    # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i-1])
    return inverted

# inverse data transform on forecasts
def inverse_transform(data, forecasts, scaler, n_seq):
    inverted = []
    n_samples,timesteps = forecasts.shape
    # invert scaling
    forecasts = scaler.inverse_transform(forecasts)
    forecasts = forecasts[:,-n_seq:]
    index = data.shape[1] - n_seq - 2
    for i in range(n_samples):
        # create array from forecast
        forecast = np.array(forecasts[i,:])
        forecast = forecast.squeeze()
        # invert differencing
        last_ob = data.values[i,index]
        inv_diff = inverse_difference(last_ob, forecast)
        # store
        inverted.append(inv_diff)
    return np.asarray(inverted)

# evaluate the RMSE for each forecast time step
def evaluate_forecasts(true_val, forecasts):
    rmse = sqrt(mean_squared_error(true_val, forecasts))
    # print('t+%d RMSE: %f' % ((i+1), rmse))
    return rmse

# Approximated differentiable SMAPE for one prediction
def differentiable_smape(true, predicted):
    epsilon = 0.1
    true_o = true
    pred_o = predicted
    summ = np.maximum(np.abs(true_o) + np.abs(pred_o) + epsilon, 0.5 + epsilon)
    smape = (np.abs(pred_o - true_o) / summ) * 2
    return smape

def compute_mean_smape(pred,y_test):
    smape = []
    for i in range(len(pred)):
        fore = pred[i]
        actural = y_test[i]
        sm = differentiable_smape(actural,fore)
        smape.append(sm)
    mean_sm = np.asarray(smape).mean()
    return mean_sm

def validation(model, data, X_val, Y_val, n_batch, scaler, n_seq):
    forecasts = forecast_lstm(model, X_val, n_batch)
    emp = np.zeros((forecasts.shape[0],(data.shape[1]-1-forecasts.shape[1])))
    stacked = np.column_stack((emp,forecasts))
    reverted_forecasts = inverse_transform(data, stacked, scaler, n_seq)
    reverted_forecasts[reverted_forecasts<0] = 0
    rmse = evaluate_forecasts(Y_val, reverted_forecasts)
    smape = compute_mean_smape(reverted_forecasts,Y_val)
    return rmse,smape



#read in data
print('reading data')
train = pd.read_csv("data/train_2.csv")
# key = pd.read_csv("key_2.csv")

# clean data
train = train.fillna(0)
train = train.iloc[:100000]
train = train.sample(frac=1).reset_index(drop=True)
page = train['Page']
data = train.drop('Page',axis = 1)
# X_data = data.iloc[:,0:-60]
# Y_data = data.iloc[:,-60:]

# configure
n_seq = 60
n_epochs = 10
n_batch = 10
n_neurons = 1
# prepare data
print('preparing data')
scaler, X_train, X_val, Y_train, Y_val = prepare_data(data, n_seq)

# shuffle data
# shuffled_X_train = X_train.sample(frac=1).reset_index(drop=True)
# shuffled_X_val = X_val.sample(frac=1).reset_index(drop=True)

# fit model
print('fit_lstm function')
model = fit_lstm(data, X_train, Y_train, X_val, Y_val, n_batch, n_epochs, n_neurons, n_seq, scaler)

rmse,smape = validation(model, data, X_val, Y_val, n_batch, scaler, n_seq)
print('Validation loss: rmse:%f, smape:%f' %(rmse,smape))


