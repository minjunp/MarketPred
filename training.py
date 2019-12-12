import json
import requests
import pandas as pd
import sys
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, AveragePooling1D, BatchNormalization, Activation, LSTM, Dropout, concatenate
from tensorflow import keras



"""get 200 days from 2012–10–10 until today
we can print each column by using . """

"""split into train and test data"""
def train_test_split(df, test_size=0.1):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data

def line_plot(line1, line2, label1=None, label2=None, title=''):
    fig, ax = plt.subplots(1, figsize=(16, 9))
    ax.plot(line1, label=label1, linewidth=2)
    ax.plot(line2, label=label2, linewidth=2)
    ax.set_ylabel('price [USD]', fontsize=14)
    ax.set_title(title, fontsize=18)
    ax.legend(loc='best', fontsize=18)
    plt.show()

"""visualize how dataset is split"""
#line_plot(train.close, test.close, 'training', 'test', 'BTC')

"""building a model
For training the LSTM, the data was split into windows of 7 days
(this number is arbitrary, I simply chose a week here)
and within each window I normalised the data to zero base,
i.e. the first entry of each window is 0 and all other values
represent the change with respect to the first value. Hence,
I am predicting price changes, rather than absolute price."""

def normalise_zero_base(df):
    """ Normalise dataframe column-wise to reflect changes with
        respect to first entry.
    """
    return df / df.iloc[0] - 1

def extract_window_data(df, window=7, zero_base=False):
    """ Convert dataframe to overlapping sequences/windows of
        length `window`.
    """
    window_data = []
    for idx in range(len(df) - window):
        tmp = df[idx: (idx + window)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)

def prepare_data(df, window=7, zero_base=False, test_size=0.1):
    """ Prepare data for LSTM. """
    # train test split
    # window size = len(test_size)
    train_data, test_data = train_test_split(df, test_size)

    # extract window data
    X_train = extract_window_data(train_data, window, zero_base)
    X_test = extract_window_data(test_data, window, zero_base)

    y_train_close = train_data.Close[window:].values
    y_test_close = test_data.Close[window:].values
    y_train_high = train_data.High[window:].values
    y_test_high = test_data.High[window:].values
    y_train_low = train_data.Low[window:].values
    y_test_low = test_data.Low[window:].values
    y_train_open = train_data.Open[window:].values
    y_test_open = test_data.Open[window:].values


    if zero_base:
        y_train_close = y_train_close / train_data.close[:-window].values - 1
        y_test_close = y_test_close / test_data.close[:-window].values - 1
        y_train_high = y_train_high / train_data.high[:-window].values - 1
        y_test_high = y_test_high / test_data.high[:-window].values - 1
        y_train_low = y_train_low / train_data.low[:-window].values - 1
        y_test_low = y_test_low / test_data.low[:-window].values - 1
        y_train_open = y_train_open / train_data.open[:-window].values - 1
        y_test_open = y_test_open / test_data.open[:-window].values - 1

    return train_data, test_data, X_train, X_test, y_train_close, y_test_close, y_train_high, y_test_high, y_train_low, y_test_low, y_train_open, y_test_open

#train, test, X_train, X_test, y_train_close, y_test_close, y_train_high, y_test_high, y_train_low, y_test_low, y_train_open, y_test_open = prepare_data(hist)
# zero_bast = False
# train, test, X_train, X_test, y_train_close, y_test_close, y_train_high, y_test_high, y_train_low, y_test_low, y_train_open, y_test_open = prepare_data(hist, zero_base=False)

"""
def build_lstm_model_ver1(input_data, output_size, neurons=20,
                     activ_func='linear', dropout=0.25,
                     loss='mae', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer)
    return model

model = build_lstm_model_ver1(X_train, output_size=1)
#history = model.fit(X_train, y_train_close, epochs=50, batch_size=4)

preds = model.predict(X_test).squeeze()
#print('**************************')
#print(preds)
#sys.exit()
#line_plot(train.close, test.close, 'training', 'test', 'BTC')
#sys.exit()

input_data = hist
"""

# def create_model():
#     first_input = keras.Input(shape=(7,2), name = 'first_input')
#     second_input = keras.Input(shape=(7,2), name = 'second_input')

#     first_layer = LSTM(20)
#     #first_layer = Conv1D(filters=30, kernel_size=2, data_format='channels_last', input_shape=(7,2))

#     first = first_layer(first_input)
#     second = first_layer(second_input)
#     x = concatenate([first, second])
#     #x = Dropout(0.25)(x)
#     output1 = Dense(1, activation='linear', name = 'output1')(x)
#     output2 = Dense(1, activation='linear', name = 'output2')(x)

#     model = keras.Model(inputs=[first_input, second_input], outputs=[output1, output2])
#     #print model summary
#     model.summary()

#     #plot the models
#     #keras.utils.plot_model(model, 'multi_input_and_output_model.png', show_shapes=True)

#     # for single outputs
#     #model.compile(loss='mae', optimizer='adam')
#     # for two outputs
#     model.compile(optimizer='adam', loss={'output1': 'mae', 'output2': 'mae'}, loss_weight=[1., 0.2])

#     return model


def create_model():
    first_input = keras.Input(shape=(7,2), name = 'first_input')
    second_input = keras.Input(shape=(7,2), name = 'second_input')

    first_layer_1 = LSTM(256)
    first_layer_2 = LSTM(256)
    #first_layer = Conv1D(filters=30, kernel_size=2, data_format='channels_last', input_shape=(7,2))

    first = first_layer_1(first_input)
    second = first_layer_2(second_input)

    x = concatenate([first, second])

    x = Dropout(0.21)(x)

    output1 = Dense(1, activation='linear', name = 'output1')(x)
    output2 = Dense(1, activation='linear', name = 'output2')(x)

    model = keras.Model(inputs=[first_input, second_input], outputs=[output1, output2])
    #print model summary
    model.summary()

    #plot the models
    #keras.utils.plot_model(model, 'nn_model.png', show_shapes=True)

    # for single outputs
    #model.compile(loss='mae', optimizer='adam')
    # for two outputs
    model.compile(optimizer='adam', loss={'output1': 'mse', 'output2': 'mse'}, loss_weight=[1., 1.])

    return model


def training_main(passed_df, if_stock):
    """
    Entry point to all the functions in this file. Used for training.
    """
    print("Training on the passed in dataset...")

    register_matplotlib_converters()
    # endpoint = 'https://min-api.cryptocompare.com/data/histoday'
    # # n days interval
    # res = requests.get(endpoint + '?fsym=BTC&tsym=USD&limit=200')
    # hist = pd.DataFrame(json.loads(res.content)['Data'])
    # print(hist.head())
    # hist = hist.set_index('time')
    # hist.index = pd.to_datetime(hist.index, unit='s')
    # print(hist.dtypes)
    # print(hist.head())

    hist = passed_df
    model = create_model()
    train, test = train_test_split(hist, test_size=0.1)

    # if if_percentage_change:
    #     train = train.pct_change()
    #     test = test.pct_change()
    #     print("this is train after the percentage change!!!")

    # print(train.head())
    # print(test.head())
    train, test, X_train, X_test, y_train_close, y_test_close, y_train_high, y_test_high, y_train_low, y_test_low, y_train_open, y_test_open = prepare_data(hist, zero_base=False)

    x1_train = X_train[:,:,[0,3]]
    x2_train = X_train[:,:,[1,2]]
    x1_test = X_test[:,:,[0,3]]
    x2_test = X_test[:,:,[1,2]]

    #history = model.fit({'first_input': x1_train, 'second_input': x2_train}, y_train_low, epochs=10, batch_size=4, validation_split=0.1)
    if not if_stock:
        model.fit({'first_input': x1_train, 'second_input': x2_train}, {'output1': y_train_low,'output2': y_train_high}, epochs=30, batch_size=200, validation_split=0.1)
        print('********************')
    # sys.exit()
    history2 = model.evaluate({'first_input': x1_test, 'second_input': x2_test}, {'output1': y_test_low,'output2': y_test_high})
    pred = model.predict({'first_input': x1_test, 'second_input': x2_test})

    #targets = test[target_col][window:]
    # preds = model.predict(X_test).squeeze()
    # history2 = model.evaluate(X_test, y_test)

    # print(pred)
    # print(history2)
    """
    we are reporting the loss - mean squared error - and this is very low, meaning
    predicting very well..
    """
    print("...and done!")
    return pred, history2, test.index, model


