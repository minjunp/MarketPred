# This file is used for hyperparameter tuning

from hyperopt import tpe
from pandas.plotting import register_matplotlib_converters
from tensorflow.keras.layers import Conv1D,MaxPooling1D,Flatten,Dense,AveragePooling1D,BatchNormalization,Activation, \
    LSTM,Dropout,concatenate
from tensorflow import keras
from hyperopt import fmin
from hyperopt import hp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def test_nan(df):
    """
    This function tests if there are any NaNs' inside the passed in dataframe, and if so,
    replace it with the column mean.

    Parameter: df -> dataframe to be checked
    Return: df -> dataframe with NaN's handled
    """
    print("Is there any NaN in the passed in df: ",df.isnull().values.any(),"\n")
    if df.isnull().values.any():
        null_columns=df.columns[df.isnull().any()]
        print("Column and number of NaNs' in the df:")
        print(df[null_columns].isnull.sum())
        print(df[df.isnull().any(axis=1)][null_columns].head(),"\n\n")

        print("Before cleaned dataframe shape",df.shape)
        # Remove NaNs' from the df
        processed_df=df.fillna(df.mean())
        return processed_df

    else:
        return df


def handle_commas(df):
    # There have been rows where the numbers contain ',' -> 8,573.98 instead of 8573.98
    # remove these commas
    cols=list(df.columns)
    cols=cols[1:]  # ignore date column

    for col in cols:
        try:
            df[col]=df[col].astype(float)

        except:  # there are errors, such as valueError coming from the ','
            df[col]=df[col].str.replace(",","")
            df[col]=df[col].astype(float)

    return df


def log_scale_df(df,small_num):
    """
	This function futhure scales the input into log scale. -> as recommended for cryptocurrency and stock (by professor)

	"""
    df=df.mask(df == 0).fillna(small_num)
    log_df=np.log(df)

    return log_df


def unscale_log(input_list,feature_list):
    # print(input_list)
    scaled_df=pd.DataFrame.from_records(input_list)
    scaled_df=scaled_df.transpose()

    # remove the np array inside each dataframe cell
    scaled_df=scaled_df.applymap(lambda x: x[0] if isinstance(x,np.ndarray) else x)
    unscaled_df=np.exp(scaled_df)  # assuming log scale
    unscaled_df.columns=feature_list

    return unscaled_df


def scale_df(df,scaler):
    """
    This function takes in a dataframe and a scaler (minmax/ standard scaler etc.) and
    handle the data depending on the scaler passed in.

    Parameter: df -> dataframe to be processed
                scaler -> min-max scaler/standard scaler etc. depending on need
    Return: df -> processed dataframe
    """
    cols=df.columns[:]  # ignore the first column: date
    df_scaled=pd.DataFrame(scaler.fit_transform(df[cols]),index=df.index,columns=cols)

    # add the date back
    # df_scaled.insert(0, 'Date', df['Date'])
    # df_scaled.head()
    return df_scaled


def unscale_minMax(input_list,original_df,feature_list):
    scaled_df=pd.DataFrame.from_records(input_list)
    scaled_df=scaled_df.transpose()

    new_scaler=MinMaxScaler()
    new_scaler.fit(original_df[feature_list])
    unscaled_df=pd.DataFrame(new_scaler.inverse_transform(scaled_df),columns=feature_list)

    return unscaled_df


def process_main(startDate,endDate,filePath,min_max_scale,log_scale,small_num):
    # TODO: add more files to run

    print("Data preprocessing")
    cur_df=pd.read_csv(filePath)

    cur_df['Date']=pd.to_datetime(cur_df['Date'])  # set all the dataframe to start and end from the same time
    mask=(cur_df['Date'] >= startDate) & (cur_df['Date'] <= endDate)
    cur_df=cur_df.loc[mask]
    print(cur_df.head())

    checkedCur_df=test_nan(cur_df)  # checks for NaN's
    checkedCur_df=handle_commas(checkedCur_df)  # checks for , in entries

    # convert the input feature types to floats, or else it won't work with the training
    checkedCur_df=checkedCur_df.apply(pd.to_numeric)
    checkedCur_df=checkedCur_df.set_index('Date')
    checkedCur_df.index=pd.to_datetime(checkedCur_df.index,unit='ns')

    scaler=MinMaxScaler()
    if min_max_scale:
        print("Scaling dataframe to 0-1...")
        newScaled_df=scale_df(checkedCur_df,scaler)

    if log_scale:
        print("Scaling dataframe with log transform...")
        newScaled_df=log_scale_df(checkedCur_df,small_num)

    print(newScaled_df.head())
    print("...and done! \n")

    return newScaled_df,scaler,checkedCur_df


# In[37]:


def train_test_split(df,test_size=0.1):
    split_row=len(df)-int(test_size*len(df))
    train_data=df.iloc[:split_row]
    test_data=df.iloc[split_row:]
    return train_data,test_data


def line_plot(line1,line2,label1=None,label2=None,title=''):
    fig,ax=plt.subplots(1,figsize=(16,9))
    ax.plot(line1,label=label1,linewidth=2)
    ax.plot(line2,label=label2,linewidth=2)
    ax.set_ylabel('price [USD]',fontsize=14)
    ax.set_title(title,fontsize=18)
    ax.legend(loc='best',fontsize=18)
    plt.show()



def normalise_zero_base(df):
    """ Normalise dataframe column-wise to reflect changes with
        respect to first entry.
    """
    return df/df.iloc[0]-1


def extract_window_data(df,window,zero_base=True):
    """ Convert dataframe to overlapping sequences/windows of
        length `window`.
    """
    window_data=[]
    for idx in range(len(df)-window):
        tmp=df[idx: (idx+window)].copy()
        if zero_base:
            tmp=normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)


def prepare_data(df,window,test_size,zero_base):
    """ Prepare data for LSTM. """
    # train test split
    train_data,test_data=train_test_split(df,test_size)

    # extract window data

    X_train=extract_window_data(train_data,window,zero_base)
    X_test=extract_window_data(test_data,window,zero_base)

    y_train_close=train_data.Close[window:].values
    y_test_close=test_data.Close[window:].values
    y_train_high=train_data.High[window:].values
    y_test_high=test_data.High[window:].values
    y_train_low=train_data.Low[window:].values
    y_test_low=test_data.Low[window:].values
    y_train_open=train_data.Open[window:].values
    y_test_open=test_data.Open[window:].values

    if zero_base:
        y_train_close=y_train_close/train_data.Close[:-window].values-1
        y_test_close=y_test_close/test_data.Close[:-window].values-1
        y_train_high=y_train_high/train_data.High[:-window].values-1
        y_test_high=y_test_high/test_data.High[:-window].values-1
        y_train_low=y_train_low/train_data.Low[:-window].values-1
        y_test_low=y_test_low/test_data.Low[:-window].values-1
        y_train_open=y_train_open/train_data.Open[:-window].values-1
        y_test_open=y_test_open/test_data.Open[:-window].values-1

    return train_data,test_data,X_train,X_test,y_train_close,y_test_close,y_train_high,y_test_high,y_train_low,y_test_low,y_train_open,y_test_open


def create_model(nHidden,lossFunc,optimizer,window_len,dropout,activ):
    first_input=keras.Input(shape=(window_len,2),name='first_input')
    second_input=keras.Input(shape=(window_len,2),name='second_input')

    first_layer_1=LSTM(nHidden)
    first_layer_2=LSTM(nHidden)
    # first_layer = Conv1D(filters=30, kernel_size=2, data_format='channels_last', input_shape=(7,2))

    first=first_layer_1(first_input)
    second=first_layer_2(second_input)

    x=concatenate([first,second])

    x=Dropout(dropout)(x)

    output1=Dense(1,activation=activ,name='output1')(x)
    output2=Dense(1,activation=activ,name='output2')(x)

    model=keras.Model(inputs=[first_input,second_input],outputs=[output1,output2])
    # print model summary
    model.summary()

    # plot the models
    # keras.utils.plot_model(model, 'nn_model.png', show_shapes=True)

    # for single outputs
    # model.compile(loss='mae', optimizer='adam')
    # for two outputs
    model.compile(optimizer=optimizer,loss={'output1': lossFunc,'output2': lossFunc},loss_weight=[1.,1.])

    return model


parameter=dict(nHiddenNeurons=hp.choice('nHiddenNeurons',[64,128,256]),
               optimizer=hp.choice('optimizer',['sgd','Adam']),
               lossFunc=hp.choice('lossFunc',['mae','mse']),
               batch_size=hp.choice('batch_size',[100,200,500,1000]),
               window_len=hp.choice('window_len',[3,7,10]),
               epochs=hp.choice('epochs',[10,20,30]),
               dropout=hp.choice('dropout',[0.25,0.21]),
               activ=hp.choice('activ',['tanh','sigmoid','relu']))

validation_test_size=0.1


def training_main(passed_df):
    """
    Entry point to all the functions in this file. Used for training.
    Mainly calling the hyperparameter objective function
    """

    register_matplotlib_converters()
    hist=passed_df
    def objective(param):
        train,test,X_train,X_test,y_train_close,y_test_close,y_train_high,y_test_high,y_train_low,y_test_low,y_train_open,y_test_open= \
            prepare_data(hist,param['window_len'],validation_test_size,zero_base=False)
        x1_train=X_train[:,:,[0,3]]
        x2_train=X_train[:,:,[1,2]]
        x1_test=X_test[:,:,[0,3]]
        # print(x1_test)
        x2_test=X_test[:,:,[1,2]]
        model=create_model(param['nHiddenNeurons'],param['lossFunc'],
                           param['optimizer'],param['window_len'],param['dropout'],param['activ'])
        model.fit({'first_input': x1_train,'second_input': x2_train},{'output1': y_train_low,'output2': y_train_high},
                  epochs=param['epochs'],batch_size=param['batch_size'],validation_split=validation_test_size,verbose=0)
        history2=model.evaluate({'first_input': x1_test,'second_input': x2_test},
                                {'output1': y_test_low,'output2': y_test_high})

        return history2[0]

    # calling the hyperopt function and setting the maximum iteration to 100
    best_params=fmin(objective,
                     parameter,
                     algo=tpe.suggest,
                     max_evals=100)

    print(best_params)
    return best_params


# MAIN PROCESS CALLED HERE

startDate='2015-10-10'
endDate='2019-11-19'
small_num = 10**-10 # really small number to handle the 0's before log-scaling
if_minmax_scale = False
if_log_scale = True
if_percent_change = True
if_stock = False

# Read & Transform csv files into dataframes for preprocessing
bitcoin_daily_path="data/cryptocurrency/daily(2015-2019)/bitcoin_daily_10102015-11232019.csv"
bitcoin_hourly_path="data/cryptocurrency/hourly(2019)/bitcoin_hourly_2019.csv"
etherum_daily_path="data/cryptocurrency/daily(2015-2019)/Etherum_daily_08172015-11232019.csv"
etherum_hourly_path="data/cryptocurrency/hourly(2019)/Etherum_hourly_2019.csv"

# Stock
stock_daily_path="data/stock/SPDaily101015.csv"

filePath=[bitcoin_daily_path,etherum_daily_path]  # change this file name to other files

for i in range(len(filePath)):
    scaled_df, scaler, original_df =process_main(startDate, endDate, filePath[i], if_minmax_scale, if_log_scale, small_num)
    print(filePath[i])
    training_main(scaled_df)
