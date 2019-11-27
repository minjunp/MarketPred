import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler



def test_nan(df):
	"""
	This function tests if there are any NaNs' inside the passed in dataframe, and if so,
	replace it with the column mean.

	Parameter: df -> dataframe to be checked
	Return: df -> dataframe with NaN's handled
	"""
	print("Is there any NaN in the passed in df: ", df.isnull().values.any(), "\n")
	if df.isnull().values.any():
	    null_columns = df.columns[df.isnull().any()]
	    print("Column and number of NaNs' in the df:")
	    print(df[null_columns].isnull.sum())
	    print(df[df.isnull().any(axis = 1)][null_columns].head(), "\n\n")
	    
	    print("Before cleaned dataframe shape", df.shape)
	    # Remove NaNs' from the df
	    processed_df = df.fillna(df.mean())
	    return processed_df

	else:
		return df


def scale_df(df, scaler):
	"""
	This function takes in a dataframe and a scaler (minmax/ standard scaler etc.) and 
	handle the data depending on the scaler passed in.

	Parameter: df -> dataframe to be processed
				scaler -> min-max scaler/standard scaler etc. depending on need
	Return: df -> processed dataframe
	"""
	cols = df.columns[1:] #ignore the first column: date
	df_scaled = pd.DataFrame(scaler.fit_transform(df[cols]), columns=cols)

	# add the date back
	df_scaled.insert(0, 'Date', df['Date'])
	df_scaled.head()

	return df_scaled



def process_main(startDate, endDate):
	# TODO: add more files to run

	# Read & Transform csv files into dataframes for preprocessing
	bitcoin_daily_path = "/Users/Xiong/Desktop/Comp 576/Final project/Gemini_BTCUSD_d.csv"
	bitcoin_hourly_path = "/Users/Xiong/Desktop/Comp 576/Final project/bitcoin_hourly_2019.csv"

	filePath = [bitcoin_daily_path, bitcoin_hourly_path] # change this file name to other files

	print("Data preprocessing")
	for i in range(len(filePath)):
		cur_df = pd.read_csv(filePath[i])

		cur_df['Date']= pd.to_datetime(cur_df['Date'])  # set all the dataframe to start and end from the same time
		mask = (cur_df['Date'] > startDate) & (cur_df['Date'] <= endDate)
		cur_df = cur_df.loc[mask]

		checkedCur_df = test_nan(cur_df)

		print("Scaling dataframe to 0-1...")
		scaler = MinMaxScaler()
		newScaled_df = scale_df(checkedCur_df, scaler)
		print(newScaled_df.head())

	print("...and done!")
		

	return 





