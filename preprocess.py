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


def handle_commas(df):
	# There have been rows where the numbers contain ',' -> 8,573.98 instead of 8573.98
	# remove these commas
	cols = list(df.columns)
	cols = cols[1:] # ignore date column

	for col in cols:
	    try:
	        df[col] = df[col].astype(float)

	    except:       # there are errors, such as valueError coming from the ','
	        df[col] = df[col].str.replace(",", "")
	        df[col] = df[col].astype(float)

	return df

def log_scale_df(df, small_num):
	"""
	This function futhure scales the input into log scale. -> as recommended for cryptocurrency and stock (by professor)
	"""
	df = df.mask(df == 0).fillna(small_num) # replace all the 0's with a really small number: input (10^-10)
	# print("changing all the 0s")
	# print(df.head())
	log_df = np.log(df)

	return log_df


def unscale_log(input_list, feature_list):
	# print(input_list)
	scaled_df = pd.DataFrame.from_records(input_list)
	scaled_df = scaled_df.transpose()

	# remove the np array inside each dataframe cell
	scaled_df = scaled_df.applymap(lambda x: x[0] if isinstance(x, np.ndarray) else x) 
	unscaled_df = np.exp(scaled_df) # assuming log scale
	unscaled_df.columns = feature_list

	return unscaled_df




def scale_df(df, scaler):
	"""
	This function takes in a dataframe and a scaler (minmax/ standard scaler etc.) and 
	handle the data depending on the scaler passed in.

	Parameter: df -> dataframe to be processed
				scaler -> min-max scaler/standard scaler etc. depending on need
	Return: df -> processed dataframe
	"""
	cols = df.columns[:] #ignore the first column: date
	df_scaled = pd.DataFrame(scaler.fit_transform(df[cols]), index=df.index, columns=cols)

	# add the date back
	# df_scaled.insert(0, 'Date', df['Date'])
	# df_scaled.head()
	return df_scaled


def unscale_minMax(input_list, original_df, feature_list):
	scaled_df = pd.DataFrame.from_records(input_list)
	scaled_df = scaled_df.transpose()

	new_scaler = MinMaxScaler()
	new_scaler.fit(original_df[feature_list])
	unscaled_df = pd.DataFrame(new_scaler.inverse_transform(scaled_df), columns = feature_list)

	return unscaled_df




def process_main(startDate, endDate, filePath, min_max_scale, log_scale, small_num):
	# TODO: add more files to run

	print("Data preprocessing")
	cur_df = pd.read_csv(filePath)

	cur_df['Date']= pd.to_datetime(cur_df['Date'])  # set all the dataframe to start and end from the same time
	mask = (cur_df['Date'] >= startDate) & (cur_df['Date'] <= endDate)
	cur_df = cur_df.loc[mask]
	print(cur_df.head())

	checkedCur_df = test_nan(cur_df) # checks for NaN's
	checkedCur_df = handle_commas(checkedCur_df) # checks for , in entries

	# convert the input feature types to floats, or else it won't work with the training
	checkedCur_df = checkedCur_df.apply(pd.to_numeric)
	checkedCur_df = checkedCur_df.set_index('Date')
	checkedCur_df.index = pd.to_datetime(checkedCur_df.index, unit='ns')

	scaler = MinMaxScaler()
	if min_max_scale:
		print("Scaling dataframe to 0-1...")
		newScaled_df = scale_df(checkedCur_df, scaler)

	if log_scale:
		print("Scaling dataframe with log transform...")
		newScaled_df = log_scale_df(checkedCur_df, small_num)

	print(newScaled_df.head())
	print("...and done! \n")
		
	return newScaled_df, scaler, checkedCur_df






