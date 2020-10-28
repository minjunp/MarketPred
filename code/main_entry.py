import pandas as pd
import matplotlib.pyplot as plt
from preprocess import process_main, unscale_minMax, unscale_log
from training import training_main
import numpy as np

startDate = '2015-10-10'
endDate = '2019-11-19'
feature_list = ['High', 'Low']
window_len = 7
if_stock = False

small_num = 10**-10 # really small number to handle the 0's before log-scaling
if_minmax_scale = False		# option for using min max scaling
if_log_scale = True    # option for using logarithm scale
if_percent_change = True     # option for percentage return

# Read & Transform csv files into dataframes for preprocessing
bitcoin_daily_path = "./data/cryptocurrency/daily(2015-2019)/bitcoin_daily_10102015-11232019.csv"
ether_daily_path = "./data/cryptocurrency/daily(2015-2019)/Etherum_daily_08172015-11232019.csv"
bitcoin_hourly_path = "./data/cryptocurrency/hourly(2019)/bitcoin_hourly_2019.csv"
ether_hourly_path = "./data/cryptocurrency/hourly(2019)/Etherum_hourly_2019.csv"

# stock
stock_daily_path = "./data/stock/SPDaily101015.csv"

filePath = [bitcoin_daily_path, bitcoin_hourly_path, ether_daily_path, ether_hourly_path, stock_daily_path] # change this file name to other files
plotNames = ['bitcoin_daily', 'bitcoin_hourly', 'etherum_daily', 'etherum_hourly', 'stock_daily']

for i in range(len(filePath)):
	if i != 0:
		print()
		print("Training on the next file!!! File number:" + str(i+1))
		if i == len(filePath) -1: # last file is the stock file
			if_stock = True

	scaled_df, scaler, original_df = process_main(startDate, endDate, filePath[i], if_minmax_scale, if_log_scale, small_num)

	# begin training
	prediction, history, test_indx, model = training_main(scaled_df, if_stock)
	test_dateTime_indx = test_indx[window_len:] # 7 is the window size


	# scale prediction back to normal
	if if_minmax_scale:
		if if_stock:
			unscaled_pred_df = unscale_minMax(prediction, original_df, ['Low', 'High'])

		else:
			unscaled_pred_df = unscale_minMax(prediction, original_df, feature_list)

	if if_log_scale:
			unscaled_pred_df = unscale_log(prediction, ['Low', 'High'])
			# return log return
			if if_percent_change:
				unscaled_pred_df = unscaled_pred_df.apply(lambda x: np.log(x)-np.log(x.shift(1)))


	# add date to our prediction
	unscaled_pred_df.index = test_dateTime_indx
	print("this is our prediction......")
	print(unscaled_pred_df.head())

	# draw a line plot of pred vs. actual
	test_old_df = original_df.loc[test_dateTime_indx[0]: test_dateTime_indx[-1]]

	if if_log_scale and if_percent_change:
		test_old_df = test_old_df.apply(lambda x: np.log(x)-np.log(x.shift(1)))

	plot_df = test_old_df[['High', 'Low']].copy()
	plot_df = plot_df.rename(columns={"High": "High_truth", "Low": "Low_truth"})
	plot_df = plot_df.join(unscaled_pred_df, how='outer')

	print("----------our prediction for the features: high and low--------")
	print()
	print(plot_df.head())
	print(history)

	# plot and save the line plot
	plt.tight_layout(pad=0.4, w_pad=0.5)
	plot1 = plot_df[['High', 'High_truth']].plot.line(figsize=(8,5))
	plot1.set_title(plotNames[i]+'High', fontsize=15)
	plot1.set_ylabel("Percentage change")
	fig1 = plot1.get_figure()
	fig1.savefig(plotNames[i] + '_high_' + '.png')

	plt.tight_layout(pad=0.4, w_pad=0.5)
	plot2 = plot_df[['Low', 'Low_truth']].plot.line(figsize=(7,5))
	plot2.set_title(plotNames[i]+'Low', fontsize=15)
	plot2.set_ylabel("Percentage change")
	fig2 = plot2.get_figure()
	fig2.savefig(plotNames[i] + '_low_' +'.png')
