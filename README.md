# Predicting stock price movement
Our pipeline uses Long short-term memory (LSTM) deep learning pipeline to predict stock price movement
You can find our final report:
```
Final-Report.pdf
```
and proposal for this project:
```
Proposal.pdf
```

## Repository Structure
* data folder: stock & cryptocurrency price data
* code folder: codes used to make price prediction. 

## How to Run the code
In your terminal, navigate to this directory.
Then in the terminal, enter the following command to run everything.
```
$ python main_entry.py
```

The code is current set to use logarithm scale with percentage return, if you would like to 
use min max scaling, modify the variable ``` if_minmax_scale ``` to ``` True ```, 
``` if_log_scale ``` and ``` if_percent_change ``` to ``` False ``` inside the main_entry.py file
