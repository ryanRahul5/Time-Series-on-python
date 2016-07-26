# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 18:06:27 2016

@author: Rahul
"""

################ Time series analysis of futures data using pandas

import pandas as pd
import numpy as np
import matplotlib.pylab as plt


data = pd.read_csv('C:\Users\Dell\Desktop\Pairs Trading\FuturePrices.csv')
print data.ix[:10]
print data.head()

print data.dtypes
data1 = data.ix[:,['Date','YESBANK']]
print data1.head()

ts = data['YESBANK']
ts.head(10)


plt.plot(ts)

## a different approach 

data = pd.read_csv('C:\Users\Dell\Desktop\Pairs Trading\FuturePrices.csv', parse_dates = 'Date', index_col = 'Date') 

ts = data['YESBANK']
ts.head(5)

ts['2010-01-07']
ts['2011-08-21': '2011-08-25']
# parse_dates: This specifies the column which contains the date-time information.
# As we say above, the column name is ‘Month’.
#index_col: A key idea behind using Pandas for TS data is that the index has to be the variable 
#depicting date-time information. So this argument tells pandas to use the ‘Month’ column as index.

# checking for the trend
plt.plot(ts)

# overall increasing trend with cyclicity

# Checking Stationarity of Time Series

from statsmodels.tsa.stattools import adfuller  # for ADF-test

def test_stationarity(timeseries):
    
    rollmean = pd.rolling_mean(timeseries, window = 8)
    rollstd  = pd.rolling_std(timeseries, window = 8)
    
    # plot rolling stats
    
    orig = plt.plot(timeseries, color = 'red', label = 'original series')
    mean = plt.plot(rollmean, color='black', label = 'rolling mean')
    std  = plt.plot(rollstd, color = 'blue', label = 'rolling std')
    plt.legend(loc = 'best')
    plt.title('R_mean vs R_std')
    plt.show(block = False)
    
    # DF test for stationarity
    print 'Dickey-Fuller test:'
    dftest = adfuller(timeseries, autolag = 'AIC')
    dfoutput = pd.Series(dftest[0:4], index = ['t-stat','p-val', '#lags used','no. of obs'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput

    

test_stationarity(ts)

# t-stat val > critical value hence we can't reject null hypothesis (TS is non-stationary)

## how to make a TS stationary?



#The underlying principle is to model or estimate the trend and seasonality
# in the series and remove those from the series to get a stationary series.
# Then statistical forecasting techniques can be implemented on this series. 
#The final step would be to convert the forecasted values into the original
# scale by applying trend and seasonality constraints back.


# there could be many reason behind non-ts like Trend and Seasonality

#transformation to remove trend
ts_log = np.log(ts)
plt.plot(ts_log)

# to remove the presence of noise from the TS we can use some smoothing techniques


moving_avg = pd.rolling_mean(ts_log,8)
plt.plot(ts_log)
plt.plot(moving_avg, color='red')

ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.head(8)

ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)


# exponentially moving average
expwighted_avg = pd.ewma(ts_log, halflife=8) # halflife is used to define amount of exponential decay
plt.plot(ts_log)
plt.plot(expwighted_avg, color='red')    

ts_log_ewma_diff = ts_log - expwighted_avg
test_stationarity(ts_log_ewma_diff)

## here t-stat val is lesser than 1% critical value hence this gives a better result


## ELIMINATING TREND and SEASONALITY
#differencing

ts_log_diff = ts_log - ts_log.shift()  #  first order differencing 
plt.plot(ts_log_diff)

# this appears to reduce the trend drastically

ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)

# for better result we can go for 2nd and 3rd order differencing


### Decomposing  ....


from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(ts_log, freq = 100)


trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()



## we can see that the trend, 
#seasonality are separated out from data and we can model the residuals. 
#Lets check stationarity of residuals:

ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)

## forecasting using ARIMA

#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')  # ordinary least square estimation


#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

# plot PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


from statsmodels.tsa.arima_model import ARIMA


# AR model
model = ARIMA(ts_log, order=(2, 1, 0))  
results_AR = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))

# MA model
model = ARIMA(ts_log, order=(0, 1, 2))  
results_MA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_log_diff)**2))


# Combined model
model = ARIMA(ts_log, order=(2, 1, 2))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))


predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print predictions_ARIMA_diff.head()


predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print predictions_ARIMA_diff_cumsum.head()


predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()


predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))
























