import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = 18, 8
plt.style.use('fivethirtyeight')

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['text.color'] = 'k'

import warnings
warnings.filterwarnings("ignore")
#from sklearn.utils.testing import ignore_warnings
#from sklearn.exceptions import ConvergenceWarning

from scipy.stats.stats import pearsonr
from statsmodels.tsa.statespace.varmax import VARMAX

from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.statespace.sarimax import SARIMAX


from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout

from sklearn.ensemble import IsolationForest
SHOW_PLOTS = True
                                                                                                                                                        
# Make True to see feature importance graphs
SHOW_FEATURE_IMPORTANCE = True
data=pd.read_csv('Load.txt', sep='\t', header=0, names=["Timestamp", "Load", "temperature"])
data.set_index(pd.date_range(start='2016/01/01', periods=130, freq='H'), inplace=True)
data_u=data[['temperature']]
# smooth out anomalies
# smooth out anomalies
def smooth(full_data):
    for i in full_data.columns:
        if MODELS[i][0].startswith('SARIMAX'):
            full_data[i] = outlier_smoothing(full_data[i].dropna(), plot=False)
 
    return full_data


def outlier_smoothing(X, contamination=0.15, smoothing_window=4, plot=True, random_state=22, verbose=True):
    """
    Outlier identification by IForest and 
    smoothing by rolling window median value
    """
    X_rolling_median = X.rolling(smoothing_window).median()
    X_rolling_mean = X.rolling(smoothing_window).mean()
    X_smoothing_ratio = X / X_rolling_median

    if plot:
        plt.figure(figsize=(10,10))
        plt.plot(X.index, X, label='original')
        plt.plot(X.index, X_rolling_median, label='rolling median')
        plt.title("Original vs. Rolling Median")
        plt.legend()
        plt.show()

        plt.figure(figsize=(10,10))
        plt.plot(X.index, X_smoothing_ratio, label="original:smoothing ratio")
        plt.title("Smoothing Ratio")
        plt.legend()
        plt.show()
    
    ## Find the outliers
    iso_forest = IsolationForest(contamination=contamination,\
        random_state=random_state)
    peaks = np.where(iso_forest.fit_predict(X_smoothing_ratio[smoothing_window-1:].\
        values.reshape(-1,1))<1)
    if verbose:
        print("Outliers found at ", X.index[peaks[0]+smoothing_window-1])
    if plot:
        plt.figure(figsize=(10,10))
        plt.plot(X.index, X, label='original')
        plt.plot(X.index.values[peaks[0]+smoothing_window-1],\
            X.values[peaks[0]+smoothing_window-1], 'x'
            )
        plt.title("Outlier Finders")
        plt.legend()
        plt.show()
    ## Change the outliers with corresponding smoothed values    
    X_smoothed = X.copy()

    for i in range(len(X)):
        if np.any(peaks[0]+smoothing_window-1==i):
            X_smoothed[i] = X_rolling_mean[i]

    if plot:
        plt.figure(figsize=(10,10))
        plt.plot(X.index, X, label='original')
        plt.plot(X.index, X_smoothed, label='smoothed')
        plt.title("Original vs. smoothed")
        plt.legend()
        plt.show()
    
    return X_smoothed



#Quantitative Scoring using MAPE
def MAPE(gt, pred):
    mape = []

    for g, p in zip(gt, pred):
        mape.append(max(0, 1 - abs((g-p)/g)))

    return np.mean(mape)


def Max_APE(gt, pred):
    """
    Returns max absolute percentage error
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.max(np.abs((y_true - y_pred) / y_true)) * 100

def RMSE(y_true, y_pred):
    """
    Returns root mean square error
    """
    return sqrt(mean_squared_error(y_true, y_pred))


import itertools
import statsmodels.api as sm
# this function is for selection of best parameter in sarimax model , like order and seasonal_order
def best_parameters_sarimax(series, exog=None):
    '''
    Finds the best parameters for a given series for SARIMAX algorithm.
    Input: series: the series for which the parameters are to be determined.
            exog: extra features to be considered
    Output: the best parameters for the series and model.
    return (1, 1, 1), param_seasonal
    '''
    #return (1, 1, 1)
#, param_seasonal
    #param=(1,1,1)
    result_param = -1
    result_param_seasonal = -1
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(series,
                                                exog=exog,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit()
                #print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
                try:
                    if results.aic < minimum:
                        result_param = param
                        result_param_seasonal = param_seasonal
                except:
                    result_param = param
                    result_param_seasonal = param_seasonal
                    minimum = results.aic
            except:
                continue
    print(result_param, result_param_seasonal)
    return result_param, result_param_seasonal

def apply_model_sarimax(series, best_param, best_param_seasonal):

    
    mod = sm.tsa.statespace.SARIMAX(series,
                                    order=best_param,
                                    seasonal_order=best_param_seasonal,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    return results

def create_dataset(dataset, look_back=1):
    '''
    Convert an array of values into a dataset matrix
    Input: Array data
    Output: reshape into X=t, Y=t+1 (the next timestamp)
    '''
    dataset = dataset.reshape(dataset.shape[0], 1)
    
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)
#@ignore_warnings(category=ConvergenceWarning)
def cross_validation_and_model_comparison(train_df, number_of_steps_to_predict):
    print('LSTM')
    look_back = 24
    units = 50      
    trainX, trainY = create_dataset(train_df.values, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    model = Sequential()
    model.add(LSTM(units=units, return_sequences= True,input_shape=( 1,look_back)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(units=units))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=300, batch_size=20, verbose=0)

    testX = np.array([trainY[-look_back:]])

    forecast = []
    for i in range(0,number_of_steps_to_predict):
        testX = np.reshape(testX, (testX.shape[0], 1, look_back))
        testPredict = model.predict(testX)
        testX = np.roll(testX, -1)
        testX[0][0][look_back - 1] = testPredict[0][0]

        forecast.append(testPredict[0][0])

    return forecast
def make_predictions(target_data):
    train_df=target_data
    #change the number of steps of how much day you want
    number_of_steps=12
    forecast=cross_validation_and_model_comparison(train_df,number_of_steps)
    return forecast
if __name__ == '__main__':
    
    print('Reading files')
    
    print('Finding related features')
#     related_features = find_correlated_features(target_data, target_data)
    related_features = None
    
    print('Making full forecast')
    full_forecasts = make_predictions(data_u)
def make_final_data(list1):
    temp=pd.DataFrame()
    temp['time']=pd.date_range(start='2016/01/06 10:00', periods=12, freq='H')
    temp['temperature(LSTM)']=list1
    return temp
sub=make_final_data(full_forecasts)
model=sm.tsa.statespace.SARIMAX(data_u['temperature'],order=(1,1,1),seasonal_order=(1,1,2,24))
model_fit=model.fit()
ll=model_fit.predict(start=130,end=141,dynamic=False)
sub['temperature(SARIMA)']=ll
sub['temperature(SARIMA)'].plot(figsize=(16,5))
sub.to_csv('forecasted.csv', index=False)
sub.to_csv('forecasted.txt', index=False, sep=' ')