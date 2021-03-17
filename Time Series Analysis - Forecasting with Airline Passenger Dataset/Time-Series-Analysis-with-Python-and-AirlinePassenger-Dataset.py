# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 09:18:25 2018
"""
# DataSet : 
# international-airline-passengers.csv

"""
# Using ARIMA Model
# Base Model 
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import matplotlib
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.ar_model import AR


warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

matplotlib.rcParams['axes.labelsize'] = 10
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['text.color'] = 'k'
matplotlib.rcParams['figure.figsize'] = 10, 7

dataset = pd.read_csv("international-airline-passengers.csv", header=0, parse_dates=[0]
                    , index_col=0, squeeze=True)

print(dataset.shape)
print(dataset.head(25))

plt.plot(dataset)
plt.show()

decomposition = sm.tsa.seasonal_decompose(dataset, model='additive')
fig = decomposition.plot()
plt.show()

# ARIMA Models
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q)) # it replaces loop for iteration
print(pdq)

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# This step is parameter Selection for our furniture’s sales ARIMA Time Series Model. 
# Our goal here is to use a “grid search” to find the optimal set of parameters 
# that yields the best performance for our model.

params = []
params_seasonal = []
AICs = []

for param in pdq:
    print(param)
    for param_seasonal in seasonal_pdq:
        print(param_seasonal)
        try:
            mod = sm.tsa.statespace.SARIMAX(dataset,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            
            params.append(param)
            params_seasonal.append(param_seasonal)
            AICs.append(results.aic)
        except:
            continue

# The above output suggests that SARIMAX(1, 1, 1)x(1, 1, 0, 12) yields the lowest AIC value 
# of 297.78. Therefore we should consider this to be optimal option.

# Get the index of minimum AIC from the list
index_min_aic = np.argmin(AICs)

# Best parameter sets for ARIMA
print(params[index_min_aic])
print(params_seasonal[index_min_aic])
print(AICs[index_min_aic])

# Fitting the ARIMA model
mod = sm.tsa.statespace.SARIMAX(dataset,
                                order=(0, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary())
print(results.summary().tables[1])
print(results.summary().tables[2])

# -------------------------------------------------------------
# Visualise model's behaviourusing built-in diagnostics
# ------------------------------------------------------------- 
results.plot_diagnostics(figsize=(9, 8))
plt.show()

# -----------------------------
# Validating forecasts
# -----------------------------
pred = results.get_prediction(start=pd.to_datetime('1955-01'), dynamic=False)

pred_ci = pred.conf_int()
ax = dataset['1955-01':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='Forecast', alpha=.7, figsize=(9, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Passengers')
plt.legend()
plt.show()

# print
print(pred.predicted_mean)
print(len(pred.predicted_mean))

print(dataset['1955-01':])
print(len(dataset['1955-01':]))

y_forecasted = pred.predicted_mean
y_truth = dataset['1955-01':]

# Evaluation of the model
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

coefficient_of_dermination = r2_score(y_truth, y_forecasted)
print("R squared: ", coefficient_of_dermination)

mae = mean_absolute_error(y_truth, y_forecasted)
print('The Mean Absolute Error of our forecasts is {}'.format(round(mae, 2)))

mse = mean_squared_error(y_truth, y_forecasted)
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))


# Visualisation
pred_uc = results.get_forecast(steps=100)
pred_ci = pred_uc.conf_int()
ax = dataset.plot(label='Observed', figsize=(9, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Passengers')
plt.legend()
plt.show()
"""


def TSF_1_using_ARMA_model():
    # ---------------------------------
    # Load packages
    # ---------------------------------
    import warnings
    warnings.filterwarnings("ignore")    
    
    import itertools
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.tsa.arima_model import ARMA
    
    # ---------------------------------
    # set plot attributes
    # ---------------------------------
    plt.style.use('fivethirtyeight')
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['xtick.labelsize'] = 10
    matplotlib.rcParams['ytick.labelsize'] = 10
    matplotlib.rcParams['text.color'] = 'k'
    matplotlib.rcParams['figure.figsize'] = 10, 7

    # ---------------------------------
    # Load Dataset
    # ---------------------------------
    dataset = pd.read_csv("international-airline-passengers.csv", 
                          header=0, parse_dates=[0],
                          index_col=0, squeeze=True)
    # print dataset
    print()
    print(dataset.shape)
    print(dataset.head(25))

    # ---------------------------------
    # Visualise Time Series Dataset
    # ---------------------------------
    # Plot Dataset
    plt.plot(dataset)
    plt.show()
    # Decompose diffentent Time Series elements e.g. trand, seasonality, Residual ... ...
    decomposition = sm.tsa.seasonal_decompose(dataset, model='additive')
    decomposition.plot()
    plt.show()

    # -------------------------------------------------
    # Grid Search for parameters - ARMA(p,q) Model 
    # -------------------------------------------------
    p = q = range(0, 4)
    pq = list(itertools.product(p, q)) # it replaces loop for iteration
    print(pq)

    params = []
    AICs = []

    print()
    for param in pq:
        try:
                mod = ARMA(dataset, order=param)
                results = mod.fit()
                
                print()
                print("Parameter values for p and q {}".format(param))
                print('ARMA{} - AIC:{}'.format(param, results.aic))
                print()
            
                params.append(param)
                AICs.append(results.aic)
        except:
                continue

    # Get the index of minimum AIC from the list
    index_min_aic = np.argmin(AICs)

    # Best parameter sets (p, q) for ARMA
    print(params[index_min_aic])
    print(AICs[index_min_aic])

    # -------------------------------------------------
    # Fit ARMA(p,q) Model with the best parameter sets
    # -------------------------------------------------
    model = ARMA(dataset, order = params[index_min_aic])
    results = model.fit()

    # Get summary of the model
    print(results.summary())
    print(results.summary().tables[1])
    print(results.summary().tables[2])


    # ------------------------------------------------
    # Validating forecasts from the fitted model
    # ------------------------------------------------
    pred    = results.predict(start=pd.to_datetime('1955-01'), dynamic=False)
    actual  = dataset['1955-01':].plot(label='observed')
    
    pred.plot(ax=actual, label='Forecast', alpha=.7, figsize=(9, 7))

    actual.set_xlabel('Date')
    actual.set_ylabel('Passengers')
    plt.legend()
    plt.show()

    # -------------------------------------------------------
    # Evaluating the model using different KPIs or metrics
    # -------------------------------------------------------
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
    y_forecasted    = results.predict(start=pd.to_datetime('1955-01'), dynamic=False)
    y_truth         = dataset['1955-01':]

    coefficient_of_dermination = r2_score(y_truth, y_forecasted)
    print("R squared: ", coefficient_of_dermination)

    mae = mean_absolute_error(y_truth, y_forecasted)
    print('The Mean Absolute Error of our forecasts is {}'.format(round(mae, 2)))

    mse = mean_squared_error(y_truth, y_forecasted)
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

    msle = mean_squared_log_error(y_truth, y_forecasted)
    print('The Mean Squared Log Error of our forecasts is {}'.format(round(msle, 2)))

    print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

    # -----------------------------------------------
    # Forecasts (Prediction) and Visualisation
    # -----------------------------------------------
    pred_uc = results.forecast(steps=100)
    plt.plot(pred_uc[0])
    plt.plot(pred_uc[2])
    plt.show()
    
    print()
    print("Forecasted Values: ")
    print(pred_uc[0])
    
    # Visualise forecasts
    ax = dataset.plot(label='Observed', figsize=(9, 7))
    results.plot_predict('1961', '1968', dynamic=True, ax=ax,  plot_insample=False)    
    ax.set_xlabel('Date')
    ax.set_ylabel('Passengers')
    plt.legend()
    plt.show()

#TSF_1_using_ARMA_model()



def TSF_2_using_ARIMA_model():
    # ---------------------------------
    # Load packages
    # ---------------------------------
    import warnings
    warnings.filterwarnings("ignore")    
    
    import itertools
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.tsa.arima_model import ARIMA
    from pandas.tools.plotting import autocorrelation_plot
    
    # ---------------------------------
    # set plot attributes
    # ---------------------------------
    plt.style.use('fivethirtyeight')
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['xtick.labelsize'] = 10
    matplotlib.rcParams['ytick.labelsize'] = 10
    matplotlib.rcParams['text.color'] = 'k'
    matplotlib.rcParams['figure.figsize'] = 10, 7

    # ---------------------------------
    # Load Dataset
    # ---------------------------------
    dataset = pd.read_csv("international-airline-passengers.csv", 
                          header=0, parse_dates=[0],
                          index_col=0, squeeze=True)
    # print dataset
    print()
    print(dataset.shape)
    print(dataset.head(25))

    # ---------------------------------
    # Visualise Time Series Dataset
    # ---------------------------------
    # Plot Dataset
    plt.plot(dataset)
    plt.show()
    # Decompose diffentent Time Series elements e.g. trand, seasonality, Residual ... ...
    decomposition = sm.tsa.seasonal_decompose(dataset, model='additive')
    decomposition.plot()
    plt.show()

    # Auto-correlation plot
    autocorrelation_plot(dataset)
    plt.show()

    # -------------------------------------------------
    # Grid Search for parameters - ARMA(p,q) Model 
    # -------------------------------------------------
    p = d = q = range(0, 10)
    pdq = list(itertools.product(p, d, q)) # it replaces loop for iteration
    print(pdq)

    params = []
    AICs = []

    print()
    for param in pdq:
        try:
                mod = ARIMA(dataset, order=param)
                results = mod.fit()
                
                print()
                print("Parameter values for p, d and q {}".format(param))
                print('ARIMA{} - AIC:{}'.format(param, results.aic))
                print()
            
                params.append(param)
                AICs.append(results.aic)
        except:
                continue

    # Get the index of minimum AIC from the list
    index_min_aic = np.argmin(AICs)

    # Best parameter sets (p, d, q) for ARIMA
    print(params[index_min_aic])
    print(AICs[index_min_aic])

    # -------------------------------------------------
    # Fit ARIMA(p,d,q) Model with the best parameter sets
    # -------------------------------------------------
    model = ARIMA(dataset, order = params[index_min_aic])
    results = model.fit()

    # Get summary of the model
    print(results.summary())
    print(results.summary().tables[1])
    print(results.summary().tables[2])

    # plot residual errors
    residuals = pd.DataFrame(results.resid)
    residuals.plot()
    plt.show()
    residuals.plot(kind='kde')
    plt.show()
    
    print(residuals.describe())

    # ------------------------------------------------
    # Validating forecasts from the fitted model
    # ------------------------------------------------
    pred    = results.predict(start=pd.to_datetime('1955-01'), dynamic=False)
    actual  = dataset['1955-01':].plot(label='observed')
    
    pred.plot(ax=actual, label='Forecast', alpha=.7, figsize=(9, 7))

    actual.set_xlabel('Date')
    actual.set_ylabel('Passengers')
    plt.legend()
    plt.show()

    # -------------------------------------------------------
    # Evaluating the model using different KPIs or metrics
    # -------------------------------------------------------
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
    y_forecasted    = results.predict(start=pd.to_datetime('1955-01'), dynamic=False)
    y_truth         = dataset['1955-01':]

    coefficient_of_dermination = r2_score(y_truth, y_forecasted)
    print("R squared: ", coefficient_of_dermination)

    mae = mean_absolute_error(y_truth, y_forecasted)
    print('The Mean Absolute Error of our forecasts is {}'.format(round(mae, 2)))

    mse = mean_squared_error(y_truth, y_forecasted)
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

    #msle = mean_squared_log_error(y_truth, y_forecasted)
    #print('The Mean Squared Log Error of our forecasts is {}'.format(round(msle, 2)))

    print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

    # -----------------------------------------------
    # Forecasts (Prediction) and Visualisation
    # -----------------------------------------------
    pred_uc = results.forecast(steps=100)
    plt.plot(pred_uc[0])
    plt.plot(pred_uc[2])
    plt.show()
    
    print()
    print("Forecasted Values: ")
    print(pred_uc[0])
    
    # Visualise forecasts
    ax = dataset.plot(label='Observed', figsize=(9, 7))
    results.plot_predict('1961', '1968', dynamic=True, ax=ax,  plot_insample=False)    
    ax.set_xlabel('Date')
    ax.set_ylabel('Passengers')
    plt.legend()
    plt.show()

#TSF_2_using_ARIMA_model()


def TSF_3_using_holtwinters_model():
    # ---------------------------------
    # Load packages
    # ---------------------------------
    import warnings
    warnings.filterwarnings("ignore")    
    
    import itertools
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
    import statsmodels    
    import statsmodels.api as sm
    from pandas.tools.plotting import autocorrelation_plot
    
    # ---------------------------------
    # set plot attributes
    # ---------------------------------
    plt.style.use('fivethirtyeight')
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['xtick.labelsize'] = 10
    matplotlib.rcParams['ytick.labelsize'] = 10
    matplotlib.rcParams['text.color'] = 'k'
    matplotlib.rcParams['figure.figsize'] = 10, 7

    # ---------------------------------
    # Load Dataset
    # ---------------------------------
    dataset = pd.read_csv("international-airline-passengers.csv", 
                          header=0, parse_dates=[0],
                          index_col=0, squeeze=True)
    # print dataset
    print()
    print(dataset.shape)
    print(dataset.head(25))

    # ---------------------------------
    # Visualise Time Series Dataset
    # ---------------------------------
    # Plot Dataset
    plt.plot(dataset)
    plt.show()
    # Decompose diffentent Time Series elements e.g. trand, seasonality, Residual ... ...
    decomposition = sm.tsa.seasonal_decompose(dataset, model='additive')
    decomposition.plot()
    plt.show()

    # Auto-correlation plot
    autocorrelation_plot(dataset)
    plt.show()

    # -------------------------------------------------
    # Grid Search for parameters - ARMA(p,q) Model 
    # -------------------------------------------------

    trend = ["add", "mul"]
    damped = [True, False]
    seasonal = ["add", "mul"]
    
    pdq = list(itertools.product(trend, damped, seasonal)) # it replaces loop for iteration
    print(pdq)

    params = []
    AICs = []

    print()
    for param in pdq:
        try:
            mod = statsmodels.tsa.holtwinters.ExponentialSmoothing(dataset, 
                        trend=param[0], damped=param[1], seasonal=param[2], 
                        seasonal_periods=12)
            results = mod.fit()
                
            print()
            print("Parameter values for trend, damped and seasonal {}".format(param))
            print('holtwinters.ExponentialSmoothing{} - AIC:{}'.format(param, results.aic))
            print()
            
            params.append(param)
            AICs.append(results.aic)
        except:
                continue

    # Get the index of minimum AIC from the list
    index_min_aic = np.argmin(AICs)

    # Best parameter sets (trend, dumped, seasonal) for ExponentialSmoothing
    print(params[index_min_aic])
    print(AICs[index_min_aic])

    # ----------------------------------------------------------------
    # Fit ExponentialSmoothing Model with the best parameter sets
    # ----------------------------------------------------------------
    model = statsmodels.tsa.holtwinters.ExponentialSmoothing(dataset, 
                        trend     =params[index_min_aic][0], 
                        damped    =params[index_min_aic][1], 
                        seasonal  =params[index_min_aic][2], 
                        seasonal_periods=12)    
    results = model.fit()

    # Get summary of the model
    #print(results.summary())
    #print(results.summary().tables[1])
    #print(results.summary().tables[2])

    # plot residual errors
    residuals = pd.DataFrame(results.resid)
    residuals.plot()
    plt.show()
    residuals.plot(kind='kde')
    plt.show()
    
    print(residuals.describe())

    # ------------------------------------------------
    # Validating forecasts from the fitted model
    # ------------------------------------------------
    pred    = results.predict(start=pd.to_datetime('1955-01'))
    actual  = dataset['1955-01':].plot(label='observed')
    
    pred.plot(ax=actual, label='Forecast', alpha=.7, figsize=(9, 7))

    actual.set_xlabel('Date')
    actual.set_ylabel('Passengers')
    plt.legend()
    plt.show()

    # -------------------------------------------------------
    # Evaluating the model using different KPIs or metrics
    # -------------------------------------------------------
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
    y_forecasted    = results.predict(start=pd.to_datetime('1955-01'))
    y_truth         = dataset['1955-01':]

    coefficient_of_dermination = r2_score(y_truth, y_forecasted)
    print("R squared: ", coefficient_of_dermination)

    mae = mean_absolute_error(y_truth, y_forecasted)
    print('The Mean Absolute Error of our forecasts is {}'.format(round(mae, 2)))

    mse = mean_squared_error(y_truth, y_forecasted)
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

    msle = mean_squared_log_error(y_truth, y_forecasted)
    print('The Mean Squared Log Error of our forecasts is {}'.format(round(msle, 2)))

    print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

    # -----------------------------------------------
    # Forecasts (Prediction) and Visualisation
    # -----------------------------------------------
    pred_uc = results.forecast(steps=10)
    plt.plot(pred_uc)
    plt.show()
    
    print()
    print("Forecasted Values: ")
    print(pred_uc)
    
    # Visualise forecasts
    ax = dataset.plot(label='Observed', figsize=(9, 7))
    plt.plot(results.predict('1961', '1968'))
    #pred.plot(label='Forecast', alpha=.7, figsize=(9, 7))    
    ax.set_xlabel('Date')
    ax.set_ylabel('Passengers')
    plt.legend()
    plt.show()

#TSF_3_using_holtwinters_model()


def TSF_4_using_SARIMAX_model():
    # ---------------------------------
    # Load packages
    # ---------------------------------
    import warnings
    warnings.filterwarnings("ignore")    
    
    import itertools
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
    import statsmodels.api as sm
    from pandas.tools.plotting import autocorrelation_plot
    
    # ---------------------------------
    # set plot attributes
    # ---------------------------------
    plt.style.use('fivethirtyeight')
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['xtick.labelsize'] = 10
    matplotlib.rcParams['ytick.labelsize'] = 10
    matplotlib.rcParams['text.color'] = 'k'
    matplotlib.rcParams['figure.figsize'] = 10, 7

    # ---------------------------------
    # Load Dataset
    # ---------------------------------
    dataset = pd.read_csv("international-airline-passengers.csv", 
                          header=0, parse_dates=[0],
                          index_col=0, squeeze=True)
    # print dataset
    print()
    print(dataset.shape)
    print(dataset.head(25))

    # ---------------------------------
    # Visualise Time Series Dataset
    # ---------------------------------
    # Plot Dataset
    plt.plot(dataset)
    plt.show()
    # Decompose diffentent Time Series elements e.g. trand, seasonality, Residual ... ...
    decomposition = sm.tsa.seasonal_decompose(dataset, model='additive')
    decomposition.plot()
    plt.show()

    # Auto-correlation plot
    autocorrelation_plot(dataset)
    plt.show()

    # -------------------------------------------------
    # Grid Search for parameters - SARIMAX(p,q) Model 
    # -------------------------------------------------
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q)) # it replaces loop for iteration
    print(pdq)
    
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    print(seasonal_pdq)

    params = []
    params_seasonal = []
    AICs = []

    print()
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                    mod = sm.tsa.statespace.SARIMAX(dataset,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
                    results = mod.fit()
                    
                    print()
                    print("Parameter values for p, d and q {}x{}".format(param, param_seasonal))
                    print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
                    print()
            
                    params.append(param)
                    params_seasonal.append(param_seasonal)
                    AICs.append(results.aic)
            except:
                    continue

    # Get the index of minimum AIC from the list
    index_min_aic = np.argmin(AICs)

    # Best parameter sets (p, d, q) for ARIMA
    print(params[index_min_aic])
    print(params_seasonal[index_min_aic])
    print(AICs[index_min_aic])

    # -------------------------------------------------
    # Fit ARIMA(p,d,q) Model with the best parameter sets
    # -------------------------------------------------
    model = sm.tsa.statespace.SARIMAX(dataset,
                                      order=params[index_min_aic],
                                      seasonal_order=params_seasonal[index_min_aic],
                                      enforce_stationarity=False,
                                      enforce_invertibility=False)
    results = model.fit()

    # Get summary of the model
    print(results.summary())
    print(results.summary().tables[1])
    print(results.summary().tables[2])

    # plot residual errors
    residuals = pd.DataFrame(results.resid)
    residuals.plot()
    plt.show()
    residuals.plot(kind='kde')
    plt.show()
    
    print(residuals.describe())

    # -------------------------------------------------------------
    # Visualise model's behaviourusing built-in diagnostics
    # ------------------------------------------------------------- 
    results.plot_diagnostics(figsize=(9, 8))
    plt.show()

    # ------------------------------------------------
    # Validating forecasts from the fitted model
    # ------------------------------------------------
    pred    = results.predict(start=pd.to_datetime('1955-01'), dynamic=False)
    actual  = dataset['1955-01':].plot(label='observed')
    
    pred.plot(ax=actual, label='Forecast', alpha=.7, figsize=(9, 7))

    actual.set_xlabel('Date')
    actual.set_ylabel('Passengers')
    plt.legend()
    plt.show()

    # -------------------------------------------------------
    # Evaluating the model using different KPIs or metrics
    # -------------------------------------------------------
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
    y_forecasted    = results.predict(start=pd.to_datetime('1955-01'), dynamic=False)
    y_truth         = dataset['1955-01':]

    coefficient_of_dermination = r2_score(y_truth, y_forecasted)
    print("R squared: ", coefficient_of_dermination)

    mae = mean_absolute_error(y_truth, y_forecasted)
    print('The Mean Absolute Error of our forecasts is {}'.format(round(mae, 2)))

    mse = mean_squared_error(y_truth, y_forecasted)
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

    msle = mean_squared_log_error(y_truth, y_forecasted)
    print('The Mean Squared Log Error of our forecasts is {}'.format(round(msle, 7)))

    print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

    # -----------------------------------------------
    # Forecasts (Prediction) and Visualisation
    # -----------------------------------------------
    pred_uc = results.get_forecast(steps=20)
    pred_ci = pred_uc.conf_int()
    ax = dataset.plot(label='Observed', figsize=(9, 7))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('Passengers')
    plt.legend()
    plt.show()

#TSF_4_using_SARIMAX_model()


def TSF_5_using_fbProphet():
    pass

#TSF_5_using_fbProphet()

# LSTM
def TSF_6_using_TF():
    # ---------------------------------
    # Load packages
    # ---------------------------------
    import warnings
    warnings.filterwarnings("ignore")    
    
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
    import statsmodels.api as sm
    from pandas.tools.plotting import autocorrelation_plot
    
    # ---------------------------------
    # set plot attributes
    # ---------------------------------
    plt.style.use('fivethirtyeight')
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['xtick.labelsize'] = 10
    matplotlib.rcParams['ytick.labelsize'] = 10
    matplotlib.rcParams['text.color'] = 'k'
    matplotlib.rcParams['figure.figsize'] = 10, 7

    # ---------------------------------
    # Load Dataset
    # ---------------------------------
    dataset = pd.read_csv("international-airline-passengers.csv", 
                          header=0, parse_dates=[0],
                          index_col=0, squeeze=True)
    
    #dataset = pd.read_csv("international-airline-passengers.csv")
    #dataset = list(dataset["passengers"])

    # print dataset
    print()
    print(dataset.shape)
    print(dataset.head(25))

    # ---------------------------------
    # Visualise Time Series Dataset
    # ---------------------------------
    # Plot Dataset
    plt.plot(dataset)
    plt.show()
    # Decompose diffentent Time Series elements e.g. trand, seasonality, Residual ... ...
    decomposition = sm.tsa.seasonal_decompose(dataset, model='additive')
    decomposition.plot()
    plt.show()

    # Auto-correlation plot
    autocorrelation_plot(dataset)
    plt.show()

    # split a multivariate sequence into samples
    from numpy import array
    def split_sequences(sequences, n_steps):
    	X, y = list(), list()
    	for i in range(len(sequences)):
    		# find the end of this pattern
    		end_ix = i + n_steps
    		# check if we are beyond the dataset
    		if end_ix > len(sequences)-1:
    			break
    		# gather input and output parts of the pattern
            
    		seq_x, seq_y = sequences[i:end_ix], sequences[end_ix]
    		X.append(seq_x)
    		y.append(seq_y)
    	return array(X), array(y)

    # choose a number of time steps
    n_steps = 3
    # convert into input/output
    X, y = split_sequences(dataset, n_steps)
    
    print(X.shape)
    print(y)

    # summarize the data
    for i in range(len(X)):
        print(X[i], y[i])

    #print(es)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    
    from keras.models import Sequential
    from keras.layers import LSTM
    from keras.layers import Dense

    # define model - using LSTM model
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(n_steps, n_features)))
    #model.add(Dense(5, activation='relu'))    
    model.add(Dense(output_dim = 1))
    model.compile(optimizer='adam', loss='mse')
    
    # fit model
    model.fit(X, y, epochs=5000, verbose=1)

    # demonstrate prediction
    dataset = pd.read_csv("international-airline-passengers.csv")
    dataset = dataset['passengers']
    
    # convert into input/output
    X, y = split_sequences(dataset, n_steps)    

    x_input = X.reshape((X.shape[0], X.shape[1], n_features))
    yhat = model.predict(x_input, verbose=1)
    #print(yhat)

    df_pred = pd.DataFrame.from_records(yhat, columns = ['predicted'])
    df_pred = df_pred.reset_index(drop=True)
    
    df_actual = dataset[n_steps:len(dataset)]
    df_actual = df_actual.reset_index(drop=True)

    # report performance
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error

    coefficient_of_dermination = r2_score(df_actual, df_pred)
    print("R squared: ", coefficient_of_dermination)

    mae = mean_absolute_error(df_actual, df_pred)
    print('The Mean Absolute Error of our forecasts is {}'.format(round(mae, 2)))

    mse = mean_squared_error(df_actual, df_pred)
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

    msle = mean_squared_log_error(df_actual, df_pred)
    print('The Mean Squared Log Error of our forecasts is {}'.format(round(msle, 2)))

    print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

    # plot
    ax = df_actual.plot(label='Observed', figsize=(9, 7))
    df_pred.plot(ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Passengers')
    plt.legend()
    plt.show()

    # ---------------------------------------------------------------------------
    # Future Predictions
    predictions = model.predict(x_input, verbose=1)
    future_time_steps = 24
    x1 = x_input[-1:,:,:]   # take the last input
    p1 = predictions[-1:]   # take the last prediction
    
    for i in range(future_time_steps):
    
        x2 = np.array([[x1[0][1], x1[0][2], p1]])
        p2 = model.predict(x2, verbose=1)    
        predictions = np.append(predictions, p2)

        x1 = x2
        p1 = p2

    yhat = predictions
    yhat = np.reshape(yhat,(-1, 1))

    df_pred = pd.DataFrame.from_records(yhat, columns = ['predicted'])
    df_pred = df_pred.reset_index(drop=True)
    
    df_actual = dataset[n_steps:len(dataset)]
    df_actual = df_actual.reset_index(drop=True)    

    # plot
    ax = df_actual.plot(label='Observed', figsize=(9, 7))
    df_pred.plot(ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Passengers')
    plt.legend()
    plt.show()
    # ---------------------------------------------------------------------------
#TSF_6_using_TF()

# CNN
def TSF_7_using_TF():
    # ---------------------------------
    # Load packages
    # ---------------------------------
    import warnings
    warnings.filterwarnings("ignore")    
    
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
    import statsmodels.api as sm
    from pandas.tools.plotting import autocorrelation_plot
    
    # ---------------------------------
    # set plot attributes
    # ---------------------------------
    plt.style.use('fivethirtyeight')
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['xtick.labelsize'] = 10
    matplotlib.rcParams['ytick.labelsize'] = 10
    matplotlib.rcParams['text.color'] = 'k'
    matplotlib.rcParams['figure.figsize'] = 10, 7

    # ---------------------------------
    # Load Dataset
    # ---------------------------------
    dataset = pd.read_csv("international-airline-passengers.csv", 
                          header=0, parse_dates=[0],
                          index_col=0, squeeze=True)
    
    #dataset = pd.read_csv("international-airline-passengers.csv")
    #dataset = list(dataset["passengers"])

    # print dataset
    print()
    print(dataset.shape)
    print(dataset.head(25))

    # ---------------------------------
    # Visualise Time Series Dataset
    # ---------------------------------
    # Plot Dataset
    plt.plot(dataset)
    plt.show()
    # Decompose diffentent Time Series elements e.g. trand, seasonality, Residual ... ...
    decomposition = sm.tsa.seasonal_decompose(dataset, model='additive')
    decomposition.plot()
    plt.show()

    # Auto-correlation plot
    autocorrelation_plot(dataset)
    plt.show()

    # split a multivariate sequence into samples
    from numpy import array
    def split_sequences(sequences, n_steps):
    	X, y = list(), list()
    	for i in range(len(sequences)):
    		# find the end of this pattern
    		end_ix = i + n_steps
    		# check if we are beyond the dataset
    		if end_ix > len(sequences)-1:
    			break
    		# gather input and output parts of the pattern
            
    		seq_x, seq_y = sequences[i:end_ix], sequences[end_ix]
    		X.append(seq_x)
    		y.append(seq_y)
    	return array(X), array(y)

    # choose a number of time steps
    n_steps = 3
    # convert into input/output
    X, y = split_sequences(dataset, n_steps)
    
    print(X.shape)
    print(y)

    # summarize the data
    for i in range(len(X)):
        print(X[i], y[i])

    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    
    #from numpy import array
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Flatten
    from keras.layers.convolutional import Conv1D
    from keras.layers.convolutional import MaxPooling1D

    # define model - using CNN model
    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(200, activation='relu'))    
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    # fit model
    model.fit(X, y, epochs=5000, verbose=1)

    # demonstrate prediction
    dataset = pd.read_csv("international-airline-passengers.csv")
    dataset = dataset['passengers']
    
    # convert into input/output
    X, y = split_sequences(dataset, n_steps)    

    x_input = X.reshape((X.shape[0], X.shape[1], n_features))
    yhat = model.predict(x_input, verbose=1)
    #print(yhat)

    df_pred = pd.DataFrame.from_records(yhat, columns = ['predicted'])
    df_pred = df_pred.reset_index(drop=True)
    
    df_actual = dataset[n_steps:len(dataset)]
    df_actual = df_actual.reset_index(drop=True)

    # report performance
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error

    coefficient_of_dermination = r2_score(df_actual, df_pred)
    print("R squared: ", coefficient_of_dermination)

    mae = mean_absolute_error(df_actual, df_pred)
    print('The Mean Absolute Error of our forecasts is {}'.format(round(mae, 2)))

    mse = mean_squared_error(df_actual, df_pred)
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

    msle = mean_squared_log_error(df_actual, df_pred)
    print('The Mean Squared Log Error of our forecasts is {}'.format(round(msle, 2)))

    print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

    # plot
    ax = df_actual.plot(label='Observed', figsize=(9, 7))
    df_pred.plot(ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Passengers')
    plt.legend()
    plt.show()

    # ---------------------------------------------------------------------------
    # Future Predictions
    predictions = model.predict(x_input, verbose=1)
    future_time_steps = 24
    x1 = x_input[-1:,:,:]   # take the last input
    p1 = predictions[-1:]   # take the last prediction
    
    for i in range(future_time_steps):
    
        x2 = np.array([[x1[0][1], x1[0][2], p1]])
        p2 = model.predict(x2, verbose=1)    
        predictions = np.append(predictions, p2)

        x1 = x2
        p1 = p2

    yhat = predictions
    yhat = np.reshape(yhat,(-1, 1))

    df_pred = pd.DataFrame.from_records(yhat, columns = ['predicted'])
    df_pred = df_pred.reset_index(drop=True)
    
    df_actual = dataset[n_steps:len(dataset)]
    df_actual = df_actual.reset_index(drop=True)    

    # plot
    ax = df_actual.plot(label='Observed', figsize=(9, 7))
    df_pred.plot(ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Passengers')
    plt.legend()
    plt.show()
    # ---------------------------------------------------------------------------

#TSF_7_using_TF()


# MLP
def TSF_8_using_TF():
    # ---------------------------------
    # Load packages
    # ---------------------------------
    import warnings
    warnings.filterwarnings("ignore")    
    
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
    import statsmodels.api as sm
    from pandas.tools.plotting import autocorrelation_plot
    
    # ---------------------------------
    # set plot attributes
    # ---------------------------------
    plt.style.use('fivethirtyeight')
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['xtick.labelsize'] = 10
    matplotlib.rcParams['ytick.labelsize'] = 10
    matplotlib.rcParams['text.color'] = 'k'
    matplotlib.rcParams['figure.figsize'] = 10, 7

    # ---------------------------------
    # Load Dataset
    # ---------------------------------
    dataset = pd.read_csv("international-airline-passengers.csv", 
                          header=0, parse_dates=[0],
                          index_col=0, squeeze=True)
    
    #dataset = pd.read_csv("international-airline-passengers.csv")
    #dataset = list(dataset["passengers"])

    # print dataset
    print()
    print(dataset.shape)
    print(dataset.head(25))

    # ---------------------------------
    # Visualise Time Series Dataset
    # ---------------------------------
    # Plot Dataset
    plt.plot(dataset)
    plt.show()
    # Decompose diffentent Time Series elements e.g. trand, seasonality, Residual ... ...
    decomposition = sm.tsa.seasonal_decompose(dataset, model='additive')
    decomposition.plot()
    plt.show()

    # Auto-correlation plot
    autocorrelation_plot(dataset)
    plt.show()

    # split a multivariate sequence into samples
    from numpy import array
    def split_sequences(sequences, n_steps):
    	X, y = list(), list()
    	for i in range(len(sequences)):
    		# find the end of this pattern
    		end_ix = i + n_steps
    		# check if we are beyond the dataset
    		if end_ix > len(sequences)-1:
    			break
    		# gather input and output parts of the pattern
            
    		seq_x, seq_y = sequences[i:end_ix], sequences[end_ix]
    		X.append(seq_x)
    		y.append(seq_y)
    	return array(X), array(y)

    # choose a number of time steps
    n_steps = 3
    # convert into input/output
    X, y = split_sequences(dataset, n_steps)
    
    print(X.shape)
    print(y)

    # summarize the data
    for i in range(len(X)):
        print(X[i], y[i])

    # reshape from [samples, timesteps] into [samples, timesteps, features]
    #n_features = 1
    #X = X.reshape((X.shape[0], X.shape[1], n_features))
    
    from keras.models import Sequential
    from keras.layers import Dense

    # define model - using MLP model
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=n_steps))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    model.summary()

    from keras.utils.vis_utils import plot_model    
    plot_model(model)
    
    # fit model
    model.fit(X, y, epochs=10000, verbose=1)

    # demonstrate prediction
    dataset = pd.read_csv("international-airline-passengers.csv")
    dataset = dataset['passengers']
    
    # convert into input/output
    X, y = split_sequences(dataset, n_steps)    

    #x_input = X.reshape((X.shape[0], X.shape[1], n_features))
    yhat = model.predict(X, verbose=1)
    #print(yhat)

    df_pred = pd.DataFrame.from_records(yhat, columns = ['predicted'])
    df_pred = df_pred.reset_index(drop=True)
    
    df_actual = dataset[n_steps:len(dataset)]
    df_actual = df_actual.reset_index(drop=True)

    # report performance
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
    print()

    coefficient_of_dermination = r2_score(df_actual, df_pred)
    print("R squared: ", coefficient_of_dermination)

    mae = mean_absolute_error(df_actual, df_pred)
    print('The Mean Absolute Error of our forecasts is {}'.format(round(mae, 2)))

    mse = mean_squared_error(df_actual, df_pred)
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

    msle = mean_squared_log_error(df_actual, df_pred)
    print('The Mean Squared Log Error of our forecasts is {}'.format(round(msle, 2)))

    print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

    # plot
    ax = df_actual.plot(label='Observed', figsize=(9, 7))
    df_pred.plot(ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Passengers')
    plt.legend()
    plt.show()

    # ---------------------------------------------------------------------------
    # Future Predictions
    predictions = model.predict(X, verbose=1)
    future_time_steps = 24
    
    x1 = X[-1:]   # take the last input
    p1 = predictions[-1:]   # take the last prediction
    
    for i in range(future_time_steps):
    
        x2 = np.array([[x1[0][1], x1[0][2], p1]])
        p2 = model.predict(x2, verbose=1)    
        predictions = np.append(predictions, p2)

        x1 = x2
        p1 = p2

    yhat = predictions
    yhat = np.reshape(yhat,(-1, 1))

    df_pred = pd.DataFrame.from_records(yhat, columns = ['predicted'])
    df_pred = df_pred.reset_index(drop=True)
    
    df_actual = dataset[n_steps:len(dataset)]
    df_actual = df_actual.reset_index(drop=True)    

    # plot
    ax = df_actual.plot(label='Observed', figsize=(9, 7))
    df_pred.plot(ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Passengers')
    plt.legend()
    plt.show()
    # ---------------------------------------------------------------------------

#TSF_8_using_TF()



def TSF_9_using_AR_model():
    # ---------------------------------
    # Load packages
    # ---------------------------------
    import warnings
    warnings.filterwarnings("ignore")    
    
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.tsa.ar_model import AR
    
    # ---------------------------------
    # set plot attributes
    # ---------------------------------
    plt.style.use('fivethirtyeight')
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['xtick.labelsize'] = 10
    matplotlib.rcParams['ytick.labelsize'] = 10
    matplotlib.rcParams['text.color'] = 'k'
    matplotlib.rcParams['figure.figsize'] = 10, 7

    # ---------------------------------
    # Load Dataset
    # ---------------------------------
    dataset = pd.read_csv("international-airline-passengers.csv", 
                          header=0, parse_dates=[0],
                          index_col=0, squeeze=True)
    # print dataset
    print()
    print(dataset.shape)
    print(dataset.head(25))

    # ---------------------------------
    # Visualise Time Series Dataset
    # ---------------------------------
    # Plot Dataset
    plt.plot(dataset)
    plt.show()
    # Decompose diffentent Time Series elements e.g. trand, seasonality, Residual ... ...
    decomposition = sm.tsa.seasonal_decompose(dataset, model='additive')
    decomposition.plot()
    plt.show()

    # -------------------------------------------------
    # AR Model 
    # -------------------------------------------------

    model = AR(dataset)
    results = model.fit()
                
    # ------------------------------------------------
    # Validating forecasts from the fitted model
    # ------------------------------------------------
    pred    = results.predict(start=pd.to_datetime('1955-01'), dynamic=False)
    actual  = dataset['1955-01':].plot(label='observed')
    
    pred.plot(ax=actual, label='Forecast', alpha=.7, figsize=(9, 7))

    actual.set_xlabel('Date')
    actual.set_ylabel('Passengers')
    plt.legend()
    plt.show()

    # -------------------------------------------------------
    # Evaluating the model using different KPIs or metrics
    # -------------------------------------------------------
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
    y_forecasted    = results.predict(start=pd.to_datetime('1955-01'), dynamic=False)
    y_truth         = dataset['1955-01':]

    coefficient_of_dermination = r2_score(y_truth, y_forecasted)
    print("R squared: ", coefficient_of_dermination)

    mae = mean_absolute_error(y_truth, y_forecasted)
    print('The Mean Absolute Error of our forecasts is {}'.format(round(mae, 2)))

    mse = mean_squared_error(y_truth, y_forecasted)
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

    msle = mean_squared_log_error(y_truth, y_forecasted)
    print('The Mean Squared Log Error of our forecasts is {}'.format(round(msle, 2)))

    print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

    # -----------------------------------------------
    # Forecasts (Prediction) and Visualisation
    # -----------------------------------------------
    pred    = results.predict(start=pd.to_datetime('1955-01'), 
                              end=pd.to_datetime('1968-01'),
                              dynamic=False)
    actual  = dataset['1955-01':].plot(label='observed')
    
    pred.plot(ax=actual, label='Forecast', alpha=.7, figsize=(9, 7))

    actual.set_xlabel('Date')
    actual.set_ylabel('Passengers')
    plt.legend()
    plt.show()

#TSF_9_using_AR_model()


def TSF_10_using_ARIMA_model():
# ARIMA model : ARIMA(1,0,0) = First Order Autoregressive Model    

    # ---------------------------------
    # Load packages
    # ---------------------------------
    import warnings
    warnings.filterwarnings("ignore")    
    
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.tsa.arima_model import ARIMA
    
    # ---------------------------------
    # set plot attributes
    # ---------------------------------
    plt.style.use('fivethirtyeight')
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['xtick.labelsize'] = 10
    matplotlib.rcParams['ytick.labelsize'] = 10
    matplotlib.rcParams['text.color'] = 'k'
    matplotlib.rcParams['figure.figsize'] = 10, 7

    # ---------------------------------
    # Load Dataset
    # ---------------------------------
    dataset = pd.read_csv("international-airline-passengers.csv", 
                          header=0, parse_dates=[0],
                          index_col=0, squeeze=True)
    # print dataset
    print()
    print(dataset.shape)
    print(dataset.head(25))

    # ---------------------------------
    # Visualise Time Series Dataset
    # ---------------------------------
    # Plot Dataset
    plt.plot(dataset)
    plt.show()
    # Decompose diffentent Time Series elements e.g. trand, seasonality, Residual ... ...
    decomposition = sm.tsa.seasonal_decompose(dataset, model='additive')
    decomposition.plot()
    plt.show()

    # -------------------------------------------------
    # AR Model 
    # -------------------------------------------------

    model = ARIMA(dataset, order=(1, 0, 0))
    results = model.fit()



    # Get summary of the model
    print(results.summary())
    print(results.summary().tables[1])
    print(results.summary().tables[2])

    # plot residual errors
    residuals = pd.DataFrame(results.resid)
    residuals.plot()
    plt.show()
    residuals.plot(kind='kde')
    plt.show()
    
    print(residuals.describe())

    # -------------------------------------------------------------
    # Visualise model's behaviourusing built-in diagnostics
    # ------------------------------------------------------------- 
    #results.plot_diagnostics(figsize=(9, 8))
    #plt.show()




                
    # ------------------------------------------------
    # Validating forecasts from the fitted model
    # ------------------------------------------------
    pred    = results.predict(start=pd.to_datetime('1955-01'), dynamic=False)
    actual  = dataset['1955-01':].plot(label='observed')
    
    pred.plot(ax=actual, label='Forecast', alpha=.7, figsize=(9, 7))

    actual.set_xlabel('Date')
    actual.set_ylabel('Passengers')
    plt.legend()
    plt.show()

    # -------------------------------------------------------
    # Evaluating the model using different KPIs or metrics
    # -------------------------------------------------------
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
    y_forecasted    = results.predict(start=pd.to_datetime('1955-01'), dynamic=False)
    y_truth         = dataset['1955-01':]

    coefficient_of_dermination = r2_score(y_truth, y_forecasted)
    print("R squared: ", coefficient_of_dermination)

    mae = mean_absolute_error(y_truth, y_forecasted)
    print('The Mean Absolute Error of our forecasts is {}'.format(round(mae, 2)))

    mse = mean_squared_error(y_truth, y_forecasted)
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

    msle = mean_squared_log_error(y_truth, y_forecasted)
    print('The Mean Squared Log Error of our forecasts is {}'.format(round(msle, 2)))

    print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

    # -----------------------------------------------
    # Forecasts (Prediction) and Visualisation
    # -----------------------------------------------
    pred    = results.predict(start=pd.to_datetime('1955-01'), 
                              end=pd.to_datetime('1965-01'),
                              dynamic=False)
    actual  = dataset['1955-01':].plot(label='observed')
    
    pred.plot(ax=actual, label='Forecast', alpha=.7, figsize=(9, 7))

    actual.set_xlabel('Date')
    actual.set_ylabel('Passengers')
    plt.legend()
    plt.show()

#TSF_10_using_ARIMA_model()



def TSF_11_using_SARIMA_model():
# ARIMA model with seasonality : Using Seasonal random walk model: ARIMA(0,0,0)s(0,1,0)

    # ---------------------------------
    # Load packages
    # ---------------------------------
    import warnings
    warnings.filterwarnings("ignore")    
    
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    
    # ---------------------------------
    # set plot attributes
    # ---------------------------------
    plt.style.use('fivethirtyeight')
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['xtick.labelsize'] = 10
    matplotlib.rcParams['ytick.labelsize'] = 10
    matplotlib.rcParams['text.color'] = 'k'
    matplotlib.rcParams['figure.figsize'] = 10, 7

    # ---------------------------------
    # Load Dataset
    # ---------------------------------
    dataset = pd.read_csv("international-airline-passengers.csv", 
                          header=0, parse_dates=[0],
                          index_col=0, squeeze=True)
    # print dataset
    print()
    print(dataset.shape)
    print(dataset.head(25))

    # ---------------------------------
    # Visualise Time Series Dataset
    # ---------------------------------
    # Plot Dataset
    plt.plot(dataset)
    plt.show()
    # Decompose diffentent Time Series elements e.g. trand, seasonality, Residual ... ...
    decomposition = sm.tsa.seasonal_decompose(dataset, model='additive')
    decomposition.plot()
    plt.show()

    # -------------------------------------------------
    # AR Model 
    # -------------------------------------------------

    model = SARIMAX(dataset, order=(0, 0, 0), seasonal_order=(0, 1, 0, 12))
    results = model.fit(disp=False)


    # Get summary of the model
    print(results.summary())
    print(results.summary().tables[1])
    print(results.summary().tables[2])

    # plot residual errors
    residuals = pd.DataFrame(results.resid)
    residuals.plot()
    plt.show()
    residuals.plot(kind='kde')
    plt.show()
    
    print(residuals.describe())

    # -------------------------------------------------------------
    # Visualise model's behaviourusing built-in diagnostics
    # ------------------------------------------------------------- 
    results.plot_diagnostics(figsize=(9, 8))
    plt.show()




                
    # ------------------------------------------------
    # Validating forecasts from the fitted model
    # ------------------------------------------------
    pred    = results.predict(start=pd.to_datetime('1955-01'), dynamic=False)
    actual  = dataset['1955-01':].plot(label='observed')
    
    pred.plot(ax=actual, label='Forecast', alpha=.7, figsize=(9, 7))

    actual.set_xlabel('Date')
    actual.set_ylabel('Passengers')
    plt.legend()
    plt.show()

    # -------------------------------------------------------
    # Evaluating the model using different KPIs or metrics
    # -------------------------------------------------------
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
    y_forecasted    = results.predict(start=pd.to_datetime('1955-01'), dynamic=False)
    y_truth         = dataset['1955-01':]

    coefficient_of_dermination = r2_score(y_truth, y_forecasted)
    print("R squared: ", coefficient_of_dermination)

    mae = mean_absolute_error(y_truth, y_forecasted)
    print('The Mean Absolute Error of our forecasts is {}'.format(round(mae, 2)))

    mse = mean_squared_error(y_truth, y_forecasted)
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

    msle = mean_squared_log_error(y_truth, y_forecasted)
    print('The Mean Squared Log Error of our forecasts is {}'.format(round(msle, 2)))

    print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

    # -----------------------------------------------
    # Forecasts (Prediction) and Visualisation
    # -----------------------------------------------
    pred    = results.predict(start=pd.to_datetime('1955-01'), 
                              end=pd.to_datetime('1965-01'),
                              dynamic=False)
    actual  = dataset['1955-01':].plot(label='observed')
    
    pred.plot(ax=actual, label='Forecast', alpha=.7, figsize=(9, 7))

    actual.set_xlabel('Date')
    actual.set_ylabel('Passengers')
    plt.legend()
    plt.show()

#TSF_11_using_SARIMA_model()


def TSF_12_using_SARIMA_model():
# ARIMA model with seasonality : Using Seasonal random trend model: ARIMA(0,1,0)s(0,1,0)

    # ---------------------------------
    # Load packages
    # ---------------------------------
    import warnings
    warnings.filterwarnings("ignore")    
    
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    
    # ---------------------------------
    # set plot attributes
    # ---------------------------------
    plt.style.use('fivethirtyeight')
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['xtick.labelsize'] = 10
    matplotlib.rcParams['ytick.labelsize'] = 10
    matplotlib.rcParams['text.color'] = 'k'
    matplotlib.rcParams['figure.figsize'] = 10, 7

    # ---------------------------------
    # Load Dataset
    # ---------------------------------
    dataset = pd.read_csv("international-airline-passengers.csv", 
                          header=0, parse_dates=[0],
                          index_col=0, squeeze=True)
    # print dataset
    print()
    print(dataset.shape)
    print(dataset.head(25))

    # ---------------------------------
    # Visualise Time Series Dataset
    # ---------------------------------
    # Plot Dataset
    plt.plot(dataset)
    plt.show()
    # Decompose diffentent Time Series elements e.g. trand, seasonality, Residual ... ...
    decomposition = sm.tsa.seasonal_decompose(dataset, model='additive')
    decomposition.plot()
    plt.show()

    # -------------------------------------------------
    # AR Model 
    # -------------------------------------------------

    model = SARIMAX(dataset, order=(0,1, 0), seasonal_order=(0, 1, 0, 12))
    results = model.fit(disp=False)


    # Get summary of the model
    print(results.summary())
    print(results.summary().tables[1])
    print(results.summary().tables[2])

    # plot residual errors
    residuals = pd.DataFrame(results.resid)
    residuals.plot()
    plt.show()
    residuals.plot(kind='kde')
    plt.show()
    
    print(residuals.describe())

    # -------------------------------------------------------------
    # Visualise model's behaviourusing built-in diagnostics
    # ------------------------------------------------------------- 
    results.plot_diagnostics(figsize=(9, 8))
    plt.show()


                
    # ------------------------------------------------
    # Validating forecasts from the fitted model
    # ------------------------------------------------
    pred    = results.predict(start=pd.to_datetime('1955-01'), dynamic=False)
    actual  = dataset['1955-01':].plot(label='observed')
    
    pred.plot(ax=actual, label='Forecast', alpha=.7, figsize=(9, 7))

    actual.set_xlabel('Date')
    actual.set_ylabel('Passengers')
    plt.legend()
    plt.show()

    # -------------------------------------------------------
    # Evaluating the model using different KPIs or metrics
    # -------------------------------------------------------
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
    y_forecasted    = results.predict(start=pd.to_datetime('1955-01'), dynamic=False)
    y_truth         = dataset['1955-01':]

    coefficient_of_dermination = r2_score(y_truth, y_forecasted)
    print("R squared: ", coefficient_of_dermination)

    mae = mean_absolute_error(y_truth, y_forecasted)
    print('The Mean Absolute Error of our forecasts is {}'.format(round(mae, 2)))

    mse = mean_squared_error(y_truth, y_forecasted)
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

    msle = mean_squared_log_error(y_truth, y_forecasted)
    print('The Mean Squared Log Error of our forecasts is {}'.format(round(msle, 2)))

    print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

    # -----------------------------------------------
    # Forecasts (Prediction) and Visualisation
    # -----------------------------------------------
    pred    = results.predict(start=pd.to_datetime('1955-01'), 
                              end=pd.to_datetime('1965-01'),
                              dynamic=False)
    actual  = dataset['1955-01':].plot(label='observed')
    
    pred.plot(ax=actual, label='Forecast', alpha=.7, figsize=(9, 7))

    actual.set_xlabel('Date')
    actual.set_ylabel('Passengers')
    plt.legend()
    plt.show()

#TSF_12_using_SARIMA_model()



def TSF_13_using_SARIMA_model():
# ARIMA model with seasonality : Using General seasonal ARIMA models: (0,1,1)s(0,1,1)

    # ---------------------------------
    # Load packages
    # ---------------------------------
    import warnings
    warnings.filterwarnings("ignore")    
    
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    
    # ---------------------------------
    # set plot attributes
    # ---------------------------------
    plt.style.use('fivethirtyeight')
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['xtick.labelsize'] = 10
    matplotlib.rcParams['ytick.labelsize'] = 10
    matplotlib.rcParams['text.color'] = 'k'
    matplotlib.rcParams['figure.figsize'] = 10, 7

    # ---------------------------------
    # Load Dataset
    # ---------------------------------
    dataset = pd.read_csv("international-airline-passengers.csv", 
                          header=0, parse_dates=[0],
                          index_col=0, squeeze=True)
    # print dataset
    print()
    print(dataset.shape)
    print(dataset.head(25))

    # ---------------------------------
    # Visualise Time Series Dataset
    # ---------------------------------
    # Plot Dataset
    plt.plot(dataset)
    plt.show()
    # Decompose diffentent Time Series elements e.g. trand, seasonality, Residual ... ...
    decomposition = sm.tsa.seasonal_decompose(dataset, model='additive')
    decomposition.plot()
    plt.show()

    # -------------------------------------------------
    # AR Model 
    # -------------------------------------------------

    model = SARIMAX(dataset, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12))
    results = model.fit(disp=False)


    # Get summary of the model
    print(results.summary())
    print(results.summary().tables[1])
    print(results.summary().tables[2])

    # plot residual errors
    residuals = pd.DataFrame(results.resid)
    residuals.plot()
    plt.show()
    residuals.plot(kind='kde')
    plt.show()
    
    print(residuals.describe())

    # -------------------------------------------------------------
    # Visualise model's behaviourusing built-in diagnostics
    # ------------------------------------------------------------- 
    results.plot_diagnostics(figsize=(9, 8))
    plt.show()


                
    # ------------------------------------------------
    # Validating forecasts from the fitted model
    # ------------------------------------------------
    pred    = results.predict(start=pd.to_datetime('1955-01'), dynamic=False)
    actual  = dataset['1955-01':].plot(label='observed')
    
    pred.plot(ax=actual, label='Forecast', alpha=.7, figsize=(9, 7))

    actual.set_xlabel('Date')
    actual.set_ylabel('Passengers')
    plt.legend()
    plt.show()

    # -------------------------------------------------------
    # Evaluating the model using different KPIs or metrics
    # -------------------------------------------------------
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
    y_forecasted    = results.predict(start=pd.to_datetime('1955-01'), dynamic=False)
    y_truth         = dataset['1955-01':]

    coefficient_of_dermination = r2_score(y_truth, y_forecasted)
    print("R squared: ", coefficient_of_dermination)

    mae = mean_absolute_error(y_truth, y_forecasted)
    print('The Mean Absolute Error of our forecasts is {}'.format(round(mae, 2)))

    mse = mean_squared_error(y_truth, y_forecasted)
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

    msle = mean_squared_log_error(y_truth, y_forecasted)
    print('The Mean Squared Log Error of our forecasts is {}'.format(round(msle, 2)))

    print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

    # -----------------------------------------------
    # Forecasts (Prediction) and Visualisation
    # -----------------------------------------------
    pred    = results.predict(start=pd.to_datetime('1955-01'), 
                              end=pd.to_datetime('1965-01'),
                              dynamic=False)
    actual  = dataset['1955-01':].plot(label='observed')
    
    pred.plot(ax=actual, label='Forecast', alpha=.7, figsize=(9, 7))

    actual.set_xlabel('Date')
    actual.set_ylabel('Passengers')
    plt.legend()
    plt.show()

#TSF_13_using_SARIMA_model()



def TSF_14_using_SARIMA_model():
# ARIMA model with seasonality : Using ARIMA models: (4,1,3)s(0,2,1) 

    # ---------------------------------
    # Load packages
    # ---------------------------------
    import warnings
    warnings.filterwarnings("ignore")    
    
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    
    # ---------------------------------
    # set plot attributes
    # ---------------------------------
    plt.style.use('fivethirtyeight')
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['xtick.labelsize'] = 10
    matplotlib.rcParams['ytick.labelsize'] = 10
    matplotlib.rcParams['text.color'] = 'k'
    matplotlib.rcParams['figure.figsize'] = 10, 7

    # ---------------------------------
    # Load Dataset
    # ---------------------------------
    dataset = pd.read_csv("international-airline-passengers.csv", 
                          header=0, parse_dates=[0],
                          index_col=0, squeeze=True)
    # print dataset
    print()
    print(dataset.shape)
    print(dataset.head(25))

    # ---------------------------------
    # Visualise Time Series Dataset
    # ---------------------------------
    # Plot Dataset
    plt.plot(dataset)
    plt.show()
    # Decompose diffentent Time Series elements e.g. trand, seasonality, Residual ... ...
    decomposition = sm.tsa.seasonal_decompose(dataset, model='additive')
    decomposition.plot()
    plt.show()

    # -------------------------------------------------
    # AR Model 
    # -------------------------------------------------

    model = SARIMAX(dataset, order=(4, 1, 3), seasonal_order=(0, 2, 1, 12))
    results = model.fit(disp=False)
                


    # Get summary of the model
    print(results.summary())
    print(results.summary().tables[1])
    print(results.summary().tables[2])

    # plot residual errors
    residuals = pd.DataFrame(results.resid)
    residuals.plot()
    plt.show()
    residuals.plot(kind='kde')
    plt.show()
    
    print(residuals.describe())

    # -------------------------------------------------------------
    # Visualise model's behaviourusing built-in diagnostics
    # ------------------------------------------------------------- 
    results.plot_diagnostics(figsize=(9, 8))
    plt.show()





    # ------------------------------------------------
    # Validating forecasts from the fitted model
    # ------------------------------------------------
    pred    = results.predict(start=pd.to_datetime('1955-01'), dynamic=False)
    actual  = dataset['1955-01':].plot(label='observed')
    
    pred.plot(ax=actual, label='Forecast', alpha=.7, figsize=(9, 7))

    actual.set_xlabel('Date')
    actual.set_ylabel('Passengers')
    plt.legend()
    plt.show()

    # -------------------------------------------------------
    # Evaluating the model using different KPIs or metrics
    # -------------------------------------------------------
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
    y_forecasted    = results.predict(start=pd.to_datetime('1955-01'), dynamic=False)
    y_truth         = dataset['1955-01':]

    coefficient_of_dermination = r2_score(y_truth, y_forecasted)
    print("R squared: ", coefficient_of_dermination)

    mae = mean_absolute_error(y_truth, y_forecasted)
    print('The Mean Absolute Error of our forecasts is {}'.format(round(mae, 2)))

    mse = mean_squared_error(y_truth, y_forecasted)
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

    msle = mean_squared_log_error(y_truth, y_forecasted)
    print('The Mean Squared Log Error of our forecasts is {}'.format(round(msle, 2)))

    print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

    # -----------------------------------------------
    # Forecasts (Prediction) and Visualisation
    # -----------------------------------------------
    pred    = results.predict(start=pd.to_datetime('1955-01'), 
                              end=pd.to_datetime('1965-01'),
                              dynamic=False)
    actual  = dataset['1955-01':].plot(label='observed')
    
    pred.plot(ax=actual, label='Forecast', alpha=.7, figsize=(9, 7))

    actual.set_xlabel('Date')
    actual.set_ylabel('Passengers')
    plt.legend()
    plt.show()

#TSF_14_using_SARIMA_model()



# LSTM
def TSF_15_using_TF():
    # ---------------------------------
    # Load packages
    # ---------------------------------
    import warnings
    warnings.filterwarnings("ignore")    
    
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
    import statsmodels.api as sm
    from pandas.tools.plotting import autocorrelation_plot
    
    # ---------------------------------
    # set plot attributes
    # ---------------------------------
    plt.style.use('fivethirtyeight')
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['xtick.labelsize'] = 10
    matplotlib.rcParams['ytick.labelsize'] = 10
    matplotlib.rcParams['text.color'] = 'k'
    matplotlib.rcParams['figure.figsize'] = 10, 7

    # ---------------------------------
    # Load Dataset
    # ---------------------------------
    dataset = pd.read_csv("international-airline-passengers.csv", 
                          header=0, parse_dates=[0],
                          index_col=0, squeeze=True)
    
    #dataset = pd.read_csv("international-airline-passengers.csv")
    #dataset = list(dataset["passengers"])

    # print dataset
    print()
    print(dataset.shape)
    print(dataset.head(25))

    # ---------------------------------
    # Visualise Time Series Dataset
    # ---------------------------------
    # Plot Dataset
    plt.plot(dataset)
    plt.show()
    # Decompose diffentent Time Series elements e.g. trand, seasonality, Residual ... ...
    decomposition = sm.tsa.seasonal_decompose(dataset, model='additive')
    decomposition.plot()
    plt.show()

    # Auto-correlation plot
    autocorrelation_plot(dataset)
    plt.show()

    # split a multivariate sequence into samples
    from numpy import array
    def split_sequences(sequences, n_steps):
    	X, y = list(), list()
    	for i in range(len(sequences)):
    		# find the end of this pattern
    		end_ix = i + n_steps
    		# check if we are beyond the dataset
    		if end_ix > len(sequences)-1:
    			break
    		# gather input and output parts of the pattern
            
    		seq_x, seq_y = sequences[i:end_ix], sequences[end_ix]
    		X.append(seq_x)
    		y.append(seq_y)
    	return array(X), array(y)

    # choose a number of time steps
    n_steps = 3

    # convert into input/output
    X, y = split_sequences(dataset[1:125], n_steps)
    
    print(X.shape)
    print(y)

    # summarize the data
    for i in range(len(X)):
        print(X[i], y[i])

    #print(es)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    
    from keras.models import Sequential
    from keras.layers import LSTM
    from keras.layers import Dense

    # define model - using LSTM model
    model = Sequential()
    #model.add(LSTM(128, activation='tanh', stateful=True, 
    #               batch_input_shape=(1, n_steps, n_features)))
    
    model.add(LSTM(128, activation='relu', input_shape=(n_steps, n_features)))    
    model.add(Dense(50, activation='relu'))    
    model.add(Dense(output_dim = 1))
    model.compile(optimizer='adam', loss='mse')
    
    model.summary()
    
    # fit model
    model.fit(X, y, epochs=10000, verbose=1)

    # demonstrate prediction
    dataset = pd.read_csv("international-airline-passengers.csv")
    dataset = dataset['passengers']
    
    # convert into input/output
    X, y = split_sequences(dataset, n_steps)    

    x_input = X.reshape((X.shape[0], X.shape[1], n_features))
    yhat = model.predict(x_input, verbose=1)

    df_pred = pd.DataFrame.from_records(yhat, columns = ['predicted'])
    df_pred = df_pred.reset_index(drop=True)
    
    df_actual = dataset[n_steps:len(dataset)]
    df_actual = df_actual.reset_index(drop=True)

    # report performance
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error

    coefficient_of_dermination = r2_score(df_actual, df_pred)
    print("R squared: ", coefficient_of_dermination)

    mae = mean_absolute_error(df_actual, df_pred)
    print('The Mean Absolute Error of our forecasts is {}'.format(round(mae, 2)))

    mse = mean_squared_error(df_actual, df_pred)
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

    msle = mean_squared_log_error(df_actual, df_pred)
    print('The Mean Squared Log Error of our forecasts is {}'.format(round(msle, 2)))

    print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

    # plot
    ax = df_actual.plot(label='Observed', figsize=(9, 7))
    df_pred.plot(ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Passengers')
    plt.legend()
    plt.show()

    # ---------------------------------------------------------------------------
    # Future Predictions
    predictions = model.predict(x_input, verbose=1)
    future_time_steps = 24
    x1 = x_input[-1:,:,:]   # take the last input
    p1 = predictions[-1:]   # take the last prediction
    
    for i in range(future_time_steps):
    
        x2 = np.array([[x1[0][1], x1[0][2], p1]])
        p2 = model.predict(x2, verbose=1)    
        predictions = np.append(predictions, p2)

        x1 = x2
        p1 = p2

    yhat = predictions
    yhat = np.reshape(yhat,(-1, 1))

    df_pred = pd.DataFrame.from_records(yhat, columns = ['predicted'])
    df_pred = df_pred.reset_index(drop=True)
    
    df_actual = dataset[n_steps:len(dataset)]
    df_actual = df_actual.reset_index(drop=True)    

    # plot
    ax = df_actual.plot(label='Observed', figsize=(9, 7))
    df_pred.plot(ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Passengers')
    plt.legend()
    plt.show()
    # ---------------------------------------------------------------------------

#TSF_15_using_TF()

