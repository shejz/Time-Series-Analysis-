
# -------------------------------------------------------------
# Time Series Forecasting I using Linear model: y = b * x + a  |
# -------------------------------------------------------------

# Load library
library(DBI)
library(corrgram)
library(caret)
library(gridExtra)
library(ggpubr)
library(forecast)
library(ggplot2)

# Process in parallel on Windows
library(doParallel) 
cl <- makeCluster(detectCores(), type='PSOCK')
registerDoParallel(cl)

# For MAC OSX and Unix like System
# library(doMC)
# registerDoMC(cores = 4)

# Load the Datasets: 
dataSet <- read.csv("international-airline-passengers.csv", header = TRUE, sep = ',')
colnames(dataSet)

# Print top 10 rows in the dataset
head(dataSet, 10)
# Print last 10 rows in the dataset
tail(dataSet, 10)
# Dimention of dataset
dim(dataSet)
# Check data types of each column
table(unlist(lapply(dataSet, class)))

plot.ts(dataSet[, c(2)])
plot.ts(dataSet["passengers"])

# Covert the dataset to a vector i.e. creating "y" variable
dataSet  <- as.numeric(dataSet[, c("passengers")])

# Create "x" variable as row numbers / names
names(dataSet) <- 1:length(dataSet)

df <- cbind(read.table(text = names(dataSet)), dataSet)
x = df$V1; y = df$dataSet

# Define the Linear model: y = b * x + a
Model = lm(y ~ x)

# Summarise the fitted model 
summary(Model)

# Summarise the r_squared for actual and fitted data 
r2 <- cor(fitted(Model), y)^2
summary(Model)$r.squared 
print(r2)

# Generate the equations
paste('y =', coef(Model)[[2]], '* x', '+', coef(Model)[[1]])

# Generate the trendline and fitted values
tendency  = coef(Model)[[2]] * x + coef(Model)[[1]]; print(tendency)

# Generate the forecast in the future time horizon
steps = 25
x_in_Future <- (length(x)+1) : (length(x)+steps)

forecastedValues  = coef(Model)[[2]]  * x_in_Future + coef(Model)[[1]];  print(forecastedValues)

# Plotting Observed versus Predicted
res <- stack(data.frame(Observed = c(y,forecastedValues), 
                        Predicted = c(tendency, forecastedValues)))
res <- cbind(res, x = rep(c(x,x_in_Future) , 2))

require("lattice")
g1 <- xyplot(values ~ x, data = res, group = ind, auto.key = TRUE, grid = TRUE,
             type=c("p","l"))

library(gridExtra)
grid.arrange(g1, nrow = 1)


# -----------------------------------------------------------------
# Time Series Forecasting using Polynimial model of order 1        |
# -----------------------------------------------------------------

# Define the Polynomial model: y ~ poly(x,1, raw = TRUE)
Model  = lm(y  ~ poly(x,1, raw = TRUE))

# Summarise the fitted model
summary(Model)
# Summarise the r_squared for actual and fitted data 
summary(Model)$r.squared
r2 <- cor(fitted(Model), y)^2 
print(r2)

# Generate the equation
paste('y =', coef(Model)[[2]], '* x', '+', coef(Model)[[1]])

# Generate the trendline and fitted values
tendency  = coef(Model)[[2]] * x + coef(Model)[[1]];   print(tendency)

# Generate the forecast in the future time horizon
steps = 25
x_in_Future <- (length(x)+1) : (length(x)+steps)

forecastedValues = coef(Model)[[2]]  * x_in_Future + coef(Model)[[1]];  print(forecastedValues)

# Plotting Observed versus Predicted
res <- stack(data.frame(Observed = c(y,forecastedValues), 
                        Predicted = c(tendency, forecastedValues)))
res <- cbind(res, x = rep(c(x,x_in_Future) , 2))


g1 <- xyplot(values ~ x, data = res, group = ind, auto.key = TRUE, grid = TRUE,
             type=c("p","l"))


grid.arrange(g1, nrow = 1)

# -----------------------------------------------------------------
# Time Series Forecasting using Polynimial model of order 2        |
# -----------------------------------------------------------------

# Define the Polynomial model: y ~ poly(x,2, raw = TRUE)
Model = lm(y ~ poly(x,2, raw = TRUE))

# Summarise the fitted model
summary(Model)
# Summarise the r_squared for actual and fitted data 
summary(Model)$r.squared
r2 <- cor(fitted(Model), y)^2; print(r2)

# Generate the trendline and fitted values
tendency  = coef(Model)[[3]] * x^2 + coef(Model)[[2]] * x + coef(Model)[[1]]
print(tendency)

# Generate the forecast in the future time horizon
steps = 25
x_in_Future <- (length(x)+1) : (length(x)+steps)

forecastedValues  = coef(Model)[[3]]  * x_in_Future^2 + coef(Model)[[2]]  * x_in_Future + coef(Model)[[1]]; print(forecastedValues)

# Plotting Observed versus Predicted
res <- stack(data.frame(Observed = c(y,forecastedValues), 
                        Predicted = c(tendency, forecastedValues)))
res <- cbind(res, x = rep(c(x,x_in_Future) , 2))


g1 <- xyplot(values ~ x, data = res, group = ind, auto.key = TRUE, grid = TRUE,
             type=c("p","l"))
grid.arrange(g1, nrow = 1)

# ------------------------------------------------------------------------------------


# -----------------------------------------------------------------
# Time Series Forecasting using Polynimial model of order 3        |
# -----------------------------------------------------------------

# Define the Polynomial model: y ~ poly(x,3, raw = TRUE)
Model = lm(y ~ poly(x,3, raw = TRUE))

# Summarise the fitted model
summary(Model)
# Summarise the r_squared for actual and fitted data 
summary(Model)$r.squared; r2 <- cor(fitted(Model), y)^2; print(r2)

# Generate the trendline and fitted values
tendency = coef(Model)[[4]] * x^3 + coef(Model)[[3]] * x^2 + coef(Model)[[2]] * x + coef(Model)[[1]]
print(tendency)

# Generate the forecast in the future time horizon
steps = 25
x_in_Future <- (length(x)+1) : (length(x)+steps)

forecastedValues = coef(Model)[[4]] * x_in_Future^3 + coef(Model)[[3]] * x_in_Future^2 + coef(Model)[[2]] * x_in_Future + coef(Model)[[1]]  
print(forecastedValues)

# Plotting Observed versus Predicted
res <- stack(data.frame(Observed = c(y,forecastedValues), 
                        Predicted = c(tendency, forecastedValues)))
res <- cbind(res, x = rep(c(x,x_in_Future) , 2))


g1 <- xyplot(values ~ x, data = res, group = ind, auto.key = TRUE, grid = TRUE,
             type=c("p","l"))

grid.arrange(g1, nrow = 1)


# -----------------------------------------------------------------
# Time Series Forecasting using Polynimial model of order 4        |
# -----------------------------------------------------------------

# Define the Polynomial model: y ~ poly(x,4, raw = TRUE)
Model  = lm(y  ~ poly(x,4, raw = TRUE))

# Summarise the fitted model
summary(Model)
# Summarise the r_squared for actual and fitted data 
summary(Model)$r.squared; r2 <- cor(fitted(Model), y)^2; print(r2)

# Generate the trendline and fitted values
tendency = coef(Model)[[5]] * x^4 + coef(Model)[[4]] * x^3 + coef(Model)[[3]] * x^2 + coef(Model)[[2]] * x + coef(Model)[[1]]
print(tendency)

# Generate the forecast in the future time horizon
steps = 25
x_in_Future <- (length(x)+1) : (length(x)+steps)

forecastedValues = coef(Model)[[5]] * x_in_Future^4 + coef(Model)[[4]] * x_in_Future^3 + coef(Model)[[3]] * x_in_Future^2 + coef(Model)[[2]] * x_in_Future + coef(Model)[[1]]  
print(forecastedValues)

# Plotting Observed versus Predicted
res <- stack(data.frame(Observed = c(y,forecastedValues), 
                        Predicted = c(tendency, forecastedValues)))
res <- cbind(res, x = rep(c(x,x_in_Future) , 2))

g1 <- xyplot(values ~ x, data = res, group = ind, auto.key = TRUE, grid = TRUE,
             type=c("p","l"))

grid.arrange(g1, nrow = 1)

# -----------------------------------------------------------------
# Time Series Forecasting using Polynimial model of order 5        |
# -----------------------------------------------------------------

# Define the Polynomial model: y ~ poly(x,5, raw = TRUE)
Model = lm(y  ~ poly(x,5, raw = TRUE))

# Summarise the fitted model
summary(Model)
# Summarise the r_squared for actual and fitted data 
summary(Model)$r.squared; r2 <- cor(fitted(Model), y)^2; print(r2)

# Generate the trendline and fitted values
tendency = coef(Model)[[6]] * x^5 + coef(Model)[[5]] * x^4 + coef(Model)[[4]] * x^3 + coef(Model)[[3]] * x^2 + coef(Model)[[2]] * x + coef(Model)[[1]]
print(tendency)

# Generate the forecast in the future time horizon
steps = 25
x_in_Future <- (length(x)+1) : (length(x)+steps)

forecastedValues = coef(Model)[[6]] * x_in_Future^5 + coef(Model)[[5]] * x_in_Future^4 + coef(Model)[[4]] * x_in_Future^3 + coef(Model)[[3]] * x_in_Future^2 + coef(Model)[[2]] * x_in_Future + coef(Model)[[1]]
print(forecastedValues)

# Plotting Observed versus Predicted
res <- stack(data.frame(Observed = c(y,forecastedValues), 
                        Predicted = c(tendency, forecastedValues)))
res <- cbind(res, x = rep(c(x,x_in_Future) , 2))

g1 <- xyplot(values ~ x, data = res, group = ind, auto.key = TRUE, grid = TRUE,
             type=c("p","l"))

grid.arrange(g1, nrow = 1)

# -----------------------------------------------------------------
# Time Series Forecasting using Polynimial model of order 6        |
# -----------------------------------------------------------------

# Define the Polynomial model: y ~ poly(x,6, raw = TRUE)
Model = lm(y  ~ poly(x,6, raw = TRUE))

# Summarise the fitted model
summary(Model)
# Summarise the r_squared for actual and fitted data 
summary(Model)$r.squared; r2 <- cor(fitted(Model), y)^2; print(r2)

# Generate the trendline and fitted values
tendency = coef(Model)[[7]] * x^6 + coef(Model)[[6]] * x^5 + coef(Model)[[5]] * x^4 + coef(Model)[[4]] * x^3 + coef(Model)[[3]] * x^2 + coef(Model)[[2]] * x + coef(Model)[[1]]
print(tendency)

# Generate the forecast in the future time horizon
x_in_Future <- (length(x)+1) : (length(x)+25)

forecastedValues = coef(Model)[[7]] * x_in_Future^6 + coef(Model)[[6]] * x_in_Future^5 + coef(Model)[[5]] * x_in_Future^4 + coef(Model)[[4]] * x_in_Future^3 + coef(Model)[[3]] * x_in_Future^2 + coef(Model)[[2]] * x_in_Future + coef(Model)[[1]]
print(forecastedValues)

# Plotting Observed versus Predicted
res <- stack(data.frame(Observed = c(y,forecastedValues), 
                        Predicted = c(tendency, forecastedValues)))
res <- cbind(res, x = rep(c(x,x_in_Future) , 2))

g1 <- xyplot(values ~ x, data = res, group = ind, auto.key = TRUE, grid = TRUE,
             type=c("p","l"))

grid.arrange(g1, nrow = 1)


# -----------------------------------------------------------------
# Time Series Forecasting using Logarithimic model                 |
# -----------------------------------------------------------------

# Define the model: y ~ log(x)
Model  = lm(y ~ log(x))

# Summarise the fitted model
summary(Model)
# Summarise the r_squared for actual and fitted data 
summary(Model)$r.squared; r2 <- cor(fitted(Model), y)^2; print(r2)

# Generate the equation
paste('y =', coef(Model)[[2]], '* log(x)', '+',  coef(Model)[[1]])

# Generate the trendline and fitted values
tendency  =  coef(Model)[[2]]  * log(x) + coef(Model)[[1]];  print(tendency)

# Generate the forecast in the future time horizon
x_in_Future <- (length(x)+1) : (length(x)+25)

forecastedValues = coef(Model)[[2]]  * log(x_in_Future) + coef(Model)[[1]]
print(forecastedValues)

# Plotting Observed versus Predicted
res <- stack(data.frame(Observed = c(y,forecastedValues), 
                        Predicted = c(tendency, forecastedValues)))
res <- cbind(res, x = rep(c(x,x_in_Future) , 2))

g1 <- xyplot(values ~ x, data = res, group = ind, auto.key = TRUE, grid = TRUE,
             type=c("p","l"))

grid.arrange(g1, nrow = 1)


# -----------------------------------------------------------------
# Time Series Forecasting using Simple Exponential Smoothing       |
# using HoltWinters Function : WHERE beta=FALSE, gamma=FALSE       |
# -----------------------------------------------------------------

# ******************************************************************
# using Original data
dataSet_forecasts  <- HoltWinters(dataSet, beta=FALSE, gamma=FALSE)

# fitted model
dataSet_forecasts$fitted;  plot(dataSet_forecasts)

# Forecast in Future time horizon
forecast  <- forecast:::forecast.HoltWinters(dataSet_forecasts, h = 25)

# Print and Visualise Forecasted Values
print(forecast)
print(forecast$residuals)
forecast:::plot.forecast(forecast)

# -----------------------------------------------------------------
# Holt's Exponential Smoothing                                     |   
# using HoltWinters Function : WHERE gamma=FALSE                   |
# -----------------------------------------------------------------

# ******************************************************************
# using Original data
dataSet_forecasts <- HoltWinters(dataSet, gamma=FALSE)

# fitted model
dataSet_forecasts$fitted;  plot(dataSet_forecasts)

# Forecast in Future time horizon
forecast  <- forecast:::forecast.HoltWinters(dataSet_forecasts, h = 25)

# Print and Visualise Forecasted Values
print(forecast)
print(forecast$residuals)
forecast:::plot.forecast(forecast)

# --------------------------------------------------------------
# ARIMA model : ARIMA(1,0,0) = First Order Autoregressive Model |
# --------------------------------------------------------------

# ARIMA Model 
fit_arima  <- arima(dataSet,  order = c(1,0,0), optim.method = "BFGS", method = "CSS-ML") #"CSS-ML", "ML", "CSS"
checkresiduals(fit_arima) 

# Calculate R squared value : r2
r2 <- cor(fitted(fit_arima),  dataSet)^2;  print(r2)

x <- dataSet            # actual
y <- fitted(fit_arima)  # predicted

# plot actual vs predicted
ts.plot(x, y, 
        gpars = list(col = c("black", "red")))
legend("topleft", legend = c("Actual", "Predicted"), col = c("black", "red"), lty = 1)

# Plotting Observed and Predicted with CI
forecast <- forecast:::forecast.Arima(fit_arima, h = 25)
print(forecast); autoplot(forecast(fit_arima))
plot(forecast)

# --------------------------------------------------------------
# ARIMA model : ARIMA(0,1,0) = Random Walk Model                |
# --------------------------------------------------------------

# ARIMA Model 
fit_arima  <- arima(dataSet,  order = c(0,1,0))
print(fit_arima)
checkresiduals(fit_arima) 

# Calculate R squared value : r2
r2 <- cor(fitted(fit_arima),  dataSet)^2;  print(r2)

x <- dataSet            # actual
y <- fitted(fit_arima)  # predicted

# plot actual vs predicted
ts.plot(x, y, 
        gpars = list(col = c("black", "red")))
legend("topleft", legend = c("Actual", "Predicted"), col = c("black", "red"), lty = 1)

# Plotting Observed and Predicted with CI
forecast <- forecast:::forecast.Arima(fit_arima, h = 25)
print(forecast); autoplot(forecast(fit_arima))


# ----------------------------------------------------------------------------
# ARIMA model : ARIMA(1,1,0) = Differenced First Ordered Autoregressive Model |
# ----------------------------------------------------------------------------

# Finding Auto-correlation in the series for selecting a Candidate ARIMA model 
# Using ACF()
acf(dataSet, lag.max=20)   

# Finding Partial Auto-correlation in the series for selecting a Candidate ARIMA model 
# Using PACF()
pacf(dataSet, lag.max=20)  

# ARIMA Model 
fit_arima  <- arima(dataSet,  order = c(1,1,0)); checkresiduals(fit_arima) 

# Calculate R squared value : r2
r2 <- cor(fitted(fit_arima),  dataSet)^2;  print(r2)

x <- dataSet            # actual
y <- fitted(fit_arima)  # predicted

# plot actual vs predicted
ts.plot(x, y, 
        gpars = list(col = c("black", "red")))
legend("topleft", legend = c("Actual", "Predicted"), col = c("black", "red"), lty = 1)

# Plotting Observed and Predicted with CI
forecast <- forecast:::forecast.Arima(fit_arima, h = 25)
print(forecast); autoplot(forecast(fit_arima))


# ----------------------------------------------------------------
# ARIMA model : ARIMA(0,1,1) = Simple Exponential Smoothing Model |
# ----------------------------------------------------------------

# Finding Auto-correlation in the series for selecting a Candidate ARIMA model 
# Using ACF()
acf(dataSet, lag.max=20)   

# Finding Partial Auto-correlation in the series for selecting a Candidate ARIMA model 
# Using PACF()
pacf(dataSet, lag.max=20)  

# ARIMA Model 
fit_arima  <- arima(dataSet,  order = c(0,1,1)); checkresiduals(fit_arima) 

# Calculate R squared value : r2
r2 <- cor(fitted(fit_arima),  dataSet)^2;  print(r2)

x <- dataSet            # actual
y <- fitted(fit_arima)  # predicted

# plot actual vs predicted
ts.plot(x, y, 
        gpars = list(col = c("black", "red")))
legend("topleft", legend = c("Actual", "Predicted"), col = c("black", "red"), lty = 1)

# Plotting Observed and Predicted with CI
forecast <- forecast:::forecast.Arima(fit_arima, h = 25)
print(forecast); autoplot(forecast(fit_arima))

# --------------------------------------------------------------
# ARIMA model : ARIMA(0,2,1) = Linear Exponential Smoothing Model
# --------------------------------------------------------------

# Finding Auto-correlation in the series for selecting a Candidate ARIMA model 
# Using ACF()
acf(dataSet, lag.max=20)   

# Finding Partial Auto-correlation in the series for selecting a Candidate ARIMA model 
# Using PACF()
pacf(dataSet, lag.max=20)  

# ARIMA Model 
fit_arima  <- arima(dataSet,  order = c(0,2,1)); checkresiduals(fit_arima) 

# Calculate R squared value : r2
r2 <- cor(fitted(fit_arima),  dataSet)^2;  print(r2)

x <- dataSet            # actual
y <- fitted(fit_arima)  # predicted

# plot actual vs predicted
ts.plot(x, y, 
        gpars = list(col = c("black", "red")))
legend("topleft", legend = c("Actual", "Predicted"), col = c("black", "red"), lty = 1)

# Plotting Observed and Predicted with CI
forecast <- forecast:::forecast.Arima(fit_arima, h = 22)
print(forecast); autoplot(forecast(fit_arima))

# -----------------------------------------------------------------------------
# ARIMA model : ARIMA(1,1,2) = Damped trend Linear Exponential Smoothing Model |
# -----------------------------------------------------------------------------

# Finding Auto-correlation in the series for selecting a Candidate ARIMA model 
# Using ACF()
acf(dataSet, lag.max=20)   

# Finding Partial Auto-correlation in the series for selecting a Candidate ARIMA model 
# Using PACF()
pacf(dataSet, lag.max=20)  

# ARIMA Model 
fit_arima  <- arima(dataSet,  order = c(1,1,2)); checkresiduals(fit_arima) 

# Calculate R squared value : r2
r2 <- cor(fitted(fit_arima),  dataSet)^2;  print(r2)

x <- dataSet            # actual
y <- fitted(fit_arima)  # predicted

# plot actual vs predicted
ts.plot(x, y, 
        gpars = list(col = c("black", "red")))
legend("topleft", legend = c("Actual", "Predicted"), col = c("black", "red"), lty = 1)

# Plotting Observed and Predicted with CI
forecast <- forecast:::forecast.Arima(fit_arima, h = 25)
print(forecast); autoplot(forecast(fit_arima))

# --------------------------------------------------------------
# ARIMA model : Using auto.ARIMA                                |
# --------------------------------------------------------------

# Finding Auto-correlation in the series for selecting a Candidate ARIMA model 
# Using ACF()
acf(dataSet, lag.max=20)   

# Finding Partial Auto-correlation in the series for selecting a Candidate ARIMA model 
# Using PACF()
pacf(dataSet, lag.max=20)  

# Find an appropiate ARIMA model using auto.arima(dataset)
auto.arima(dataSet)

# ARIMA Model 
fit_arima  <- arima(dataSet,  order = c(4,1,2)); checkresiduals(fit_arima) 

# Calculate R squared value : r2
r2 <- cor(fitted(fit_arima),  dataSet)^2;  print(r2)

x <- dataSet            # actual
y <- fitted(fit_arima)  # predicted

# plot actual vs predicted
ts.plot(x, y, 
        gpars = list(col = c("black", "red")))
legend("topleft", legend = c("Actual", "Predicted"), col = c("black", "red"), lty = 1)

# Plotting Observed and Predicted with CI
forecast <- forecast:::forecast.Arima(fit_arima, h = 25)
print(forecast); autoplot(forecast(fit_arima))

# --------------------------------------------------------------------------------------
# ARIMA model with seasonality : Using Seasonal random walk model: ARIMA(0,0,0)s(0,1,0) |
# --------------------------------------------------------------------------------------

# Finding Auto-correlation in the series for selecting a Candidate ARIMA model 
# Using ACF()
acf(dataSet, lag.max=20)   

# Finding Partial Auto-correlation in the series for selecting a Candidate ARIMA model 
# Using PACF()
pacf(dataSet, lag.max=20)  

# ARIMA Model with seasonality
fit_arima  <- arima(dataSet,  order = c(0,0,0), seasonal = c(0,1,0)); 
checkresiduals(fit_arima) 

# Calculate R squared value : r2
r2 <- cor(fitted(fit_arima),  dataSet)^2;  print(r2)

x <- dataSet            # actual
y <- fitted(fit_arima)  # predicted

# plot actual vs predicted
ts.plot(x, y, 
        gpars = list(col = c("black", "red")))
legend("topleft", legend = c("Actual", "Predicted"), col = c("black", "red"), lty = 1)

# Plotting Observed and Predicted with CI
forecast <- forecast:::forecast.Arima(fit_arima, h = 25)
print(forecast); autoplot(forecast(fit_arima))

# ---------------------------------------------------------------------------------------
# ARIMA model with seasonality : Using Seasonal random trend model: ARIMA(0,1,0)s(0,1,0) |
# ---------------------------------------------------------------------------------------

# Finding Auto-correlation in the series for selecting a Candidate ARIMA model 
# Using ACF()
acf(dataSet, lag.max=20)   

# Finding Partial Auto-correlation in the series for selecting a Candidate ARIMA model 
# Using PACF()
pacf(dataSet, lag.max=20)  

# ARIMA Model with seasonality
fit_arima  <- arima(dataSet,  order = c(0,1,0), seasonal = c(0,1,0)); 
checkresiduals(fit_arima) 

# Calculate R squared value : r2
r2 <- cor(fitted(fit_arima),  dataSet)^2;  print(r2)

x <- dataSet            # actual
y <- fitted(fit_arima)  # predicted

# plot actual vs predicted
ts.plot(x, y, 
        gpars = list(col = c("black", "red")))
legend("topleft", legend = c("Actual", "Predicted"), col = c("black", "red"), lty = 1)

# Plotting Observed and Predicted with CI
forecast <- forecast:::forecast.Arima(fit_arima, h = 25)
print(forecast); autoplot(forecast(fit_arima))


# ------------------------------------------------------------------------------------
# ARIMA model with seasonality : Using General seasonal ARIMA models: (0,1,1)s(0,1,1) |
# ------------------------------------------------------------------------------------

# Finding Auto-correlation in the series for selecting a Candidate ARIMA model 
# Using ACF()
acf(dataSet, lag.max=20)   

# Finding Partial Auto-correlation in the series for selecting a Candidate ARIMA model 
# Using PACF()
pacf(dataSet, lag.max=20)  

# ARIMA Model with seasonality
fit_arima  <- arima(dataSet,  order = c(0,1,1), seasonal = c(0,1,1)); 
checkresiduals(fit_arima) 

# Calculate R squared value : r2
r2 <- cor(fitted(fit_arima),  dataSet)^2;  print(r2)

x <- dataSet            # actual
y <- fitted(fit_arima)  # predicted

# plot actual vs predicted
ts.plot(x, y, 
        gpars = list(col = c("black", "red")))
legend("topleft", legend = c("Actual", "Predicted"), col = c("black", "red"), lty = 1)

# Plotting Observed and Predicted with CI
forecast <- forecast:::forecast.Arima(fit_arima, h = 25)
print(forecast); autoplot(forecast(fit_arima))

# -------------------------------------------------------------------
# ARIMA model with seasonality : Using ARIMA models: (4,1,3)s(0,2,1) |
# -------------------------------------------------------------------

# Finding Auto-correlation in the series for selecting a Candidate ARIMA model 
# Using ACF()
acf(dataSet, lag.max=20)   

# Finding Partial Auto-correlation in the series for selecting a Candidate ARIMA model 
# Using PACF()
pacf(dataSet, lag.max=20)  

# ARIMA Model with seasonality
fit_arima  <- arima(dataSet,  order = c(4,1,3), seasonal = c(0,2,1)); 
checkresiduals(fit_arima) 

# Calculate R squared value : r2
r2 <- cor(fitted(fit_arima),  dataSet)^2;  print(r2)

x <- dataSet            # actual
y <- fitted(fit_arima)  # predicted

# plot actual vs predicted
ts.plot(x, y, 
        gpars = list(col = c("black", "red")))
legend("topleft", legend = c("Actual", "Predicted"), col = c("black", "red"), lty = 1)

# Plotting Observed and Predicted with CI
forecast <- forecast:::forecast.Arima(fit_arima, h = 25)
print(forecast); autoplot(forecast(fit_arima))


# --------------------------------------------------------------------------------
fit_arima <- arima(dataSet$P_Uniques, order = c(0,1,1), seasonal = c(2,0,3),
                   optim.method = "BFGS", method = "CSS-ML") #"CSS-ML", "ML", "CSS"
#"Nelder-Mead", "BFGS", "CG", "L-BFGS-B", "SANN","Brent" 
checkresiduals(fit_arima)
autoplot(fit_arima)
# --------------------------------------------------------------------------------


# --------------------------------------------------------------
# Using NNetAR model                                            |
# --------------------------------------------------------------

fit_nnetar <- nnetar(dataSet, repeats = 100, size = 4)
print(fit_nnetar)

checkresiduals(fit_nnetar)

# Calculate R squared value : r2
r <- cor(fitted(fit_nnetar)[14:length(dataSet)], dataSet[14:length(dataSet)])
r2 <- cor(fitted(fit_nnetar)[14:length(dataSet)], dataSet[14:length(dataSet)])^2
print(r2)

x <- dataSet            # actual
y <- fitted(fit_nnetar)  # predicted

# plot actual vs predicted
ts.plot(x, y, 
        gpars = list(col = c("black", "red")))
legend("topleft", legend = c("Actual", "Predicted"), col = c("black", "red"), lty = 1)


# Plotting Observed and Predicted with CI
forecast_ <- forecast:::forecast.nnetar(fit_nnetar, h = 25, level = c(75,95), PI = TRUE)
autoplot(forecast_)

par(mfrow=c(1,1))
plot(forecast_)

# --------------------------------------------------------------
# Using MLP model                                               |
# --------------------------------------------------------------
# load library
library(nnfor)

fit_mlp <- mlp(ts(dataSet), hd = 4, lags = 12, reps = 20)
print(fit_mlp)

# Calculate R squared value : r2
r <- cor(fitted(fit_mlp)[1:(length(dataSet)-13)], dataSet[1:(length(dataSet)-13)])
r2 <- cor(fitted(fit_mlp)[1:(length(dataSet)-13)], dataSet[1:(length(dataSet)-13)])^2
print(r2)

# Plotting Observed and Predicted with CI
forecast_ <- forecast:::forecast(fit_mlp, h = 25)
autoplot(forecast_)

par(mfrow=c(1,1))
plot(forecast_)

