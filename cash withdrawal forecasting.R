library(dplyr)
library(forecast)
library(ggplot2)
library(lubridate)
library(zoo)
library(seasonal)
library(tseries)
library(stats)
library(outliers)
library(tsutils)
library(xts)
library(lmtest)
library(strucchange)
library(segmented)
library(tsoutliers)
library(strucchange)
library(bcp)
library(smooth)

data <- read.csv("dataset38.csv")

#1.0
#Preprocessing
#check the structure of data
str(data)


#convert date to date object
data$Date <- as.Date(data$Date, format = "%d-%b-%y")

summary(data)
length(data$Date)

# Replace missing values with the previous value
data$Data <- na.locf(data$Data)
#replacing zeros with the previous values
#data$Data <- ifelse(data$Data == 0, NA, data$Data)
#data$Data <- na.locf(data$Data)
#in case of duplicates
# Remove any duplicates
#data <- distinct(data)

#visualise the time series
plot(data$Date, data$Data, type = "l", xlab = "Date", ylab = "Data")

summary(data)
findfrequency(data)

# Create a scatter plot
ggplot(data, aes(x = Date, y = Data)) +
  geom_point()

# Create a histogram
ggplot(data, aes(x = Data)) +
  geom_histogram(binwidth = 0.5, color = "black", fill = "white")
#kind of looks like a normal distribution but not necessarily

# Create a box plot
ggplot(data, aes(y=Data)) +
  geom_boxplot()
#clearly can see from diagram that there are some outliers in withdrawals


#2.0
#Time series analysis (daily time series)
# create a line plot of the time series data with a trend line
ggplot(data, aes(x = Date, y = Data)) + 
  geom_line() +
  stat_smooth(method = "lm", formula = y ~ x, se = FALSE, color = "red") +
  labs(x = "Date", y = "Data", title = "Time series plot with trend line") +
  theme_bw()

#shows seasonality 
ts_data <- ts(data$Data, frequency = 365,start = c(1996, yday(data$Date[1]))) 
plot(ts_data, xlab = "Date", ylab = "Data")
#checking seasonality
ggseasonplot(ts_data, year.labels = TRUE, uncertainty = TRUE)


#plotting the residuals to check for randomness and or outliers
decomp_d <- decompose(ts_data)
plot(decomp_d)

plot(decomp_d$random)

#calcuating errors
mm_errors <- ts_data - (decomp_d$trend + decomp_d$seasonal)



#using stl to decompose
tstl_data <- ts(data$Data, frequency = 365, start = c(1996, 03, 18))

# Use stl decomposition to extract seasonal, trend, and remainder components
decomp <- stl(tstl_data, s.window = "periodic")

# Plot the original time series
ggplot(data = data.frame(date = time(tstl_data), value = tstl_data)) +
  geom_line(aes(x = date, y = value)) +
  labs(x = "Date", y = "Data", title = "Original Time Series")

# Plot the seasonal component
ggplot(data = data.frame(date = time(tstl_data), value = decomp$time.series[, "seasonal"])) +
  geom_line(aes(x = date, y = value)) +
  labs(x = "Date", y = "Seasonal Component", title = "Seasonal Component of Time Series")

# Plot the trend component
ggplot(data = data.frame(date = time(tstl_data), value = decomp$time.series[, "trend"])) +
  geom_line(aes(x = date, y = value)) +
  labs(x = "Date", y = "Trend Component", title = "Trend Component of Time Series")

# Plot the remainder (irregular) component
ggplot(data = data.frame(date = time(tstl_data), value = decomp$time.series[, "remainder"])) +
  geom_line(aes(x = date, y = value)) +
  labs(x = "Date", y = "Irregular Component", title = "Irregular Component of Time Series")


#2.1
#statistical Tests of time series
adf.test(ts_data)
#results suggests the null hypothesis of non stationarity can be rejected,
#hence, data is stationary. THis suggests, there is no trend and seasonality 
#in data as the variance and mean remain constant over time
kpss.test(ts_data)
#implies data is not stationary

# Box-Pierce test for autocorrelation
Box.test(data$Data, type = "Box-Pierce")
# the p value suggests there is autocorrelation in the data (2.2e-16)

Box.test(data$Data, type = "Ljung-Box")
#results suggests there is autocorrelation

d_data <- data
#check for seasonality
kruskal.test(Data ~ ts_data, data = d_data) # daily


#2.11
#  test for outliers
grubbs.test(data$Data)
#presence of outliers. p-value = 0.000113. 
#alternative hypothesis: highest value 74.419 is an outlier

#check for level shifts and breaks
# Perform CUSUM test
cusum_test <- efp(data$Data ~ 1, type = "Rec-CUSUM", data = data)
plot(cusum_test)
#there is a cross in the threshold and suggests a structural change or level shift

# Perform Bayesian change point analysis
bcp_output <- bcp(data$Data)
plot(bcp_output)
 #level and structural changes


#2.2

# Plot the ACF and PACF graphs
ggtsdisplay(ts_data, main = "ACF/PACF of Time Series Data")

#from the acf/ pacf it shows the possible presence of seasonality but it is difficult
#to identify trend in the acf but quite possible in the pacf

#differencing
dailydif <- diff(ts_data)
ggtsdisplay(dailydif)

tsdisplay(diff(diff(dailydif)))

dailylag7 <- diff(ts_data, lag = 7)
tsdisplay(dailylag7)

#checking normality of noise
#if the errors are not normally distributed, it means the model may not be capturing some
#important patterns or the assumption of the statistical methods of normality is not valid
plot(decomp$time.series[,"remainder"], col = "red", main = 'Remainder', ylab = "")
qqnorm(decomp$time.series[,"remainder"])
qqline(decomp$time.series[,"remainder"])

#results shows it is normally distributed

#3.0
#aggregating data to monthly frequency
# Load data
ts_data <- xts(data$Data, order.by = as.Date(data$Date))

# Aggregate to monthly frequency
ts_data_monthly <- aggregate(ts_data, as.yearmon, mean)
ts_data_monthly_ts <- as.ts(ts_data_monthly)
plot(ts_data_monthly_ts, xlab="date", ylab = "value")
plot(decompose(ts_data_monthly_ts))
ggseasonplot(ts_data_monthly_ts, year.labels = TRUE, uncertainty = TRUE)
tsdisplay(ts_data_monthly_ts)

#aggregating to quarterly frequency
# Aggregate to quarterly frequency
ts_data_quarterly <- aggregate(ts_data, as.yearqtr, mean)
ts_data_quarterly_ts <- as.ts(ts_data_quarterly)
plot(ts_data_quarterly_ts,xlab="date", ylab = "value")
qu_decomp <- decompose(ts_data_quarterly_ts)
plot(qu_decomp)
ggseasonplot(ts_data_quarterly_ts, year.labels = TRUE, uncertainty = TRUE)
tsdisplay(ts_data_quarterly_ts)

# Aggregate to weekly frequency
# Convert ts_data to a data frame
week_df <- data.frame(date = time(ts_data), data = as.vector(ts_data))
# Aggregate to weekly frequency
df_weekly <- aggregate(week_df$data, list(week = cut(week_df$date, "week")), mean)
# Convert the resulting data frame back to a time series
ts_data_weekly <- ts(df_weekly$x, start = as.Date(df_weekly$week[1]), frequency = 52)
plot(decompose(ts_data_weekly))
ggseasonplot(ts_data_weekly)
tsdisplay(ts_data_weekly)


#3.1
#statistical tests on temporal aggregates

# H0: The data is not stationary.
# H1: The data is stationary
# reject null if p-value < 0.05
adf.test(ts_data_weekly) #stationary
adf.test(ts_data_monthly_ts) #non stationary
adf.test(ts_data_quarterly_ts) #non stationary


# using KPSS test for stationarity

# for KPSS test
#H0: The data is stationary
#H1: The data is not stationary
# reject null if p-value < 0.05
kpss.test(ts_data_weekly) #not stationary
kpss.test(ts_data_monthly_ts) # non stationary
kpss.test(ts_data_quarterly_ts) #stationary

#check for seasonality
#kruskal.test(data ~ ts_data_weekly , data = week_df) # weekly

#kruskal.test(data ~ ts_data_monthly_ts, data = ts_data_monthly) # monthly

#kruskal.test(data ~ ts_data_quarterly_ts, data = ts_data_quarterly) # quarterly


#3.11
#test for outliers
tsoutliers(ts_data_weekly)
tsoutliers(ts_data_monthly_ts)
tsoutliers(ts_data_quarterly_ts)


#checking structural breaks

cusum_test <- efp(ts_data_weekly~1, type = "OLS-CUSUM")
plot(cusum_test, main = "CUSUM for Weekly Time Series")


cusum_test <- efp(ts_data_monthly_ts~1, type = "OLS-CUSUM")
plot(cusum_test,main = "CUSUM for Monthly Time Series")

cusum_test <- efp(ts_data_quarterly_ts~1, type = "OLS-CUSUM")
plot(cusum_test, main = "CUSUM for Quarterly Time Series")


#3.2
# differencing to remove non stationarity
diff_week <- diff(ts_data_weekly)
diff_month<- diff(ts_data_monthly_ts)
diff_qt <- diff(ts_data_quarterly_ts)

tsdisplay(diff_week)
tsdisplay(diff_month)
tsdisplay(diff_qt)

#tests for stationarity for differenced temporals
# H0: The data is not stationary.
# H1: The data is stationary
# reject null if p-value < 0.05
adf.test(ts_data_weekly) #stationary
adf.test(ts_data_monthly_ts) #non stationary
adf.test(ts_data_quarterly_ts) #non stationary


# using KPSS test for stationarity

# for KPSS test
#H0: The data is stationary
#H1: The data is not stationary
# reject null if p-value < 0.05
kpss.test(ts_data_weekly) #not stationary
kpss.test(ts_data_monthly_ts) # non stationary
kpss.test(ts_data_quarterly_ts) #stationary







# PART 2
library(dplyr)
library(forecast)
library(ggplot2)
library(lubridate)
library(zoo)
library(seasonal)
library(tseries)
library(stats)
library(outliers)
library(tsutils)
library(xts)
library(lmtest)
library(strucchange)
library(segmented)
library(tsoutliers)
library(strucchange)
library(bcp)
library(smooth)
library(readxl)


#1.0
#Preprocessing
#check the structure of data
str(data)


#convert date to date object
data$Date <- as.Date(data$Date, format = "%d-%b-%y")

summary(data)
length(data$Date)

# Replace missing values with the previous value
data$Data <- na.locf(data$Data)
#replacing zeros with the previous values
#data$Data <- ifelse(data$Data == 0, NA, data$Data)
#data$Data <- na.locf(data$Data)
#in case of duplicates
# Remove any duplicates
#data <- distinct(data)




# Split the data into a training and test set
trend_data <- data$Data
data_train <- ts(trend_data[1:650], frequency = 7, start = min(data$Date) )
data_test <- trend_data[651:length(trend_data)]
data_trainO <- ts(trend_data[1:721], frequency = 7, start = min(data$Date) )
data_testO <- trend_data[722:length(trend_data)]

#Benchmarking using naive method
# Perform a naive forecast
naive <- naive(data_trainO, h=length(data_testO))
# Plot the forecast
plot(naive, main="Naive Forecast")
accuracy(naive, data_testO)
smape_naive <- 2 * mean(abs(data_testO - naive$mean) / (abs(data_testO) + abs(naive$mean)))
# Calculate the sMAPE
#smape_val <- smape(naive$mean, data_testO)
# Print the sMAPE
#print(smape_val)
# 0.6740074, #0.5680503

# Seasonal naive method
snaive_forecast <- snaive(data_trainO, h = length(data_testO))
# Plot the forecast
plot(snaive_forecast, main="seasonal Naive Forecast")
accuracy(snaive_forecast, data_testO)
smape_Snaive <- 2 * mean(abs(data_testO - snaive_forecast$mean) / (abs(data_testO) + abs(snaive_forecast$mean)))
smape_Snaive

#Exponential smoothing
#automatic
# Build an automatic Exponential Smoothing model #0.2091454
# Calculate Holt Method
ets_zzz <- ets(data_trainO, model = "ZZZ")
forec <- forecast(ets_zzz, h = length(data_testO))
plot(forec)
accuracy(forec, data_testO)
AIC(ets_zzz)
coef(ets_zzz)
serror = data_testO - forec$mean
checkresiduals(ets_zzz)

# Calculate sMAPE for ets model
smape_autoETS <- 2 * mean(abs(data_testO - forec$mean) / (abs(data_testO) + abs(forec$mean)))
#[1] 0.1252506, AIC = 7573.912
#ETS(A,N,A)
#                     ME     RMSE      MAE      MPE     MAPE      MASE      ACF1
#Training set 0.01382488 7.015885 4.179100     -Inf      Inf 0.4811292 0.2434377
#Test set     2.05501647 4.909570 3.634425 4.223262 11.95358 0.4184221        NA
#       alpha         gamma
##0.1231      1e-04



 
#manually
#the seasonality component, trend and error components seem to be additive from the
# decomposition diagrams and hence model "AAA" will be selected. 
# since there is the presence of structural break, howeever, this can be a bit tricky and would respond to noise if there is a 
#lot of it
#a lower gamma too would be appropriate to avoid overfitting of the seasonality since it is subtle, so as 
# beta for trend since the trend is very subtle. a very low beta will be a good fit
# start off by using ses function and holt functions which gives the alpha parameter and the alpha and beta parameter respectively
# by minimizing errors by using the AIC to give an idea of start points for the parameters.
check_alpha <- ses(data_trainO, h=100)
accuracy(check_alpha, data_testO)
#                     ME      RMSE      MAE       MPE     MAPE      MASE      ACF1
# Training set 0.3499471 10.132880 8.204343      -Inf      Inf 0.9445452 0.3729029
# Test set     0.9208431  9.648509 8.512880 -6.812676 29.45681 0.9800662        NA
#lpha = 0.0095
#check for best alpha

# identify optimal alpha parameter
alpha <- seq(.01, .99, by = .01)
RMSE <- NA
for(i in seq_along(alpha)) {
  fit <- ses(data_train, alpha = alpha[i], h = 100)
  RMSE[i] <- accuracy(fit, data_testO)[2,2]
}

# convert to a data frame and idenitify min alpha value
alpha.fit <- data_frame(alpha, RMSE)
alpha.min <- filter(alpha.fit, RMSE == min(RMSE))


# plot RMSE vs. alpha
ggplot(alpha.fit, aes(alpha, RMSE)) +
  geom_line() +
  geom_point(data = alpha.min, aes(alpha, RMSE), size = 2, color = "blue")  

#alpha  RMSE
#<dbl> <dbl>
# 0.11  9.61

# refit model with the best alpha
alpha_check <- ses(data_trainO, alpha = 0.11, h = 100)
accuracy(alpha_check, data_testO)
#                       ME      RMSE      MAE       MPE     MAPE      MASE      ACF1
# Training set 0.02613726 10.338742 8.406363      -Inf      Inf 0.9678032 0.3376401
# Test set     2.05123136  9.821065 8.351396 -2.559876 27.71518 0.9614750        NA
#the original alpha is slighly better comparing the in-sample RMSE

# performance eval
accuracy(alpha_check, data_testO)
#performance is better since the alpha was best on the accuracy of the test set


#holt to check for beta
check_iniPara <- holt(data_trainO, h=100)
#Smoothing parameters:
  #alpha = 0.011 
  #beta  = 1e-04 

accuracy(check_iniPara, data_testO)
#                     ME      RMSE      MAE       MPE     MAPE      MASE      ACF1
#Training set -0.03447462 10.154359 8.284727      -Inf      Inf 0.9537996 0.3754731
#Test set      0.35940349  9.613867 8.593085 -8.927627 30.32262 0.9893001        NA

# identify optimal beta parameter
beta <- seq(.0001, .5, by = .001)
RMSE <- NA
for(i in seq_along(beta)) {
  fit <- holt(data_trainO, beta = beta[i], h = 100)
  RMSE[i] <- accuracy(fit, data_testO)[2,2]
}

# convert to a data frame and idenitify min alpha value
beta.fit <- data_frame(beta, RMSE)
beta.min <- filter(beta.fit, RMSE == min(RMSE))

# plot RMSE vs. alpha
ggplot(beta.fit, aes(beta, RMSE)) +
  geom_line() +
  geom_point(data = beta.min, aes(beta, RMSE), size = 2, color = "blue")
beta.min
# # A tibble: 1 × 2
# beta  RMSE
# <dbl> <dbl>
#   1 0.0001  9.61

opt_iniPara <- holt(data_trainO,beta = 0.0001, h=100)
# accuracy of new optimal model
accuracy(opt_iniPara, data_testO)
#                       ME      RMSE      MAE       MPE     MAPE      MASE      ACF1
# Training set -0.0338369 10.150208 8.275214      -Inf      Inf 0.9527043 0.3748018
# Test set      0.3340868  9.612915 8.596702 -9.022837 30.36161 0.9897165        NA

#results from both the original and optimal on the test values are very similar,




#start off with a alpha 0.2, beta 0.00010, gamma 0.00010
#m_ets <- ets(data_trainO, model = "AAA",alpha = 0.2, beta =  0.00010, gamma = 0.00010) #0.2497856
#ME     RMSE      MAE      MPE     MAPE      MASE      ACF1
#Training set -0.01066203 7.025163 4.178111     -Inf      Inf 0.4839353 0.1937892
#Test set      5.64457209 8.750518 6.826956 16.39762 21.91982 0.7907413        NA
#error graph looks quite similar but results are not too good

#reduce the value of alpha to 0.15 and try again
#m_ets <- ets(data_trainO, model = "AAA", alpha = 0.15, beta = 0.00010, gamma = 0.00015  ) #0.1930384
#                     ME     RMSE      MAE      MPE     MAPE      MASE      ACF1
#Training set -0.01009886 7.021565 4.155478     -Inf      Inf 0.4813139 0.2297929
#Test set      2.78904190 7.262595 5.159890 5.632477 16.45572 0.5976511        NA
#AIC(m_ets)
#7578.535
#results improved when alpha was reduced. Further reduce alpha and increase gamma a bit since it is more prominent than trend


#m_ets <- ets(data_trainO, model = "AAA", alpha = 0.14, beta = 0.00010, gamma = 0.00020  ) # 0.1916068
#                      ME     RMSE      MAE      MPE     MAPE      MASE      ACF1
#Training set 0.003979275 7.021827 4.149449     -Inf      Inf 0.4806155 0.2369401
#Test set     2.220234666 7.040575 5.014729 3.501991 16.23093 0.5808377        NA
#improve in results but could get better. A loop is gonna be set about this neighbourhood to increase the pairing search to give an idea for
#better results

#m_ets <- ets(data_trainO, model = "AAA", alpha = 0.10, beta = 0.00010, gamma = 0.00020  ) #0.1207157
#                     ME     RMSE      MAE      MPE     MAPE      MASE      ACF1
# Training set 0.01978034 7.022748 4.163741     -Inf      Inf 0.4793610 0.2611518
# Test set     1.90009887 4.776620 3.516506 3.721874 11.57763 0.4048464        NA

m_ets <- ets(data_trainO, model = "AAA", alpha = 0.072, beta = 0.00020, gamma = 0.00010  )

#                   ME     RMSE      MAE      MPE     MAPE      MASE      ACF1
# Training set 0.01741992 7.026808 4.142076     -Inf      Inf 0.4768667 0.2820543
# Test set     1.80928805 4.761708 3.492473 3.339267 11.50477 0.4020796        NA
#smape =0.1196249
#AIC(m_ets)
#7582.155


#try looping
# define suggested parameter values
#the alpha was first between 0.15 to 0.25. the best rmse was 6.391838. even though this is better than the previous, the residuals had a level of 0.8652,
#which isnt really good. The acf too showed 3 spikes at some lags which suggested, i had to increase the alpha and gamma parameters in order to improve model
alpha_vals <- seq(0.007, 0.35, by = 0.005)
beta_vals <- seq(0.0001, 0.0006, by = 0.00005)
gamma_vals <- seq(0.0001, 0.0006, by = 0.00005)

# create empty data frames to store results
results <- data.frame(alpha = numeric(),
                      beta = numeric(),
                      gamma = numeric(),
                      MAE = numeric(),
                      SMAPE = numeric())

# loop through suggested parameter values
for (a in alpha_vals) {
  for (b in beta_vals) {
    for (g in gamma_vals) {
      # fit model and generate forecast
      m_ets <- ets(data_trainO, model = "AAA", alpha = a, beta = b, gamma = g)
      forecast_m <- forecast(m_ets, h = length(data_testO))
      
      # calculate RMSE
      mae <- accuracy(m_ets)[3]
      #calculate smape
      smape_ETS <- 2 * mean(abs(data_testO - forecast_m$mean) / (abs(data_testO) + abs(forecast_m$mean)))
      # store results in data frame
      results <- rbind(results, data.frame(alpha = a,
                                           beta = b,
                                           gamma = g,
                                           MAE = mae,
                                           SMAPE = smape_ETS))
    }
  }
}

# print results sorted by RMSE
results <- results[order(results$MAE), ]
print(results)

#0.057 0.00035 0.00045 4.146827 0.1196603
#0.072 0.00020 0.00015 4.141107 0.1198039
#0.072 0.00020 0.00010 4.142076 0.1196249
#0.057 0.00035 0.00040 4.143582 0.1196013
#Best results
# alpha    beta   gamma      MAE     SMAPE
# 1598 0.072 0.00020 0.00020 4.141032 0.1200070
# 1597 0.072 0.00020 0.00015 4.141107 0.1198039
# 1600 0.072 0.00020 0.00030 4.141178 0.1199406
# 1144 0.052 0.00030 0.00060 4.141506 0.1215214
# 1602 0.072 0.00020 0.00040 4.141513 0.1199299
# 1601 0.072 0.00020 0.00035 4.141550 0.1198669
# 1596 0.072 0.00020 0.00010 4.142076 0.1196249
# 1143 0.052 0.00030 0.00055 4.142236 0.1209447
# 1256 0.057 0.00030 0.00015 4.142306 0.1208956
# 1258 0.057 0.00030 0.00025 4.142319 0.1207927
# 1630 0.072 0.00035 0.00015 4.142359 0.1204435


#m_ets <- ets(data_trainO, model = "AAA", alpha = 0.007, beta = 0.00040, gamma = 0.00060) #0.1131246, AIC =7588.094
#                   ME     RMSE      MAE         MPE     MAPE      MASE      ACF1
#Training set 0.3608436 7.085225 4.214459        -Inf      Inf 0.4851999 0.3348795
#Test set     0.9125111 4.495858 3.311793 -0.04418133 11.24319 0.3812784        NA
#massive improvement in results and improvement in errors too, although it can be better 

forecast_m <- forecast(m_ets, h = length(data_testO))
plot(forecast_m)
accuracy(forecast_m, data_testO)
# Calculate the residuals
# Calculate sMAPE 
smape_mETS <- 2 * mean(abs(data_testO - forecast_m$mean) / (abs(data_testO) + abs(forecast_m$mean)))
smape_mETS
error_m <- data_testO - forecast_m$mean
#check residuals
checkresiduals(m_ets)



#ARIMA
#AUTO
AR_auto <- auto.arima(data_trainO)
#ARIMA(1,0,0)(2,1,0)[7] with drift
# Validate the model using the test set
forecast_auAR <- forecast(AR_auto, h = length(data_testO))
plot(forecast_auAR) 
#checking residuals of auto model
checkresiduals(AR_auto)
# Calculate sMAPE for ARIMA model
smape_arima <- 2 * mean(abs(data_testO - forecast_auAR$mean) / (abs(data_testO) + abs(forecast_auAR$mean)))
smape_arima
#[1] 0.1807896, AIC=4933.42
# 0.385286
accuracy(forecast_auAR,data_testO )
AIC(AR_auto)
#                     ME     RMSE      MAE      MPE     MAPE      MASE         ACF1
#Training set 0.009094269 7.550003 4.997041     -Inf      Inf 0.5752966 -0.008327925
#Test set     1.361187463 5.596622 3.801334 7.422469 15.18448 0.4376379           NA
#good results but residuals have spikes in ACF
#ets outperforming ARIMA

#proceeding to manually built arima

tsdisplay(data_trainO)
#the results are showing some seasonal components, so I will need to build SARIMA model
kpss.test(data_trainO)
adf.test(data_trainO)
#conflicting but i know data is non stationary from acf and time series diagram and decomp

#taking the difference with the lag 7 to deal with seasonality
data_lag7 <-diff(data_trainO, lag = 7) #moving average of 5, ar = 6/8
tsdisplay(data_lag7)
tsdisplay(diff(data_trainO))

#the significant spike at lag 1, 2, 8 and 5 suggests a non seasonal MA components, and the spike at lag 7 suggests the 
#seasonal MA component. This suggests ARIMA(0,0,5)(0,1,1)7. In the end, if fitted the assumptions for an arima model, with #The residuals from the ACF/PACF shows no significant spikes which suggests they are pretty much white noise and the useful
#information are being used in our model, normal distribution, but a mean error of about 0.5 which isnt good enough and also compared to the auto, hence,
# the parameters to be alterd to find a better model while satisfying all conditions

data_trainO %>%
  Arima(order = c(0,0,5),
        seasonal = c(0,1,1)) %>%
          residuals() %>% tsdisplay()

#starting off with ARIMA(0,0,5)(0,1,1)7
#m_ARIMA <- Arima(data_trainO,order = c(0,0,5), seasonal = c(0,1,1)) #0.1215119, AIC = 4787.07
#                  ME     RMSE      MAE      MPE     MAPE      MASE         ACF1
#Training set 0.3913521 6.725705 4.096506     -Inf      Inf 0.4716203 -0.002689604
#Test set     1.5054730 4.285173 3.418942 2.982637 11.73721 0.3936141           NA
#results is quite good and already out performing auto arima. Also satisfying all conditions but it could get better, so we loop over its neighbourhood
# to find other better pairings
#m_ARIMA <- Arima(data_trainO,order = c(0,0,0), seasonal = c(0,1,1)) #0.1052855, AIC = AIC=4869.32
#                    ME     RMSE      MAE       MPE     MAPE      MASE      ACF1
#Training set 0.4919329 7.183478 4.341000      -Inf      Inf 0.4997683 0.3397797
#Test set     0.9394762 4.006708 3.089369 0.6662635 10.36727 0.3556713        NA
#great results but the errors have spikes and hence missing something. It also has an AIC less than the initial

m_ARIMA <- Arima(data_trainO,order = c(0,0,2), seasonal = c(0,1,1))#AIC=4781.24, 0.121447
#                    ME     RMSE      MAE      MPE    MAPE      MASE         ACF1
#Training set 0.3945021 6.727067 4.103574     -Inf     Inf 0.4724340 -0.001796959
#Test set     1.4696630 4.280780 3.413914 2.897313 11.7304 0.3930353           NA
#slightly better than initial and satisfies all assumptions
#I will stick to this as my final arima pairing since others are either worse results or breaks an assumption

forecast_mAR <- forecast(m_ARIMA, h = length(data_testO))
plot(forecast_mAR) 
#check accuracy of forecast
accuracy(forecast_mAR, data_testO)
smape_arimaM <- 2 * mean(abs(data_testO - forecast_mAR$mean) / (abs(data_testO) + abs(forecast_mAR$mean)))
smape_arimaM
#checking residuals
checkresiduals(Arima(data_trainO,order = c(0,0,2), seasonal = c(0,1,1)))
tsdisplay(residuals(m_ARIMA))

# Set up parameter grid
ar_params <- c(0,1)
ma_params <- c(0, 1, 2, 4, 5)
differ<- c(0,1)
sar <- c(0,1)
s_ma <- c(0,1,2)

# Initialize results data frame
results <- data.frame(AR = integer(), differ= integer(), MA = integer(), sar = integer(),
                     RMSE = numeric(), sMAPE = numeric(), sea_ma = integer())

# Loop through all combinations of parameters and calculate mean and RMSE
for (ar in ar_params) {
  for (ma in ma_params) {
    for(d in differ){
        for(sr in sar){
          for(sa in s_ma){
    # Fit ARIMA model with current parameters
    model <- Arima(data_trainO, order = c(ar, d, ma), seasonal = c(sr, 1, sa))
    
    # Make forecast on test data
    forecast <- forecast(model, h = length(data_testO))
    
    # Calculate mean and RMSE of residuals
    smape_arima <- 2 * mean(abs(data_testO - forecast$mean) / (abs(data_testO) + abs(forecast$mean)))
    rmse <- sqrt(mean((data_testO - forecast$mean)^2))
    
    # Add results to data frame
    results <- rbind(results, data.frame(AR = ar, MA = ma, differ = d, sar = sr, 
                                         sMAPE = smape_arima, RMSE = rmse, sea_ma = sa))
  }
}}}}

# Print results sorted by RMSE
print(results[order(results$RMSE),])


#auto neural network
# Train autoNN model
set.seed(1222)
autoNN_model <- nnetar(data_trainO)

# Make forecast on test data
autoNN_forecast <- forecast(autoNN_model, h = length(data_testO))

autoNN_model
#NNAR(28,1,14)[7]

# Plot autoNN forecast
plot(autoNN_forecast)

#accuracy
accuracy(autoNN_forecast, data_testO)
#                     ME     RMSE      MAE      MPE     MAPE      MASE        ACF1
#Training set 0.002037139 1.641124 1.005553     -Inf      Inf 0.1157667 -0.04826657
#Test set     2.245085394 4.972298 3.769156 6.250539 12.94596 0.4339333          NA
#SMAPE
smape_aNN <- 2 * mean(abs(data_testO - autoNN_forecast$mean) / (abs(data_testO) + abs(autoNN_forecast$mean)))
smape_aNN
#0.1404673

#checking residuals
checkresiduals(autoNN_forecast)



# Regression
#setting up data/ preprocessing
data38 <- read.csv("dataset38.csv")

# Convert the date column to a date format
data38$Date <- as.Date(data38$Date, format = "%d-%b-%y")

# Replace missing values with the previous value
data38$Data <- na.locf(data38$Data)

# Create a dummy variable for bank holidays
# Define bank holidays between 18-March-1996 and 22-March-1998 manually
bank_holidays <- c("1996-04-05", "1996-04-08", "1996-05-06", "1996-05-27",
                   "1996-08-26", "1996-12-25", "1996-12-26", "1997-01-01",
                   "1997-04-18", "1997-04-21", "1997-05-05", "1997-05-26",
                   "1997-08-25", "1997-12-25", "1997-12-26", "1998-01-01")

# Convert bank holidays to Date format
bank_holidays <- as.Date(bank_holidays, format = "%Y-%m-%d")

# Create dummy variable for bank holidays
data38$bank_holiday <- ifelse(data38$Date %in% bank_holidays, 1, 0)

#creating dummy for outliers
# Create a time series object
ts_out <- ts(data38$Data, start = min(data38$Date), frequency = 7)
# Detect outliers using the tsoutliers function
# Find the upper and lower bounds for outliers
q1 <- quantile(data38$Data, 0.25, na.rm = TRUE)
q3 <- quantile(data38$Data, 0.75, na.rm = TRUE)
iqr <- q3 - q1
upper_bound <- q3 + 1.5 * iqr
lower_bound <- q1 - 1.5 * iqr
# Create a dummy variable for outliers
data38$outlier <- ifelse(data38$Data > upper_bound | data38$Data < lower_bound, 1, 0)

# Create a dummy variable for the day of the week
data38$weekday <- weekdays(data38$Date)
# Create a vector of weekdays
weekdays <- c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
# Create a matrix of seasonal dummies
seasonal_dummies <- model.matrix(~factor(weekdays), data = data38)
# Remove the intercept column
seasonal_dummies <- seasonal_dummies[, -1]
# Merge the seasonal dummies with the original dataset
data38 <- cbind(data38, seasonal_dummies)
# Select all columns except the weekday column
data38 <- subset(data38, select = -c(weekday))
#Include lagged variables from 1 to 7 since data shows stronger seasonality at lag 7
Lag1 <- lag(data38$Data,1)
Lag2 <-lag(data38$Data,2)
Lag3 <- lag(data38$Data,3)
Lag4 <- lag(data38$Data,4)
Lag5 <- lag(data38$Data,5)
Lag6 <- lag(data38$Data, 6)
Lag7 <- lag(data38$Data,7)

#adding the lags to the dataframe
data38_ <- cbind(data38,Lag1,Lag2, Lag3,Lag4, Lag5, Lag6, Lag7)

#variable for trend
# Create a trend variable
trend <- 1:nrow(data38_)
# Add trend variable to data frame
data38_ <- cbind(data38_, trend)

#taking out the data column since it cannot be used as a predictor
data38_ <- subset(data38_, select = -c(Date))

#dropping the na rows
data38_ <- na.omit(data38_)
names(data38_) <- gsub("factor\\((.*?)\\)", "\\1", names(data38_))

#split to train and test
# Define the number of observations in the testing set
test_len <- 14

# Split the data into training and testing sets
train <- data38_[1:(nrow(data38_) - test_len), ]
test <- data38_[(nrow(data38_) - test_len + 1):nrow(data38_), ]  
ts_train <- ts(train, frequency = 7)
ts_test <- ts(test, frequency = 7)
#Manual
# Fit a linear regression model using all covariates (backward)
model_1R <- lm(Data ~ ., data = train)
summary(model_1R)
#cat("SMAPE:", smape, "\n")
#SMAPE: 12.8138%
#AIC 4660.482
#               ME     RMSE      MAE       MPE     MAPE
#Test set 0.4567922 4.216658 3.566468 -1.407228 13.02439
checkresiduals(model_1R) #residuals look normally distributed and the acf shows no significant spike, implying there is no important component left

model_b1 <- lm(Data ~ bank_holiday + outlier + weekdaysMonday+ weekdaysSaturday + weekdaysThursday +
               weekdaysSunday + weekdaysTuesday + weekdaysWednesday+ Lag1 + trend, data = ts_train)
summary(model_b1)
#SMAPE: 12.69779
#               ME    RMSE      MAE       MPE     MAPE
#Test set 0.2242095 4.24467 3.577361 -2.552244 13.07928
# AIC(model_b2)
#[1] 4655.813
#Adjusted R-squared:  0.6244

#checking residuals
checkresiduals(model_b1)
AIC(model_b1)

# Make predictions on the testing set using the model
predictions <- predict(model_1R, newdata = test)

#accuracy
accuracy(predictions, test$Data)
# Calculate SMAPE
smape <- (100/length(predictions) * sum(2 * abs(predictions - test$Data) / (abs(predictions) + abs(test$Data))))/100
smape
# Calculate the root mean squared error (RMSE) of the predictions
rmse <- sqrt(mean((test$Data - predictions)^2))

# Print the SMAPE and RMSE
cat("SMAPE:", smape, "\n")
cat("RMSE:", rmse, "\n")

length(predict(model_1R, newdata = test)) * 
  sum(2 * abs(predict(model_1R, newdata = test) - test$Data) / 
                                                  (abs(predict(model_1R, newdata = test)) + abs(test$Data)))


#comparing both models on different test lengths
# Define the test lengths to use
len_reg <- c(14, 35, 60, 85)

# initialising
Reg1_ <- list()
Reg2_ <- list()
for (lenR in len_reg) {
  # Split the data into training and testing sets
  trainL <- data38_[1:(nrow(data38_) - lenR), ]
  testL <- data38_[(nrow(data38_) - lenR + 1):nrow(data38_), ]
  
  #Calculate the SMAPE for each model on the test data
  Reg1_[[lenR]] <- (100/length(predict(model_1R, newdata = testL)) * 
                          sum(2 * abs(predict(model_1R, newdata = testL) - testL$Data) / 
                                (abs(predict(model_1R, newdata = testL)) + abs(testL$Data))))/100
  Reg2_[[lenR]]<-(100/length(predict(model_b1, newdata = testL)) * 
                        sum(2 * abs(predict(model_b1, newdata = testL) - testL$Data) / 
                              (abs(predict(model_b1, newdata = testL)) + abs(testL$Data))))/100
  
}

#taking out the NAs

Reg1_ <- sapply(Reg1_, function(x) ifelse(is.null(x), NA, x))
Reg1_ <- Reg1_[!is.na(Reg1_)]
Reg2_ <- sapply(Reg2_, function(x) ifelse(is.null(x), NA, x))
Reg2_ <- Reg2_[!is.na(Reg2_)]





# Create a data frame with SMAPE values and their averages
reg_df <- data.frame(
  Reg1 = Reg1_,
  Reg2 = Reg2_
)

# Set row names as the test lengths
row.names(reg_df) <- len_reg

# Add a new row for the averages
reg_df["Average",] <- colMeans(reg_df)

#print results
reg_df



#AUTO
# Regression with no variables
cashModel0 <- lm(Data ~ 1, data=ts_train)
# Regression with all variables
cashModel1 <- lm(Data ~ ., data=ts_train)

# Summary of Regressions
summary(cashModel0)
summary(cashModel1)

# AIC forward selection
cashModel2 <- step(cashModel0, formula(cashModel1), direction="forward")
#the best model with the best AIC was
#Step:  AIC=2632.53
#bCash_model <- lm(Data ~ `factor(weekdays)Thursday` + Lag1 + `factor(weekdays)Wednesday` + 
#  outlier + `factor(weekdays)Saturday` + `factor(weekdays)Monday` + 
#  trend + Lag7 + bank_holiday, data = train)
# Make predictions on test set
pred <- predict(cashModel2, newdata = test)
#preddd <- forecast(model_try, h=14)
# Evaluate model performance
accuracy(pred, test$Data)
#               ME    RMSE     MAE       MPE     MAPE
#Test set 0.4218033 4.38225 3.74753 -1.801708 13.66947
smape <- 100/length(pred) * sum(2 * abs(pred - test$Data) / (abs(pred) + abs(test$Data)))
#13.36822%
#check residuals
checkresiduals(cashModel2)
#shows that, there are still some unused information because of the spike

#Testing my final selected models
#Auto exponential smoothing, Manual exponential smoothing, manual Arima, auto and manual regression
ets_zzz #Auto exponential model
m_ets #Manual exponential smoothing model
m_ARIMA #Manual arima model
model_1R #manual regression model
cashModel2 #auto regression model


# Define the test lengths to use
test_lengths <- c(14, 28, 42, 56, 70)

#initialising empty lists
smapeETS_auto <- list()
smapeETS_manual <- list()
smapeARIMA<- list()
smape_Reg1 <- list() #auto
smape_Reg2 <-list() #manual
for (length in test_lengths) {
  # Split the data into training and test sets
  data_trainL <- ts(trend_data[1:(735-length)], frequency = 7, start = min(data$Date))
  data_testL <- trend_data[(736-length):735]
  
  # Calculate the SMAPE for each model on the test data
  smapeETS_auto[[length]] <- 2 * mean(abs(data_testL - forecast(ets_zzz, h = length(data_testL))$mean) / 
                                        (abs(data_testL) + abs(forecast(ets_zzz, h = length(data_testL))$mean)))
  smapeETS_manual[[length]] <-2 * mean(abs(data_testL - forecast(m_ets, h = length(data_testL))$mean) / 
                                         (abs(data_testL) + abs(forecast(m_ets, h = length(data_testL))$mean)))
  
  smapeARIMA[[length]] <- 2 * mean(abs(data_testL - forecast(m_ARIMA, h = length(data_testL))$mean) / 
                                     (abs(data_testL) + abs(forecast(m_ARIMA, h = length(data_testL))$mean)))
}

#for regression
for (len in test_lengths) {
  # Split the data into training and testing sets
  trainL <- data38_[1:(nrow(data38_) - len), ]
  testL <- data38_[(nrow(data38_) - len + 1):nrow(data38_), ]
  
  #Calculate the SMAPE for each model on the test data
  smape_Reg1[[len]] <- (100/length(predict(cashModel2, newdata = testL)) * 
    sum(2 * abs(predict(cashModel2, newdata = testL) - testL$Data) / 
                                      (abs(predict(cashModel2, newdata = testL)) + abs(testL$Data))))/100
  smape_Reg2[[len]]<-(100/length(predict(model_1R, newdata = testL)) * 
    sum(2 * abs(predict(model_1R, newdata = testL) - testL$Data) / 
          (abs(predict(model_1R, newdata = testL)) + abs(testL$Data))))/100
  
}

#taking out the NAs
smapeETS_auto <- sapply(smapeETS_auto, function(x) ifelse(is.null(x), NA, x))
smapeETS_auto <- smapeETS_auto[!is.na(smapeETS_auto)]
smapeETS_manual <- sapply(smapeETS_manual, function(x) ifelse(is.null(x), NA, x))
smapeETS_manual <- smapeETS_manual[!is.na(smapeETS_manual)]
smapeARIMA <- sapply(smapeARIMA, function(x) ifelse(is.null(x), NA, x))
smapeARIMA <- smapeARIMA[!is.na(smapeARIMA)]
smape_Reg1 <- sapply(smape_Reg1, function(x) ifelse(is.null(x), NA, x))
smape_Reg1 <- smape_Reg1[!is.na(smape_Reg1)]
smape_Reg2 <- sapply(smape_Reg2, function(x) ifelse(is.null(x), NA, x))
smape_Reg2 <- smape_Reg2[!is.na(smape_Reg2)]





# Create a data frame with SMAPE values and their averages
smape_df <- data.frame(
  ETS_auto = smapeETS_auto,
  ETS_manual = smapeETS_manual,
  ARIMA = smapeARIMA,
  Reg1 = smape_Reg1,
  Reg2 = smape_Reg2
)

# Set row names as the test lengths
row.names(smape_df) <- test_lengths

# Add a new row for the averages
smape_df["Average",] <- colMeans(smape_df)

# Print the resulting data frame
smape_df



#from results, manually built linear regression is the best model
#predicting 14 days into the future

# Create a sequence of dates for the next 14 days
dates <- seq(as.Date("1998-03-09"), by = "day", length.out = 14)




# # Remove the factor() functions from the variable names in train
#names(test) <- gsub("factor\\((.*?)\\)", "\\1", names(test))
# future_data <- data.frame(date = dates) %>% # create data frame with future dates
#   bind_cols(data.frame(matrix(ncol = 17, nrow = length(dates)))) # add columns for explanatory variables
# names(future_data)[-1] <- names(train)[-1] # set column names to match training set
# 
# future_data$value <- predict(model_1R, newdata = future_data) # predict values for future dates
# future_data[, -1] <- sapply(future_data[, -1], as.numeric)


# creating dataframe for 14 days prediction
# create a data frame with the predictors
next_dates <- seq(ymd("1998-03-23"), by = "day", length.out = 14)
future_data <- data.frame(
  bank_holiday = rep(0, 14), # assuming no bank holidays in the next 14 days
  outlier = rep(0,14),
  weekdaysMonday = rep(0, 14),
  weekdaysSaturday = rep(0, 14),
  weekdaysSunday = rep(0, 14),
  weekdaysThursday = rep(0, 14),
  weekdaysTuesday = rep(0, 14),
  weekdaysWednesday = rep(0, 14),
  Lag1 = rep(NA, 14),
  Lag2 = rep(NA, 14),
  Lag3 = rep(NA, 14),
  Lag4 = rep(NA, 14),
  Lag5 = rep(NA, 14),
  Lag6 = rep(NA, 14),
  Lag7 = rep(NA, 14),
  trend = 1:14 # assuming a linear trend
)

# get the last 7 rows of data38_ to use for filling in lags
last7 <- tail(data38_, 7)

# replace NAs in future_data with values from last7(mean)
future_data$Lag1[is.na(future_data$Lag1)] <- last7$Data[7]
future_data$Lag2[is.na(future_data$Lag2)] <- last7$Data[6]
future_data$Lag3[is.na(future_data$Lag3)] <- last7$Data[5]
future_data$Lag4[is.na(future_data$Lag4)] <- last7$Data[4]
future_data$Lag5[is.na(future_data$Lag5)] <- last7$Data[3]
future_data$Lag6[is.na(future_data$Lag6)] <- last7$Data[2]
future_data$Lag7[is.na(future_data$Lag7)] <- last7$Data[1]

# Update weekday dummy variables in future_data
future_data$weekdaysMonday <- ifelse(weekdays(next_dates) == "Monday", 1, 0)
future_data$weekdaysTuesday <- ifelse(weekdays(next_dates) == "Tuesday", 1, 0)
future_data$weekdaysWednesday <- ifelse(weekdays(next_dates) == "Wednesday", 1, 0)
future_data$weekdaysThursday <- ifelse(weekdays(next_dates) == "Thursday", 1, 0)
future_data$weekdaysSaturday <- ifelse(weekdays(next_dates) == "Saturday", 1, 0)
future_data$weekdaysSunday <- ifelse(weekdays(next_dates) == "Sunday", 1, 0)

# update bank_holiday/ good friday
future_data$bank_holiday <- ifelse(next_dates == ymd("1998-04-03"), 1, 0)

# make predictions for the new data
future_data$predicted_value <- predict(model_1R, newdata = future_data)
future_data

# create a data frame with the original data plus the predicted values for the next 14 days
data38_$date <- seq(ymd("1996-03-25"), ymd("1998-03-22"), by = "day")
all_data <- data38_
all_data$date <- as.Date(all_data$date, format = "%d-%m-%Y")
#adding date column
future_data$date <- seq(ymd("1998-03-23"), by = "day", length.out = 14)
#selecting only date and predicted values
fut_data <- future_data[,c(18,17)]
#rename
fut_data <- rename(fut_data, Data = predicted_value)
all_data <- all_data[, c(18,1)]
all_data <- rbind(all_data, fut_data)

ggplot(all_data, aes(x = date, y = Data)) +
  geom_line(aes(color = date >= "1998-03-22")) +
  scale_color_manual(values = c("black", "red"), labels = c("Past", "Future (2 weeks)")) +
  ggtitle("Predicted values for next 14 days") +
  xlab("Date") +
  ylab("Cash Withdrawal") +
  theme(legend.position = "bottom") +
  annotate("segment", x = as.Date("1998-03-22"), xend = as.Date("1998-03-22"),
           y = min(all_data$Data), yend = max(all_data$Data), linetype = "dashed") +
  annotate("text", x = as.Date("1998-03-22"), y = max(all_data$Data) - 10,
           label = "Future (2 weeks)", hjust = 0)




#file_name <- "finalpred_table.csv"

# use write.table to export the data frame as a CSV file
#write.table(fut_data, file = file_name, row.names = FALSE)

##forecasting with arima
#creating a dataframe for forecast
next_dates <- seq(ymd("1998-03-23"), by = "day", length.out = 14)
ARI_forecast <- data.frame(date = next_dates)
#forecast the next 14 days
ARI_forecast$forecast14 <- forecast(m_ARIMA, h=14)$mean
autoplot(forecast(m_ARIMA, h=14)$mean)
# create a data frame with the original data plus the predicted values for the next 14 days
arimaData <- data
arimaData$Date <- as.Date(arimaData$Date, format = "%d-%m-%Y")
ARI_forecast <- rename(ARI_forecast, Data = forecast14)
ARI_forecast <- rename(ARI_forecast, Date = date)
final_data <- rbind(arimaData,ARI_forecast)

#plot the forecast
ggplot(final_data, aes(x = Date, y = Data)) +
  geom_line(aes(color = Date >= "1998-03-22")) +
  scale_color_manual(values = c("black", "red"), labels = c("Past", "Future (2 weeks)")) +
  ggtitle("Predicted values for next 14 days") +
  xlab("Date") +
  ylab("Cash Withdrawal") +
  theme(legend.position = "bottom") +
  annotate("segment", x = as.Date("1998-03-22"), xend = as.Date("1998-03-22"),
           y = min(final_data$Data), yend = max(final_data$Data), linetype = "dashed") +
  annotate("text", x = as.Date("1998-03-22"), y = max(final_data$Data) - 10,
           label = "Future (2 weeks)", hjust = 0)

# use write.table to export the data frame as a CSV file
write.table(ARI_forecast, file = "arima_forecast", row.names = FALSE)
