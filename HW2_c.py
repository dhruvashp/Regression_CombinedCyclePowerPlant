# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 03:31:50 2020

@author: DHRUV
"""


"""
HW2
c

Assumption : For c no test/training data splitting is mentioned, thus we'll train the model with the
entire dataset and we'll also take our entire dataset to be our test data set. This does make the errors
same over the test and training data (entire dataset here) and thus errors, wherever calculated
should be presumed to be either.

All MSE's and other errors over entire dataset

"""
"""
FOR AT
"""
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math
from scipy import stats

df=pd.read_csv('Power_Plant.csv')
print(df)
X_AT = df.iloc[:,0].values
y = df.iloc[:,4].values
X_AT = X_AT.reshape(-1,1)
y = y.reshape(-1,1)
print(X_AT)
print(y)
reg_AT = LinearRegression()
reg_AT.fit(X_AT,y)
y_AT_predicted = reg_AT.predict(X_AT)
print(y_AT_predicted)

MSE_AT = mean_squared_error(y,y_AT_predicted)
print(MSE_AT)
print('The MSE for AT is : \n', MSE_AT)
total_samples = 9568
RSS_AT = MSE_AT*total_samples
RSE_AT = math.sqrt(RSS_AT/(total_samples - 2))
print(RSS_AT)
print(RSE_AT)

X_AT_flat = X_AT.flatten()
print(X_AT_flat)
X_AT_mean = np.mean(X_AT_flat)
print(X_AT_mean)
i=0
spread_X_AT = 0
vector = np.arange(0,total_samples)
for i in vector:
    spread_X_AT = spread_X_AT + (X_AT_flat[i] - X_AT_mean)**2

SE2_b1_AT = (((RSE_AT)**2)/(spread_X_AT))
SE2_b0_AT = ((1/total_samples)+(((X_AT_mean)**2)/(spread_X_AT)))*((RSE_AT)**2)

print('Standard Error Squared for AT for beta 1 is : \n', SE2_b1_AT)
print('Standard Error Squared for AT for beta 0 is : \n', SE2_b0_AT)

beta0_AT = reg_AT.intercept_
beta1_AT_i = reg_AT.coef_
beta1_AT = beta1_AT_i.flatten()

print('Beta 0 for AT is : \n', beta0_AT)
print('Beta 1 for AT is : \n', beta1_AT)

"""
For AT to ascertain the extent of statistical relationship between predictor and prediction,
we'll perform the hypothesis testing for beta1_AT. Again we assume that we're only checking
for statistical relevance of beta1_AT. We can also perform the hypothesis testing on
beta0_AT, however since it is not connected to any predictors we'll only perform the test on
beta1_AT. Also confidence interval will be taken to be 95%
"""

t_beta1_AT = (beta1_AT) / (math.sqrt(SE2_b1_AT))
print('The t-statistic for Beta 1 for AT is : \n', t_beta1_AT)
t_reference = stats.t.ppf(1-0.025, 9566)
print('The reference t value is : \n', t_reference)

if t_beta1_AT > t_reference or t_beta1_AT < -t_reference:
    print('\nNull hypothesis for Beta 1 for AT can be rejected, thus AT is a relevant predictor')
else:
    print('\n Null hypothesis cannot be rejected, AT not a relevant predictor')

plt.plot(X_AT, y, 'ro', markersize=1)
plt.plot(X_AT, y_AT_predicted, 'bo', markersize=1)
plt.xlabel('AT')
plt.ylabel('y, red = actual, blue = predicted')
plt.show()

"""
As can be seen

(i) From the hypothesis test

AT predictor is relevant. The null hypothesis was rejected.

(ii) From the plots

AT predictor's predicted y values seem to follow quite closely the plot of actual
y against AT. The line follows the mean value of y around each AT value. Thus AT is a 
statistically relevant predictor for our prediction y

"""

"""
FOR V
"""

X_V = df.iloc[:,1].values
X_V = X_V.reshape(-1,1)
print(X_V)
reg_V = LinearRegression()
reg_V.fit(X_V,y)
y_V_predicted = reg_V.predict(X_V)
print(y_V_predicted)

MSE_V = mean_squared_error(y,y_V_predicted)
print(MSE_V)
print('The MSE for V is : \n', MSE_V)
total_samples = 9568
RSS_V = MSE_V*total_samples
RSE_V = math.sqrt(RSS_V/(total_samples - 2))
print(RSS_V)
print(RSE_V)

X_V_flat = X_V.flatten()
print(X_V_flat)
X_V_mean = np.mean(X_V_flat)
print(X_V_mean)
i=0
spread_X_V = 0
for i in vector:
    spread_X_V = spread_X_V + (X_V_flat[i] - X_V_mean)**2

SE2_b1_V = (((RSE_V)**2)/(spread_X_V))
SE2_b0_V = ((1/total_samples)+(((X_V_mean)**2)/(spread_X_V)))*((RSE_V)**2)

print('Standard Error Squared for V for beta 1 is : \n', SE2_b1_V)
print('Standard Error Squared for V for beta 0 is : \n', SE2_b0_V)

beta0_V = reg_V.intercept_
beta1_V_i = reg_V.coef_
beta1_V = beta1_V_i.flatten()

print('Beta 0 for V is : \n', beta0_V)
print('Beta 1 for V is : \n', beta1_V)

"""
For V to ascertain the extent of statistical relationship between predictor and prediction,
we'll perform the hypothesis testing for beta1_V. Again we assume that we're only checking
for statistical relevance of beta1_V. We can also perform the hypothesis testing on
beta0_V, however since it is not connected to any predictors we'll only perform the test on
beta1_V. Also confidence interval will be taken to be 95%
"""

t_beta1_V = (beta1_V) / (math.sqrt(SE2_b1_V))
print('The t-statistic for Beta 1 for V is : \n', t_beta1_V)
print('The reference t value is : \n', t_reference)

if t_beta1_V > t_reference or t_beta1_V < -t_reference:
    print('\nNull hypothesis for Beta 1 for V can be rejected, thus V is a relevant predictor')
else:
    print('\n Null hypothesis cannot be rejected, V not a relevant predictor')

plt.plot(X_V, y, 'ro', markersize=1)
plt.plot(X_V, y_V_predicted, 'bo', markersize=1)
plt.xlabel('V')
plt.ylabel('y, red = actual, blue = predicted')
plt.show()

"""
As can be seen

(i) From the hypothesis test

V predictor is relevant. The null hypothesis was rejected.

(ii) From the plots

V predictor's predicted y values seem to follow quite closely the plot of actual
y against V. The line follows the mean value of y around each V value. Thus V is a 
statistically relevant predictor for our prediction y. However we can clearly see 
from the plots of V and y/y-predicted that the spread around the predicted value of y
of the actual y value is much more compared to the curves for AT. Thus V surely is a 
valid predictor however the overall fit of V, despite being quite accurate, will
have a tendency of predicting y values that may be at a larger distance from the actual
y values at that point, at least when compared with curves for AT and the prediction obtained
off of AT

"""

"""
FOR AP

"""

X_AP = df.iloc[:,2].values
X_AP = X_AP.reshape(-1,1)
print(X_AP)
reg_AP = LinearRegression()
reg_AP.fit(X_AP,y)
y_AP_predicted = reg_AP.predict(X_AP)
print(y_AP_predicted)

MSE_AP = mean_squared_error(y,y_AP_predicted)
print(MSE_AP)
print('The MSE for AP is : \n', MSE_AP)
total_samples = 9568
RSS_AP = MSE_AP*total_samples
RSE_AP = math.sqrt(RSS_AP/(total_samples - 2))
print(RSS_AP)
print(RSE_AP)

X_AP_flat = X_AP.flatten()
print(X_AP_flat)
X_AP_mean = np.mean(X_AP_flat)
print(X_AP_mean)
i=0
spread_X_AP = 0
for i in vector:
    spread_X_AP = spread_X_AP + (X_AP_flat[i] - X_AP_mean)**2

SE2_b1_AP = (((RSE_AP)**2)/(spread_X_AP))
SE2_b0_AP = ((1/total_samples)+(((X_AP_mean)**2)/(spread_X_AP)))*((RSE_AP)**2)

print('Standard Error Squared for AP for beta 1 is : \n', SE2_b1_AP)
print('Standard Error Squared for AP for beta 0 is : \n', SE2_b0_AP)

beta0_AP = reg_AP.intercept_
beta1_AP_i = reg_AP.coef_
beta1_AP = beta1_AP_i.flatten()

print('Beta 0 for AP is : \n', beta0_AP)
print('Beta 1 for AP is : \n', beta1_AP)

"""
For AP to ascertain the extent of statistical relationship between predictor and prediction,
we'll perform the hypothesis testing for beta1_AP. Again we assume that we're only checking
for statistical relevance of beta1_AP. We can also perform the hypothesis testing on
beta0_AP, however since it is not connected to any predictors we'll only perform the test on
beta1_AP. Also confidence interval will be taken to be 95%
"""

t_beta1_AP = (beta1_AP) / (math.sqrt(SE2_b1_AP))
print('The t-statistic for Beta 1 for AP is : \n', t_beta1_AP)
print('The reference t value is : \n', t_reference)

if t_beta1_AP > t_reference or t_beta1_AP < -t_reference:
    print('\nNull hypothesis for Beta 1 for AP can be rejected, thus AP is a relevant predictor')
else:
    print('\n Null hypothesis cannot be rejected, AP not a relevant predictor')

plt.plot(X_AP, y, 'ro', markersize=1)
plt.plot(X_AP, y_AP_predicted, 'bo', markersize=1)
plt.xlabel('AP')
plt.ylabel('y, red = actual, blue = predicted')
plt.show()

"""
As can be seen

(i) From the hypothesis test

AP predictor is relevant. The null hypothesis was rejected.

(ii) From the plots

Again we can draw similar conclusion's as have been drawn before. Surely AP is relevant statistically
as it passed the hypothesis test. However as can be clearly seen in the plot the spread of y_actual
is much, much more than both in AT and V. Thus despite the relevance of the predictor, the
predictor definitely will predict values of y that will skew around actual y with larger distances.

"""

"""
FOR RH
"""
X_RH = df.iloc[:,3].values
X_RH = X_RH.reshape(-1,1)
print(X_RH)
reg_RH = LinearRegression()
reg_RH.fit(X_RH,y)
y_RH_predicted = reg_RH.predict(X_RH)
print(y_RH_predicted)

MSE_RH = mean_squared_error(y,y_RH_predicted)
print(MSE_RH)
print('The MSE for RH is : \n', MSE_RH)
total_samples = 9568
RSS_RH = MSE_RH*total_samples
RSE_RH = math.sqrt(RSS_RH/(total_samples - 2))
print(RSS_RH)
print(RSE_RH)

X_RH_flat = X_RH.flatten()
print(X_RH_flat)
X_RH_mean = np.mean(X_RH_flat)
print(X_RH_mean)
i=0
spread_X_RH = 0
for i in vector:
    spread_X_RH = spread_X_RH + (X_RH_flat[i] - X_RH_mean)**2

SE2_b1_RH = (((RSE_RH)**2)/(spread_X_RH))
SE2_b0_RH = ((1/total_samples)+(((X_RH_mean)**2)/(spread_X_RH)))*((RSE_RH)**2)

print('Standard Error Squared for RH for beta 1 is : \n', SE2_b1_RH)
print('Standard Error Squared for RH for beta 0 is : \n', SE2_b0_RH)

beta0_RH = reg_RH.intercept_
beta1_RH_i = reg_RH.coef_
beta1_RH = beta1_RH_i.flatten()

print('Beta 0 for RH is : \n', beta0_RH)
print('Beta 1 for RH is : \n', beta1_RH)

"""
For RH to ascertain the extent of statistical relationship between predictor and prediction,
we'll perform the hypothesis testing for beta1_RH. Again we assume that we're only checking
for statistical relevance of beta1_RH. We can also perform the hypothesis testing on
beta0_RH, however since it is not connected to any predictors we'll only perform the test on
beta1_RH. Also confidence interval will be taken to be 95%
"""

t_beta1_RH = (beta1_RH) / (math.sqrt(SE2_b1_RH))
print('The t-statistic for Beta 1 for RH is : \n', t_beta1_RH)
print('The reference t value is : \n', t_reference)

if t_beta1_RH > t_reference or t_beta1_RH < -t_reference:
    print('\nNull hypothesis for Beta 1 for RH can be rejected, thus RH is a relevant predictor')
else:
    print('\n Null hypothesis cannot be rejected, RH not a relevant predictor')

plt.plot(X_RH, y, 'ro', markersize=1)
plt.plot(X_RH, y_RH_predicted, 'bo', markersize=1)
plt.xlabel('RH')
plt.ylabel('y, red = actual, blue = predicted')
plt.show()


"""
Again we have

(i) From Hypothesis testing, RH is a relevant predictor. We rejected the null hypothesis


(ii) The same argument as previous arguments. Here the spread is the largest among other predictors
and overall value of y predicted will be skewed from actual y values by larger distances. However
the prediction via RH predictor still somewhat relevant and more or less a good approximate.

"""

"""

Overall Conclusions 

Of all the individual predictors
AT seems best, then we have V, then AP and finally RH. This order has been chosen due to the
overall obtained MSE's for each and the spread of the actual values around the predictions made. However
all predictors seperately pass the hypothesis testing and are thus all relevant.

"""

"""

The outliers have been discussed in a seperate file

"""




"""
This code (in the comments portion) is explained thus
T is df with only AT column
T_sm adds a constant 1 column for stats model linear regression to include a constant beta 0
est does the estimation of the model
est_data gives t-models, etc. for the fit

This code is only for cross-verification of our manually obtained standard errors and t-values

When this code was run the values of t-error and standard error obtained for beta1_AT manually 
and via stats model was same, as can be checked by removing this code block from the comments section

Removing it from the comments section won't affect execution of other code portion, which
is why different variables were selected

"""

"""

T = df.drop(columns = ['AP','V','RH','PE'])
y_m = df.drop(columns = ['AT','AP','V','RH'])
T_sm = sm.add_constant(T)      
est = sm.OLS(y_m, T_sm)
est_data = est.fit()
print(est_data.summary())

"""



