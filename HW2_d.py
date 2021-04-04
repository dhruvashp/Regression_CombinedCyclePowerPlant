# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 10:46:34 2020

@author: DHRUV
"""


"""
HW2

d
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy import stats
import statsmodels.api as sm

df = pd.read_csv('Power_Plant.csv')
print(df)

X = df.iloc[:,0:4]
y = df.iloc[:,4]

print(X)
print(y)

reg = LinearRegression()
reg.fit(X,y)
y_predicted = reg.predict(X)
print(y_predicted)
MSE = mean_squared_error(y,y_predicted)
print('MSE for multi-regression model is : \n', MSE)

"""
As can be seen the MSE of the multi-regression model using all the predictors in obtained
to be 20.76, less than that obtained by any of the single predictor based regression models

Thus, multi-regression model seems to fit the data much better, as seen from the MSE perspective

The performance under MSE is only for MSE-Training, as we've used the entire dataset for the
training purpose

"""

"""
Null Hypothesis Test for the predictors
"""

beta0 = reg.intercept_
beta = reg.coef_

print('Beta 0 for this regression is : \n', beta0)
print('Other Beta for this regression are : \n', beta)

beta_AT = beta[0]
beta_V = beta[1]
beta_AP = beta[2]
beta_RH = beta[3]

print('Beta for AT is : \n', beta_AT)
print('Beta for V is : \n', beta_V)
print('Beta for AP is : \n', beta_AP)
print('Beta for RH is : \n', beta_RH)

"""
The calculation of various t-statistics can be manually done similar to how it was done in
HW2_c file for each variable. Using that p-value can be quite easily obtained and then 
either of the two hypothesis tests can be performed. 

However rather than elongating the code, code that would be repeated from HW2_c file, I've used
statsmodels to obtain the t-values, p-values and the F-statistic.

It's been assumed that in-built libraries can be used

"""

X_sm = sm.add_constant(X)      
est = sm.OLS(y, X_sm)
est_data = est.fit()
print(est_data.summary())

t_reference = stats.t.ppf(1-0.025, 9563)
print('The reference t-value is : \n', t_reference)
print('The reference F-statistic (degree of freedom (4,9563) and alpha 0.05) value is : \n2.3728 (using an online calculator)')


"""

t-test and p-values

From either the t-test or the p-values, it can be quite easily seen that all the
predictors are relevant.

Thus the null hypothesis can be rejected for all the predictors and all the predictors 
can be called statistically relevant

F-test

From the F-test the F-value obtained is much larger than the reference F calculated via an 
online calculator. Thus we can say that at least one of the predictors is relevant and that 
we can reject the hypothesis : "ALL the beta's are null". Thus we do have some linear relationship
between the prediction and the predictor in line with the actual data.

"""

"""
Summary

Using all predictors gives us a lower MSE than using just one

The null hypothesis can be rejected for all predictors using t-test

The null hypothesis ("ALL beta's are null") can be rejected using F-test

"""

