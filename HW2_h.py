# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 08:31:50 2020

@author: DHRUV
"""


"""
HW2
h
"""

"""
Assumptions

All interaction terms will be assumed to be only of order 2. We'll train and test first the model
that includes all interactions, all square terms in addition with the original terms.

Normal Terms = 4
Squared Terms = 4
Interaction Terms = 6
Total predictor terms = 14

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from scipy.stats import t
from statsmodels.tools.eval_measures import mse
from sklearn.model_selection import train_test_split

df = pd.read_csv('Power_Plant.csv')
print(df)
y = df['PE']
print(y)
poly = PolynomialFeatures(degree=2)
df_features = df.drop(columns = ['PE'])
print(df_features)
df_modified_features = pd.DataFrame(poly.fit_transform(df_features))
print(df_modified_features)
final_features = df_modified_features.drop(columns = 0)
print(final_features)
final_features.columns = ['AT', 'V', 'AP', 'RH', 'AT2', 'AT*V', 'AT*AP','AT*RH','V2','V*AP','V*RH','AP2','AP*RH','RH2']
print(final_features)
final_features['PE'] = y
final_features_appended = final_features
print(final_features_appended)
features_appended_train, features_appended_test = train_test_split(final_features_appended, train_size = 0.7)
print(features_appended_train)
print(features_appended_test)
features_appended_train.reset_index(drop=True, inplace=True)
features_appended_test.reset_index(drop=True, inplace=True)
print(features_appended_train)
print(features_appended_test)
features_train = features_appended_train.drop(columns = ['PE'])
features_test = features_appended_test.drop(columns = ['PE'])
y_train = features_appended_train['PE']
y_test = features_appended_test['PE']
print('The randomly selected 70% training features dataset is : \n', features_train)
print('The randomly selected 30% test feature dataset is : \n', features_test)
print('The randomly selected (training feature corresponding) 70% training output is : \n',y_train)
print('The randomly selected (test feature corresponding) 30% test output is : \n',y_test)

"""
The above code selects random test and train features and outputs, making sure that
features and outputs still correspond to each other. The reason being, before being
shuffled the output column was appended back. Then random shuffling and selection was done.
Then features and outputs again extracted, after again the row index was reset

"""

reg = LinearRegression()
reg.fit(features_train,y_train)
y_train_predicted = reg.predict(features_train)
y_test_predicted = reg.predict(features_test)
MSE_train = mean_squared_error(y_train,y_train_predicted)
MSE_test = mean_squared_error(y_test,y_test_predicted)

print('The y_train_predicted is : \n',pd.DataFrame(y_train_predicted,columns =['y_train_predicted']))
print('The y_test_predicted is : \n',pd.DataFrame(y_test_predicted,columns=['y_test_predicted']))
print('The MSE_train is : \n',MSE_train)
print('The MSE_test is : \n',MSE_test)

print('The intercept obtained for this model is : \n',reg.intercept_)
print('The coefficients obtained for the predictors in an array in their respective order is : \n',reg.coef_)

features_train_sm = sm.add_constant(features_train)      
est = sm.OLS(y_train, features_train_sm)
est_data = est.fit()
print(est_data.summary())


"""

Few points to note -

statsmodels, used to obtain the overall data summary, obtains this summary over the training
data. It is given the y_train and features_train, and used the predicted y for the training data
against the actual y_train and the features_train to obtain various statistics. We'll assume
this doesn't affect our analysis or our results much.

As the test and train sets are randomly generated, there is some disparity over different
test and train sets, in terms of evaluated model parameters and various other quantities.

Despite this minor disparities the following behaviour was consistent (given p boundary is
0.05 corresponding to 95% confidence interval) -

AT, V, AT*AP, V2, V*AP, V*RH all had large p values, beyond p boundary at 0.05

Thus they should be removed

However AT, V have relevant interaction terms. Thus via hierarchical principle those terms/predictors
can't be removed from the model.

Thus, we'll remove the following : AT*AP, V2, V*AP, V*RH

And we'll keep the rest predictors


Summary : AT*AP, V2, V*AP, V*RH are removed
AT, V due to hierarchical principle stay in


"""

features_train_new = features_train.drop(columns = ['AT*AP','V2','V*AP','V*RH'])
features_test_new = features_test.drop(columns = ['AT*AP','V2','V*AP','V*RH'])

"""
y_train and y_test obviously stay the same

"""
reg_new = LinearRegression()

reg_new.fit(features_train_new,y_train)
y_train_predicted_new = reg_new.predict(features_train_new)
y_test_predicted_new = reg_new.predict(features_test_new)
MSE_train_new = mean_squared_error(y_train,y_train_predicted_new)
MSE_test_new = mean_squared_error(y_test,y_test_predicted_new)

print('The y_train_predicted_new with the new model is : \n', pd.DataFrame(y_train_predicted_new,columns=['y_train_predicted_new']))
print('The y_test_predicted_new with the new model is : \n',pd.DataFrame(y_test_predicted_new,columns=['y_test_predicted_new']))
print('The MSE_train_new with the new model is : \n',MSE_train_new)
print('The MSE_test_new with the new model is : \n', MSE_test_new)

print('The new intercept is : \n',reg_new.intercept_)
print('The coefficients of the new model, in that order is : \n', reg_new.coef_)

features_train_new_sm = sm.add_constant(features_train_new)      
est_new = sm.OLS(y_train, features_train_new_sm)
est_data_new = est_new.fit()
print(est_data_new.summary())



"""
A few notes in summary -

The new model has all p values zero making all the predictors relevant and statistically significant.

As the model takes in random test and train data sets in every single run of the 
program due to train_test_split command in regards with the MSE we say the following -

MSE train and test, both for the old and the new model is found to be around 17-18

Thus MSE for both models, for both the test and the training data sets, was about same
throughout in most program runs I found.

MSE_train, MSE_test for new and old model around 17-18

Note that MSE for the normal multi-regression model was around 20

Thus in terms of MSE both these models perform better than the normal multi-regression model

"""

