# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 15:06:48 2020

@author: DHRUV
"""


"""
HW_2

i
"""
"""
Assumption

Since what constitutes the training data and what constitutes the test data hasn't been
explicitly specified, we'll take 70% of our original dataset as training and the rest as test

We could use train_test_split to generate random such sets each program run-time, however
since that was already done in HW2 part (h), we'll just split the dataset from the top and keep
it same each program run-time. The top 70% will make for training and the bottom 30% for test

"""
"""
RAW FEATURES USED HERE

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from scipy.stats import t
from statsmodels.tools.eval_measures import mse

df = pd.read_csv('Power_Plant.csv')
print(df)
train_i = df[0:6698]
print(train_i)
test_i = df[6698:9568]
print(test_i)
test_i.reset_index(drop = True,inplace = True)
print(test_i)
X_train = train_i.drop(columns=['PE'])
X_test = test_i.drop(columns = ['PE'])
y_train = train_i['PE']
y_test = test_i['PE']

print('The training feature data is : \n',X_train)
print('The test feature data is : \n',X_test)
print('The y_train output is : \n',y_train)
print('The y_test output is : \n',y_test)

train_errors = np.zeros(100)
test_errors = np.zeros(100)

"""
The features and outputs again match (are corresponding)
"""

k = np.arange(1,101)
index = np.arange(0,100)
i=0

for i in index:
    
    neigh=KNeighborsRegressor(n_neighbors = k[i])
    neigh.fit(X_train,y_train)
    y_train_predicted = neigh.predict(X_train)
    y_test_predicted = neigh.predict(X_test)
    MSE_train = mean_squared_error(y_train,y_train_predicted)
    MSE_test = mean_squared_error(y_test,y_test_predicted)
    
    train_errors[i] = MSE_train
    test_errors[i] = MSE_test



train_errors_df = pd.DataFrame(data = train_errors, index = range(1,101), columns = ['train_errors'])
test_errors_df = pd.DataFrame(data = test_errors, index = range(1,101), columns = ['test_errors'])


print(train_errors)
print(test_errors)

print(train_errors_df)
print(test_errors_df)


train_errors_min = np.amin(train_errors)
train_errors_min_index = np.argmin(train_errors)

test_errors_min = np.amin(test_errors)
test_errors_min_index = np.argmin(test_errors)

print('The minimum train error obtained is : \n', train_errors_min)
print('The k value that minimizes the train error is : \n', train_errors_min_index + 1)
print('The minimum test error obtained is : \n',test_errors_min)
print('The k value that minimizes the test error is : \n', test_errors_min_index + 1)


"""

Comments

Minimum train error for k=1 is obviously 0
if we need minimum train error for k > 1, we get it at k = 2 as 5.1139

Minimum test error is obtained for k=6 as 16.70 for which train error is 10.82


Thus, 

Minimum Train Error = 0                   for   k = 1       (Trivial)
Minimum Train Error = 5.1139              for  k > 1, k = 2  

Minimum Test Error = 16.70                for   k = 6
Train Error = 10.82                       for   k = 6


Optimal k (Minimizing Test Error) = k* = 6
Train Error = 10.82 for k = k*
Test Error = 16.70 for k = k*


Thus k = k* = 6 is the best fit point

"""

"""
Plots
"""
k = k.astype('float64')

plt.plot(np.reciprocal(k),train_errors,'r',markersize = 5)
plt.plot(np.reciprocal(k),test_errors,'b',markersize = 5)
plt.xlabel('1/k')
plt.ylabel('red = train errors, blue = test errors')
plt.show()








