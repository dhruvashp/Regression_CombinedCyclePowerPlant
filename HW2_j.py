# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 18:42:37 2020

@author: DHRUV
"""


"""
HW2
j
"""

"""
Comparing KNN regressions smallest test error with Linear Regressions smallest test error

"""

"""

Smallest KNN Regression Test Error = 16.7062    k = 6    Data = Raw

Smallest Linear Regression Test Error = (17.5 to 18.5)   Data = Raw  (part h)

Normal Multi-regression Train Error = 20                 Data = Raw  (part d)


The range has been provided for Linear Regression test error as each program run time
generated random test and training sets using train_test_split, but for KNN regression
test and train sets were fixed

Comments

Of all the linear regression models, the models used in (h) (with all interactions and 
quadratics, and then removing the insignificant terms from that model) had the least
test error. Again for all models used prior to (h) no test data was there and MSE were obtained
over the training sets itself. Still, even on those training sets the MSE that was least was obtained
for the normal multi-regression model (MSE = 20, normal multi-regression model).

Thus models in (h) were best for linear regression MSE wise (test and train both)

The KNN model gives us, with raw data, an MSE test error minorly smaller than that
obtained via linear regression in (h)

Thus we could choose KNN or even the Linear Regression model of (h) or just the 
normal multi-regression model. All of them have MSE above 15 but below 20 and thus
all the methods are quite close in terms of accuracy (predicted via MSE)


"""