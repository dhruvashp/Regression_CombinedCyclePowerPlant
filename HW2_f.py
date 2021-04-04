# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 01:12:43 2020

@author: DHRUV
"""


"""
HW2
f
"""

"""
FOR AT
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

df = pd.read_csv('Power_Plant.csv')
print(df)
X_AT = df.drop(columns = ['V','AP','RH','PE'])
y = df['PE']
print(y)
print(X_AT)
poly = PolynomialFeatures(3)
X_AT_modified=poly.fit_transform(X_AT)
print(X_AT_modified)
X_AT_new = pd.DataFrame(data = X_AT_modified)
print(X_AT_new)
X_AT_new = X_AT_new.iloc[:,1:4]
print(X_AT_new)
header_AT = ['AT','AT2','AT3']
X_AT_new.columns = header_AT
print(X_AT_new)
reg_AT = LinearRegression()
reg_AT.fit(X_AT_new, y)
y_AT_predicted = reg_AT.predict(X_AT_new)
print('The predicted vector y of output from this new model of AT is : \n', pd.DataFrame(y_AT_predicted, columns = ['y_predicted']))
print('The mean squared error (MSE) of y for AT under this new polynomial feature model is : \n', mean_squared_error(y, y_AT_predicted))
print('The intercept term for AT is : \n', reg_AT.intercept_)
print('The coefficients of AT, AT2, AT3, respectively in an array is : \n', reg_AT.coef_)
"""
The mean squared error of y in only AT case was around 29, thus there does seem to be a slight imporvement
in the error in y under a polynomial fit with degree of AT going to 3. 

However as a test data has not been analyzed for either cases, we can't say for sure how much
of a truly tangible improvement this is for data that the trained model hasn't seen

"""

"""
Analyzing relevance of each variable (AT,AT2,AT3) using t-test, using statsmodels library 
(using manual evaluation has been done before, and will just be code-elongating here)

"""

X_AT_new_sm = sm.add_constant(X_AT_new)      
est_AT = sm.OLS(y, X_AT_new_sm)
est_data_AT = est_AT.fit()
print(est_data_AT.summary())

t_reference = t.ppf(1-0.025, 9564)
print('The reference value of t-statistic is : \n', t_reference)
print('The reference value of F-statistic (3,9564,0.05) is, from an online calculator : \n', 2.6058)

"""
As can be seen from both the t-test and the p-test, all the AT terms (AT,AT2,AT3)
are relevant

The F-observed is much larger than the F-reference (2.6058) and thus we can reject the 
hypothesis : "All beta's are null". Thus at least one of the predictor is relevant.

"""

"""
FOR V
"""


X_V = df.drop(columns = ['AT','AP','RH','PE'])
print(X_V)
poly = PolynomialFeatures(3)
X_V_modified=poly.fit_transform(X_V)
print(X_V_modified)
X_V_new = pd.DataFrame(data = X_V_modified)
print(X_V_new)
X_V_new = X_V_new.iloc[:,1:4]
print(X_V_new)
header_V = ['V','V2','V3']
X_V_new.columns = header_V
print(X_V_new)
reg_V = LinearRegression()
reg_V.fit(X_V_new, y)
y_V_predicted = reg_V.predict(X_V_new)
print('The predicted vector y of output from this new model of V is : \n', pd.DataFrame(y_V_predicted, columns = ['y_predicted']))
print('The mean squared error (MSE) of y for V under this new polynomial feature model is : \n', mean_squared_error(y, y_V_predicted))
print('The intercept term for V is : \n', reg_V.intercept_)
print('The coefficients of V, V2, V3, respectively in an array is : \n', reg_V.coef_)

"""
The mean squared error of y in only V  case was around 70, thus there is an extremely minor 
improvement in the error in y under a polynomial fit with degree of V going to 3. 

Again neither model saw the test data, and thus performance under test data wasn't analyzed

Again despite the addition of two more "variables" the improvement was very minor as compared to
the only V case.

"""

"""
Analyzing relevance of each variable (V,V2,V3) using t-test, using statsmodels library 
(using manual evaluation has been done before, and will just be code-elongating here)

"""

X_V_new_sm = sm.add_constant(X_V_new)      
est_V = sm.OLS(y, X_V_new_sm)
est_data_V = est_V.fit()
print(est_data_V.summary())
print('The reference value of t-statistic is : \n', t_reference)
print('The reference value of F-statistic (3,9564,0.05) is, from an online calculator : \n', 2.6058)

"""
As can be seen from both the t-test and the p-test, For V

V : relevant
V2 : has p > 0.05 and also t-value not within appropriate bounds

Thus V2 is surely irrelevant and the null hypothesis for V2 cannot be rejected. 
V2 is thus irrelevant

V3 : has p < 0.05 (0.014 < 0.05)
Thus V3 is statistically relevant, note that the V3 coefficient is extremely small,
however with V3 cubed, the relevance of V3 as a possible variable can't be rejected.
We can thus, for V3, reject the null hypothesis


The F-observed is much larger than the F-reference (2.6058) and thus we can reject the 
hypothesis : "All beta's are null". Thus at least one of the predictor is relevant.

So for V, V and V3 are legitimate predictors. V2 can be removed

"""

"""
FOR AP
"""
"""
NOTE : The results from sklearn and statsmodels in terms of co-efficients were different. This
probably occured due to large values of AP2 and AP3. 

Here, despite difference in coefficients, both models predicted vector y with a similar accuracy.

Thus while both models, despite the difference in their coefficients were correct, we used
statsmodels due to its inbuilt functionality to calculate t-values. Thus the co-efficients using
scikit/sklearn are different than the ones obtained using statsmodels here, however the predictions
from both are similarly accurate. Thus both sklearn and statsmodels estimate coefficients differently
when the values of predictors increase to a large value, however both are equally accurate in making
the final prediction

"""

X_AP = df.drop(columns = ['AT','V','RH','PE'])
print(X_AP)
poly = PolynomialFeatures(3)
X_AP_modified=poly.fit_transform(X_AP)
print(X_AP_modified)
X_AP_new = pd.DataFrame(data = X_AP_modified)
print(X_AP_new)
X_AP_new = X_AP_new.iloc[:,1:4]
print(X_AP_new)
header_AP = ['AP','AP2','AP3']
X_AP_new.columns = header_AP
print(X_AP_new)
X_AP_new_sm = sm.add_constant(X_AP_new)      
est_AP = sm.OLS(y, X_AP_new_sm)
est_data_AP = est_AP.fit()
X_AP_new_sm_array = X_AP_new_sm.to_numpy()
print(X_AP_new_sm_array)
y_AP_predicted = est_data_AP.predict(X_AP_new_sm_array)
print('The predicted vector y of output from this new model of AP is : \n', pd.DataFrame(y_AP_predicted, columns = ['y_predicted']))
print('The mean squared error (MSE) of y for AP under this new polynomial feature model is : \n', mse(y,y_AP_predicted))
print('The intercept term for AP, along with coefficients of AP, AP2, AP3, respectively in an array is : \n', est_data_AP.params)

"""
The MSE for AP was 212 in the only AP case while here it is 211. Thus the improvement is negligible

Thus the addition of AP2 and AP3 does little to increase the models accuracy

Again no analysis done on test, unseen data

"""

"""
Analyzing relevance of each variable (AP,AP2,AP3) using t-test, using statsmodels library 
(using manual evaluation has been done before, and will just be code-elongating here)

"""
print(est_data_AP.summary())
print('The reference value of t-statistic is : \n', t_reference)
print('The reference value of F-statistic (3,9564,0.05) is, from an online calculator : \n', 2.6058)

"""
As can be seen from both the t-test and the p-test, For AP

For this model using AP, AP2, AP3

We have

p values for AP, AP2 and AP3 rather small implying that in this model, with the given coefficients
all the three variables are relevant and the null hypothesis for each can be rejected.

The F-statistic also says that total null hypothesis can be rejected making at least one of the 
predictors relevant in making the prediction.

Again we will note that the MSE hasn't been significantly improved, thus despite the
model relevance of AP2 and AP3, there is no real need to add them. Once they have been added
however they do become significant in making the predictions.

"""

"""
FOR RH
"""
X_RH = df.drop(columns = ['AT','AP','V','PE'])
print(X_RH)
poly = PolynomialFeatures(3)
X_RH_modified=poly.fit_transform(X_RH)
print(X_RH_modified)
X_RH_new = pd.DataFrame(data = X_RH_modified)
print(X_RH_new)
X_RH_new = X_RH_new.iloc[:,1:4]
print(X_RH_new)
header_RH = ['RH','RH2','RH3']
X_RH_new.columns = header_RH
print(X_RH_new)
reg_RH = LinearRegression()
reg_RH.fit(X_RH_new, y)
y_RH_predicted = reg_RH.predict(X_RH_new)
print('The predicted vector y of output from this new model of RH is : \n', pd.DataFrame(y_RH_predicted, columns = ['y_predicted']))
print('The mean squared error (MSE) of y for RH under this new polynomial feature model is : \n', mean_squared_error(y, y_RH_predicted))
print('The intercept term for RH is : \n', reg_RH.intercept_)
print('The coefficients of RH, RH2, RH3, respectively in an array is : \n', reg_RH.coef_)

"""
The MSE for the only RH case was 246. Here also it is around 246. Next to no improvement
in errors here. 

"""

"""
Analyzing relevance of each variable (RH,RH2,RH3) using t-test, using statsmodels library 
(using manual evaluation has been done before, and will just be code-elongating here)

"""

X_RH_new_sm = sm.add_constant(X_RH_new)      
est_RH = sm.OLS(y, X_RH_new_sm)
est_data_RH = est_RH.fit()
print(est_data_RH.summary())
print('The reference value of t-statistic is : \n', t_reference)
print('The reference value of F-statistic (3,9564,0.05) is, from an online calculator : \n', 2.6058)

"""
As can be seen from both the t-test and the p-test

RH, RH2 and RH3 are relevant and significant and thus can't be neglected. The null hypothesis
for these variables can be rejected.

Additionally the F-statistic says that we can reject the total null hypothesis, making at least
one predictor of the three relevant

"""


"""
Final Note

As such there wasn't a drastic improvement in the model by adding the squared and cubed terms.
Once they were added however, in most cases, they became statistically relevant to the model.

Thus we can surely say that the "linear only" model, without any higher terms of any of the 
predictors may be preferred as addition of squared and cubed terms doesn't improve significantly
the prediction accuracy.

Again we cannot talk about test data, and implications from testing our models on the 
test data

"""





