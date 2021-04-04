# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 05:14:32 2020

@author: DHRUV
"""


"""
HW2
g

"""

"""
Assumption
We have to run a full linear regression model with ALL pairwise interaction terms (in HW2 PDF)
Thus we'll run a single model with all the terms present, and all the pairwise interactions also
present and test their relevance/significance using p/t tests. So only a single model will be run

"""

"""
Model Type
There will be:
AT    V    AP    RH        AT*V          AT*AP       AT*RH        V*AP       V*RH        AP*RH

So 4 single terms and 6 interaction terms for a total of 10 terms
Thus we have 10 predictors in this model predicting PE

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
poly = PolynomialFeatures(degree=2, interaction_only = True)
df_features = df.drop(columns = ['PE'])
print(df_features)
df_modified_features = pd.DataFrame(poly.fit_transform(df_features))
print(df_modified_features)
final_features = df_modified_features.drop(columns = 0)
print(final_features)
final_features.columns = ['AT','V','AP','RH','AT*V','AT*AP','AT*RH','V*AP','V*RH','AP*RH']
print(final_features)
y = df['PE']
print(y)
reg = LinearRegression()
reg.fit(final_features,y)
y_predicted = reg.predict(final_features)
print('The y predicted using all the interaction terms is as follows : \n', pd.DataFrame(y_predicted, columns=['y_predicted']))
print('The mean squared error obtained for this interaction based model is : \n', mean_squared_error(y,y_predicted))
print('The value of intercept term for this model is : \n', reg.intercept_)
print('The value of the feature coefficients in array form for all the terms, including interaction terms is, in their respective order : \n', reg.coef_)

"""
The MSE for the multi-regression model was around 20
The MSE for this interaction based multi-regression model is 18
Thus there is a minor improvement in the prediction of the model in terms of the MSE error

Again, test data needs to be considered for a much better interpretation and for complete analysis

"""
final_features_sm = sm.add_constant(final_features)      
est = sm.OLS(y, final_features_sm)
est_data = est.fit()
print(est_data.summary())
t_reference = t.ppf(1-0.025, 9557)
print('The reference value of t-statistic is : \n',t_reference)
print('The reference value for F-statistic (10,9557,0.05) is, from an online calculator : \n 1.83169')


"""
Analyzing the p values with the p value boundary being 0.05 (95% confidence interval)

AT : We can't reject the null hypothesis for AT. AT not significant. Due to hiearchical principal however
we still can't remove it from the model.

V : Significant. Present in a significant interaction.

AP : Barely, just barely significant. At the boundary seperating significant predictor, from the 
insignificant predictor. But taking p = 0.05, 0.047 < 0.05 and thus we declare it significant and
reject the null hypothesis for AP. Again it is present in an interaction that is significant.

RH : Again barely significant. As 0.042 < 0.05, we declare it significant and reject the null. Also
present in an interaction that's significant.

AT*V : Significant

AT*AP : Null hypothesis cannot be rejected. Insignificant.

AT*RH : Significant

V*AP : Significant

V*RH : Insignificant. Null can't be rejected.

AP*RH : Significant. 0.034 < 0.05 so null can be rejected.


F-statistic implies total null hypothesis can be rejected, making at least one predictor relevant




Summarizing :

Individually in our interaction based model, 
V, AP, RH significant  

Interaction terms,
AT*V , AT*RH, V*AP, AP*RH are significant

Thus interactions between AT and V, AT and RH, V and AP, AP and RH must be considered in the model

AT is individually insignificant
AT*AP , V*RH interaction terms are insignificant and can be removed from the model


HIEARCHICAL PRINCIPLE states that despite large p-values for AT, it should be included in the
model. The reason being AT*V and V both are included. Thus given that AT is included in an interaction, and 
that interaction has a low p-value, or the interaction is relevant, the terms individually included
in the interaction must be included in the model (even if they have high p-values).

THUS,
AT cannot be removed from the model despite it's p-value being large. So even though AT is statistically
insignificant, it's presence in interaction terms that are significant implies via the hiearchical principle
that it's presence is crucial to "correctly interpret" the model.

THUS all the interaction based terms with large p can be dropped, however the main terms, the terms
that are originally present, can't be dropped even if their p values are high, if those terms are
present in interactions that are relevant.

Similar discussions for other terms can be done

We do need test data however to validate whether or not including all the relevant interaction terms
makes the model better or not. As such, the MSE difference between the normal multi-regression model
and the interaction based multi-regression model, does not truly warrant the inclusion of 6 more 
predictors other than only to get a more detailed look at the model. In terms of prediction, at least 
according to training data, normal multi-regression model is pretty accurate.

With available test data and prediction on it however, and the errors we obtain on its predictions, we can 
decide better which model of the two is more suitable.

"""





