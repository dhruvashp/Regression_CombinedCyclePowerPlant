# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 23:51:54 2020

@author: DHRUV
"""


"""
HW2

e
"""

"""
Univariate vs Multivariate Comparisons

A .csv file was made tabulating the univariate and the multivariate beta coefficients
for all the independent variables

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

table = pd.read_csv('HW2_e_Beta_Comparison_Table.csv', index_col = 0)
print(table)
df = pd.read_csv('Power_Plant.csv')
print(df)
features = df.drop(columns=['PE'])
correlation_features = features.corr()
print('The correlation matrix for our features is : \n')
print(correlation_features)
plt.plot([-2.1713],[-1.9775], 'ro')
plt.plot([-1.1681],[-0.2339], 'bo')
plt.plot([1.4898],[0.0621], 'go')
plt.plot([0.4556],[-0.1581], 'yo')
plt.xlabel('Univariate')
plt.ylabel('Multivariate')
red_patch = mpatches.Patch(color='red', label='AT')
blue_patch = mpatches.Patch(color='blue', label='V')
green_patch = mpatches.Patch(color='green', label='AP')
yellow_patch = mpatches.Patch(color='yellow', label='RH')
plt.legend(handles=[red_patch,blue_patch,green_patch,yellow_patch])
plt.show()


"""
From the values of the different predictors:

AT's value in multivariate reduced in terms of magnitude
V's value became very close to 0 (showing the relationship between AT and V)
AP's value reduced almost to 0 from a relatively large univariate value
RH's value in multivariate became negative

As can be seen the correlation between AT and V is pretty high, which gives way to an explanation
as to why V's coefficient in the multivariate regression was reduced. Had it not been reduced
there would be 'double counting'. When AT increases, V also increases. Thus if the coefficient
in the multi-regression model for V was same as that for V in the uni-regression model then the increase
in PE would be overestimated as in V's individual model all the other predictors were ignored.

The increase in V also causes an increase in AT, and AT increases PE in the multiple-regression model.
Thus to ensure the model accuracy, V's coefficient must reduce so that increase in PE is not counted
"twice", both via an increase in AT and an increase in V. Increase in either two implies an increase
in the other, to ensure that this increase in the other doesn't cause PE to increase way too much
in the multiple regression model, both V and AT's values of their coefficients go down.

Similar explanations ensue for AT and AP, AT and RH, V and AP, V and RH and also AP and RH

Using their correlation and using the univariate values we can validate the changes in the coefficients
from the univariate to the multivariate model, thus improving our "collective" understanding of the
model

"""



