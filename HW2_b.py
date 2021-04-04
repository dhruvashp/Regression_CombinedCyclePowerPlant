# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 03:05:45 2020

@author: DHRUV
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 02:41:43 2020

@author: DHRUV
"""


"""
HW2
b (i)

There are 9569 total rows in this dataset.
The first row actually contains the column header, so effective 9568 total data rows exist.
There are total 5 columns in the dataset.
The first four columns are the features (AT, V, AP, RH) and the last column has the
output or the dependent variable (PE)

As mentioned in the HW2 PDF, the feature columns are the various power plant feature 
measurements, the various rows contain these measurements, and the output column contains
the power output of the power plant (net hourly) which is the variable to be predicted based 
on our predictor columns (various measurements within the plant)

"""

"""
b (ii)

Here we'll make a scatterplot matrix, with variables AT, V, AP, RH, PE as our columns and also as our
rows with plots performed for each variable on the x-axis with another on the y-axis

A total of, thus, 25 plots, will be drawn for each variable against the other (both dependent and 
independent)

"""

import pandas as pd
import seaborn as sns
import numpy as np

df = pd.read_csv('Power_Plant.csv')
print(df)
sns.pairplot(df)

"""
The diagonal plots (of a variable against itself) show frequency of repetitions of a particular value
the variable takes on the x-axis with the frequency on the y-axis
"""

"""
Plot Findings
Focusing on the last four curves, drawn with PE on the y-axis with the feature on the x-axis
we see that

PE vs AT is quite linear in nature
PE vs V is also quite linear, but with a much larger spread
PE vs AP might follow a "mean linear relationship" however the spread is too much for a prediction of 
PE to be made just via a value of AP assuming linearity between them
PE vs RH is similar to PE vs AP, with a spread that's even larger

AT vs V have an approximate linear curve, meaning they may be correlated with each other
to a certain degree

Histogram of PE shows that PE = 440 seems to be most frequent

"""
"""
b (iii)

"""
data_summary = df.describe(include='all')
print(data_summary)

"""
25% = first quartile
50% = median
75% = third quartile
"""
data_summary.to_csv('Data_Summary_b_iii.csv')

"""
exporting to .csv file
"""
