#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 14:02:12 2018

@author: andrewcoogan
"""

#  Import statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
# / Import statements

#  Load the data from the machine larning database
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv'
data = pd.read_csv(url, header = None)  #  There is no header, we will define
data.columns = ['Age', 'Gender', 'TB', 'DB', 'AAP', 'SgptAA', 'SgotAA', 'TP', 'ALP', 'AGRatio', 'Selector']
# data.head()

tempAGRatio = data.AGRatio.dropna()  #AGRatio has an na in it, for the hist, lets just remove it

# The next blob is going to show plots, we could have done a for loop but I wanted to be able to 
# move through this section slowly so I could look at the data one at a time.
plt.show(plt.hist(data.Age))
plt.show(plt.hist(data.Gender))
plt.show(plt.hist(data.TB))
plt.show(plt.hist(data.TP))
plt.show(plt.hist(data.AAP)) ### This would be a good candidate for winsorization
plt.show(plt.hist(data.SgptAA)) ### This would be a good candidate for winsorization
plt.show(plt.hist(data.SgotAA)) ### This would be a good candidate for winsorization
plt.show(plt.hist(data.TP))
plt.show(plt.hist(data.ALP))
plt.show(plt.hist(tempAGRatio))
plt.show(plt.hist(data.Selector))

# Change the nulls in the AGRatio to the median of the group
data.loc[data.AGRatio.isna(),'AGRatio'] = np.nanmedian(data.AGRatio)

# Show plot, there are some warnings but they do not seem to effect the output
plt.show(scatter_matrix(data))

# This loop will go through all columns and output the standard deviation of each output
for i in data.columns:
    if i == 'Gender': continue # We are skipping 'Gender' because its a binary output and does not provide insight
    print(i + ' standard deviation is ' + str(np.std(data.loc[:, i].astype(int))))
    
upper_limit = 2 * np.std(data.TB)  #  Define upplimit as 2 std's from mean
lower_limit = -2 * np.std(data.TB)  #  Define lower limit as 2 std's from mean
outlier = data.TB < upper_limit # only need upper limit as all values are positive
data.loc[outlier, 'TB'] = np.median(data.TB[~outlier]) #  This replaces the outliers with the median of the nonoutliers

# Print the newly modified TB data with outliers replaced with median.
print('TB standard deviation of the TB data is: ' + str(np.std(data.TB)))

