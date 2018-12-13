#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 14:52:43 2018

@author: andrewcoogan
"""

# Import Statements
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#  Read in the data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
data = pd.read_csv(url, delim_whitespace=True, header=None)
data.columns = ["MPG", "Cylinders", "Displacement", "Horsepower", "Weight", 
               "Acceleration", "Model_Year", "Origin", "Car_Name"]

print(data.head())

#  Replace the missing values, designated '?', and replace it with np.nan's.
data = data.replace(to_replace="?", value=np.nan)
print(data.apply(lambda x: x.isnull().sum(), axis=0))


### There are missing horsepower values, I am going to remove.
#  It is a small portion of the total number of rows so it will not effect the data.
print(data.loc[data.Horsepower.isnull(),])
data = data.loc[~data.Horsepower.isnull(),]
print(data.apply(lambda x: x.isnull().sum(), axis=0))

# Origin is a multivalued categorical column
data.Origin.value_counts()

# Looking at the source data 1 is American, 2 is European, 3 is Asian
# We create a new column for the origins and set it to a bool if the appropriate 
#      column header matches the designated origin.  This is the dummy variable 
#      creation process.
data.loc[:, "American"] = (data.loc[:, "Origin"] == 1).astype(int)
data.loc[:, "European"] = (data.loc[:, "Origin"] == 2).astype(int)
data.loc[:, "Asian"] = (data.loc[:, "Origin"] == 3).astype(int)

print(pd.DataFrame(data.loc[:,["American", "European", "Asian"]]).sum(axis = 0))
#  This agrees with the above output of value_counts, so I am confident this is correctly done
#  Since we now have redundant data, we can remove the original "Origin" column

data = data.drop("Origin", axis = 1)
#  One thing to look into is pd.get_dummies, should do the same but need to explore.

print(data.head())

#  Pull out the individual columns for formatting and not mess with original dataframe
horsepower = data.Horsepower.apply(float)
acceleration = data.Acceleration.apply(float)
fit = np.poly1d(np.polyfit(horsepower, acceleration, 2))
#  I am going to fit using a polynomial with degree 2 (ie ax^2+bx+c)
sct = plt.scatter(horsepower, acceleration)
plt.plot(np.unique(horsepower), fit(np.unique(horsepower)))
plt.title("Horsepower vs Acceleration")
plt.xlabel("Horsepower")
plt.ylabel("Acceleration")
plt.show()
print("With the following parameters:")
print(fit)

#  This is going to look at the seaborn package
#  It looks generally the same but it has confidence bands, which is neat
#  Downside of this method is that there is no way of pulling parameters, so future 
#       fittings are not possible.
sns.set(color_codes = True)
sns.regplot(horsepower,acceleration, color="G", order = 2)