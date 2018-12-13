#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 20:39:31 2018

@author: andrewcoogan
"""

from sklearn.preprocessing import *
import pandas as pd
import numpy as np

x = np.array([81, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 
              6, 6, 7, 7, 7, 7, 8, 8, 9, 12, 24, 24, 25])

### For equal frequency ###
bins_eqf = 3
BinCount=len(x)/bins_eqf
print("Each Bin contains",BinCount,"elements.")
###

### For equal width ###
bins = 4
bounds = np.linspace(np.min(x), np.max(x),  bins + 1) # more straight-forward way for obtaining the boundaries of the bins

def bin(x, b): # x = x b = bounds
    nb = len(b)
    N = len(x)
    y = np.empty(N, int) # empty integer array to store the bin numbers (output)
    
    for i in range(1, nb): # repeat for each pair of bin boundaries
        y[(x >= bounds[i-1])&(x < bounds[i])] = i
    
    y[x == bounds[-1]] = nb - 1 # ensure that the borderline cases are also binned appropriately
    return y

print(bin(np.sort(x), bounds))
###

x = np.array([81, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 
              6, 6, 7, 7, 7, 7, 8, 8, 9, 12, 24, 24, 25])
MaxBin1 = 29
MaxBin2 = 55
labeled = np.empty(28, dtype=str)     
labeled[(x > -float("inf")) & (x <= MaxBin1)] = "1"
labeled[(x > MaxBin1)       & (x <= MaxBin2)] = "2"
labeled[(x > MaxBin2)       & (x <= float("inf"))] = "3"


x = np.array([81, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 
              6, 6, 7, 7, 7, 7, 8, 8, 9, 12, 24, 24, 25])
MaxBin1 = 5.5
MaxBin2 = 7.5
labeled = np.empty(28, dtype=str)     
labeled[(x > -float("inf")) & (x <= MaxBin1)]      = "1"
labeled[(x > MaxBin1)       & (x <= MaxBin2)]      = "2"
labeled[(x > MaxBin2)       & (x <= float("inf"))] = "3"



import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

import pandas as pd

x = np.array([1., -1, -1, 1, 1, 17, -3, 1, 1, 3])
zscaled = (x - np.mean(x))/np.std(x)
minmaxscaled =(x - min(x))/(max(x) - min(x))

standardization_scale = StandardScaler().fit(pd.DataFrame(x)).transform(pd.DataFrame(x))

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data"
Mamm = pd.read_csv(url, header=None)
Mamm.columns = ["BI-RADS", "Age", "Shape", "Margin", "Density", "Severity"] 

mamm_bir = Mamm["BI-RADS"]
mamm_bir = mamm_bir.replace(to_replace="?", value=np.nan).apply(pd.to_numeric, errors = "coerse")
zscaled_bie = (mamm_bir - np.mean(mamm_bir))/np.std(mamm_bir)
