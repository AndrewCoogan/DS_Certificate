#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 20:45:11 2018

@author: andrewcoogan
"""

import pandas as pd
import numpy as np

def kmeans(X, k, th):
    if k < 2:
        print('k needs to be at least 2!')
        return
    if (th <= 0.0) or (th >= 1.0):
        print('th values are beyond meaningful bounds')
        return    
    
    N, m = X.shape # dimensions of the dataset
    Y = np.zeros(N, dtype=int) # cluster labels
    C = np.random.uniform(0, 1, [k,m]) # centroids
    d = th + 1.0
    dist_to_centroid = np.zeros(k) # centroid distances
    
    while d > th:
        C_ = deepcopy(C)
        
        for i in xrange(N): # assign cluster labels to all data points            
            for j in xrange(k): 
                dist_to_centroid[j] = np.sqrt(sum((X[i,] - C[j,])**2))                
            Y[i] = np.argmin(dist_to_centroid) # assign to most similar cluster            
            
        for j in xrange(k): # recalculate all the centroids
            ind = FindAll(Y, j) # indexes of data points in cluslter j
            n = len(ind)            
            if n > 0: C[j] = sum(X[ind,]) / n
        
        d = np.mean(abs(C - C_)) # how much have the centroids shifted on average?
        
    return Y, C

X = pd.DataFrame()
X.loc[:,0] = [1.91,0.9,1.26,0.61,1.25,1.04,0.53,0.99,1.11,0.1,-0.15,0.83,0.72,0.69,0.74,
      0.72,1.09,0.68,0.67,0.82,0.74,0.94,0.64,1.44,0.76,1.06,0.79,0.88,0.76,0.85,
      0.88,0.75,0.83,0.85,0.35,0.63,-0.14,-0.04,0.3,-0.52,-0.27,-0.32,-0.08,-0.39,
      -0.06,0.09,-0.51,-0.22,-0.03,-0.12,0.01,-0.21,-0.21,0.37,1.18,0,0,-0.66,-0.1,
      1.01,1.19,-0.3,-2.2,-1.82,-1.33,-0.84,-2.17,-1.67,-1.38,-1.39,-1.32,-1.49,
      -2.16,-1.64,-1.44,-1.58,-1.53,-1.53,-0.27,-1.32,-0.89,-0.33,-1.29]
X.loc[:,1] = [1.43,0.79,0.52,1.55,0.66,0.62,1.33,1.27,1.04,2.41,1.83,1.02,1.17,0.97,0.91,
      0.14,0.53,1.15,0.96,0.87,0.27,-0.15,0.82,0.72,0.84,1.52,0.93,0.91,0.87,0.93,
      0.97,1,0.86,0.88,0.55,-1.99,-0.78,-0.32,0.67,-1.75,-0.7,-0.51,-0.37,-0.55,
      -0.42,-0.48,0.64,-0.49,-0.51,-0.32,-0.48,-0.57,-0.32,-0.28,-1.51,-0.41,-0.44,
      -2.27,-0.67,-0.32,0.43,-1.26,-1.85,-0.16,-0.89,0.05,-0.38,-0.53,-1.75,-0.98,
      -0.33,-1.41,-1.33,-0.9,-0.72,-0.77,-0.66,-0.81,-0.87,-0.94,-1.73,0.55,-0.7]

# Create initial cluster centroids
k = pd.DataFrame()
k.loc[:,0] = [-1, 1, 0]
k.loc[:,1] = [2, -2, 0]