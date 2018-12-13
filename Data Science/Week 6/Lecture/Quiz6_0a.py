#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 19:24:49 2018

@author: andrewcoogan
"""
import numpy as np

X = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
X_fs = (X - np.min(X))/(np.max(X) - np.min(X))

print(np.mean(X_fs))

X = np.array([0, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 1])
X_fs = (X - np.min(X))/(np.max(X) - np.min(X))

print(X_fs[4])
