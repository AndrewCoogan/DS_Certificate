#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 18:10:49 2018

@author: andrewcoogan
"""

import numpy as np

students = np.array([4,5," ",6,4,4,"4",-1,6,5,4,4,5,45,5,5,5,6,5,6,4,99,4,6,"?",6,6,"NA",4,5,4,6,6,6,6,4,5,4,6,4,6," ",4,5,5,5])
#  Above is the dataset I used from the canvas site.  I had to make a couple slight changes
#  I added quotes around the empty sets, ie, 5,,6 was not accepted and I had to do 5,"",6
#  NA was not accepted because there is no variable to NA, so I changed that to "NA"

def remove_outliers(arr):
    #  Check and make sure there are no non numerics in input
    if any([not str(n).isdigit() for n in arr]):
        #  If there are any non numerics just return None and a warning message
        print("Input contains non-numeric values.  Returning...")
        return None
    #  I defined that outliers are numbers that are two sigma from mean
    lH = float(np.mean(arr) + 2*np.std(arr))
    lL = float(np.mean(arr) - 2*np.std(arr))
    #  Return the input array excluding the number more than two sigma out
    return arr[~((arr < lL) | (arr > lH))]

def replace_outliers(arr):
    #  Check and make sure there are no non numerics in input
    if any([not str(n).isdigit() for n in arr]):
        #  If there are any non numerics just return None and a warning message
        print("Input containts non-numeric values.  Returning...")
        return None
    #  I defined that outliers are numbers that are two sigma from mean
    lH = float(np.mean(arr) + 2*np.std(arr))
    lL = float(np.mean(arr) - 2*np.std(arr))
    #  To_replace is a bool array that flags True when a value is an outlier
    to_replace = ((arr < lL) | (arr > lH))
    #  This replaces the outliers with the median of the non-outliers
    arr[to_replace] = np.median(arr[~to_replace])
    return arr

def replace_bad_values(arr):
    #  S is a boolean array that flags False when the element is not a numeric
    s = [str(e).strip().isdigit() for e in arr]
    #  Returns a array where all values are casted to int taking only True values from arr
    return np.array([int(o) for o in arr[s]])


### testing zone ###
#a = remove_outliers(students)  # Returns None as students contains bad values
#b = replace_outliers(students)  # Returns None as students contains bad values
#sanitized_students = replace_bad_values(students)  # Removed non numeric values and -1
#a_good = remove_outliers(sanitized_students)  # Removed two outliers (45, 99)
#b_good = replace_outliers(sanitized_students)  #  Replaced two outliers with median