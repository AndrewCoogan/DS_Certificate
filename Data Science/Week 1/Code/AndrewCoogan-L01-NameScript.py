#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 22:37:37 2018

@author: andrewcoogan
"""

# Function: whoAmI
# Returns: Who I am, Andrew Coogan
# Parameters: None 
def whoAmI():
    return("Andrew Coogan")

whoAmI()

# https://www.saltycrane.com/blog/2008/06/how-to-get-current-date-and-time-in/
# Function: whoAmI
# Returns: datetime.now() at time of run, can be used for other needs such as 
#       run time or the like.
# This function also prints out the current date.
# Parameters: None 
import datetime as dt
def returnDate():
    curr = dt.datetime.now()
    print('Current date is: ' + curr.strftime("%Y-%m-%d"))
    return(None)  # Not sure if this is syalistically the best

returnDate()