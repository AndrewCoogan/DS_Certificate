#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 21:52:54 2018

@author: andrewcoogan
"""

import pandas as pd
import numpy as np
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt

DeviceTypes = [
"Cell Phone", "Dish Washer", "Laptop", "Phone", "Refrigerator", "Server",
"Oven", "Computer", "Drill", "Server", "Saw", "Computer", "Nail Gun",
"Screw Driver", "Drill", "Saw", "Saw", "Laptop", "Oven", "Dish Washer",
"Oven", "Server", "Mobile Phone", "Cell Phone", "Server", "Phone"]
Devices = pd.DataFrame(DeviceTypes, columns=["Names"])

Devices


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
Auto = pd.read_csv(url, delim_whitespace=True, header=None)
Auto.columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "car_name"]

plt.hist(Auto.loc[:, "weight"])  # Best way to present the weight column

DeviceTypes = [
"Cell Phone", "Dish Washer", "Laptop", "Phone", "Refrigerator", "Server",
"Oven", "Computer", "Drill", "Server", "Saw", "Computer", "Nail Gun",
"Screw Driver", "Drill", "Saw", "Saw", "Laptop", "Oven", "Dish Washer",
"Oven", "Server", "Mobile Phone", "Cell Phone", "Server", "Phone"]
Devices = pd.DataFrame(DeviceTypes, columns=["Names"])

Devices.loc[Devices.loc[:, "Names"] == "Cell Phone", "Consolidated_name"] = "Phone"
Devices.loc[Devices.loc[:, "Names"] == "Phone", "Consolidated_name"] = "Phone"
Devices.loc[Devices.loc[:, "Names"] == "Mobile Phone", "Consolidated_name"] = "Phone"
Devices.loc[Devices.loc[:, "Names"] == "Dish Washer", "Consolidated_name"] = "KitchenAppliance"
Devices.loc[Devices.loc[:, "Names"] == "Refrigerator", "Consolidated_name"] = "KitchenAppliance"
Devices.loc[Devices.loc[:, "Names"] == "Oven", "Consolidated_name"] = "KitchenAppliance"
Devices.loc[Devices.loc[:, "Names"] == "Drill", "Consolidated_name"] = "Tools"
Devices.loc[Devices.loc[:, "Names"] == "Saw", "Consolidated_name"] = "Tools"
Devices.loc[Devices.loc[:, "Names"] == "Nail Gun", "Consolidated_name"] = "Tools"
Devices.loc[Devices.loc[:, "Names"] == "Screw Driver", "Consolidated_name"] = "Tools"
Devices.loc[Devices.loc[:, "Names"] == "Drill", "Consolidated_name"] = "Tools"
Devices.loc[Devices.loc[:, "Names"] == "Server", "Consolidated_name"] = "Work"
Devices.loc[Devices.loc[:, "Names"] == "Computer", "Consolidated_name"] = "Work"
Devices.loc[Devices.loc[:, "Names"] == "Server", "Consolidated_name"] = "Work"
Devices.loc[Devices.loc[:, "Names"] == "Laptop", "Consolidated_name"] = "Work"


import numpy as np
x = np.array(["WA", "Washington", "Wash", "UT", "Utah", "Utah", "UT", "Utah", "IO"])
x[x == "Washington"] = "WA"
x[x == "Wash"] = "WA"
x[x == "Utah"] = "UT"


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
Auto = pd.read_csv(url, delim_whitespace=True, header=None)
Auto.columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", 
               "acceleration", "model_year", "origin", "car_name"]


AB = ["B", "A", "B", "B", "B", "A"]
Test = pd.DataFrame(AB, columns=["AB"])
Test.loc[:, "isA"] = (Test.loc[:, "AB"] == "A").astype(int)






