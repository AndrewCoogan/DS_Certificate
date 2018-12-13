import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels import robust

#  Read and write out the data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
data = pd.read_csv(url, header=None)
#data = pd.read_csv("AdultCensusInfo.csv")
data.columns = ["Age", "Workclass", "FinlWeight", "Education", "EducationNum", "MaritalStatus", "Occupation",
                "Relationship", "Race", "Sex", "CapitolGain", "CapitolLoss", "HoursPerWeek", "NativeCountry", "Income"]
data.to_csv("AdultCensusInfo.csv", index = False)

#  First of all lets take a look at the structure and how unknowns are handled.
data.Occupation.value_counts()
#  They are '?', lets keep that in mind and remove them if we need in the future.

# Looks like the data has blank spaces around the question marks, we will need to apply strip when changing
data.loc[data.Workclass.str.strip() == "?",]

stringCols = ["Workclass", "Education", "MaritalStatus", "Occupation", "Relationship", "Race", "Sex", "NativeCountry"]
data[stringCols] = data[stringCols].apply(lambda x : x.str.strip(), axis = 0)
data = data.replace(to_replace="?", value=np.nan)
data.apply(lambda x: x.isnull().sum(), axis=0)

# The numeric columns we have are Age, FinlWeight, EducationNum, Capitol Gain/Loss, and HoursPerWeek
# Lets take a look at FinlWeight as it seems to be arbitrary from CPS data scources
fw = pd.Series(data.FinlWeight)
plt.title("Final Weight")
plt.hist(fw)
plt.show()

#  Since the tail is so long, I want to minimize the shift of data from the tails, so I am going to winsorize once.
mu_fw = np.mean(fw)
sigma_fw = np.std(fw)
limit = mu_fw + 2*sigma_fw
fw[fw > limit] = limit
plt.title("Winsorize Final Weight")
plt.hist(fw)
plt.show()

#  This actually turned out very nicely, I dont think we want to truncate it again, but lets see
pct_outlier_no_winz = len(data.FinlWeight.loc[data.FinlWeight > mu_fw + 2*sigma_fw])/len(data.FinlWeight)
new_mu = np.mean(fw)
new_sigma = np.std(fw)
pct_outier_wz = len(fw.loc[fw > new_mu + 2*new_sigma])/len(fw)
print("Before winzorization: " + "{0:.3f}%".format(pct_outlier_no_winz * 100))
print("After winzorization: " + "{0:.3f}%".format(pct_outier_wz * 100))
# In theory there are more outliers post winzorization, but I think the shape of the distribution is much more manageable.
# Perhaps this is a time we devate from using a normal distribution and look at one that can have fatter tails and a 
# left skew.

hpw = data.HoursPerWeek
hpw_normalized = (hpw - np.mean(hpw)) / np.std(hpw)
hpw_xscaled = (hpw - min(hpw)) / (max(hpw) - min(hpw))
hpw_mad = stats.norm.ppf(0.75)*(hpw - np.median(hpw)) / robust.mad(hpw)
data["HoursPerWeekNormalized"] = hpw_normalized
data["HoursPerWeekXScaled"] = hpw_xscaled
data["HoursPerWeekMAD"] = hpw_mad

# This is going to be a binnging exercise, I am going to do age
# For the record--I believe you are only as old as you act and feel, dont be offended by my catrgories :)
# Lets start with see how the age distribution looks
plt.show(plt.hist(data.Age))

data["AgeCat"] = pd.Series(np.nan * len(data.Age))
data.Age = data.Age.apply(float)
data.loc[data.loc[:, "Age"] <= 18, "AgeCat"] = "Child"
data.loc[(data.loc[:, "Age"] > 18) & (data.loc[:, "Age"] <= 35), "AgeCat"] = "Young Adult"
data.loc[(data.loc[:, "Age"] > 35) & (data.loc[:, "Age"] <= 45), "AgeCat"] = "Middle Aged Adult"
data.loc[(data.loc[:, "Age"] > 45) & (data.loc[:, "Age"] <= 70), "AgeCat"] = "Experienced Adult"
data.loc[(data.loc[:, "Age"] > 70), "AgeCat"] = "Elderly"
data.AgeCat.value_counts().plot(kind='bar')

#  I am going to one hot encode as an example the sex category as there are only two options in the set
#  The below is pretty straight forward and should be easy to expand upon.
data = data.join(pd.get_dummies(data.Sex))
data = data.drop("Sex", axis = 1)

# For consolodation we are going to look at the education
print(data.Education.value_counts())
data["Education"] = data.Education.apply(lambda l: l.strip())
data.loc[data.loc[:, "Education"] == "1st-4th", "Education"] = "Grade-School"
data.loc[data.loc[:, "Education"] == "5th-6th", "Education"] = "Grade-School"
data.loc[data.loc[:, "Education"] == "7th-8th", "Education"] = "Grade-School"
data.loc[data.loc[:, "Education"] == "HS-grad", "Education"] = "High-School"
data.loc[data.loc[:, "Education"] == "9th", "Education"] = "High-School"
data.loc[data.loc[:, "Education"] == "10th", "Education"] = "High-School"
data.loc[data.loc[:, "Education"] == "11th", "Education"] = "High-School"
data.loc[data.loc[:, "Education"] == "12th", "Education"] = "High-School"
data.loc[data.loc[:, "Education"] == "Assoc-voc", "Education"] = "Professional-School"
data.loc[data.loc[:, "Education"] == "Prof-school", "Education"] = "Professional-School"
data.loc[data.loc[:, "Education"] == "Some-college", "Education"] = "University"
data.loc[data.loc[:, "Education"] == "Assoc-acdm", "Education"] = "University"
data.loc[data.loc[:, "Education"] == "Some-college", "Education"] = "University"
data.loc[data.loc[:, "Education"] == "Bachelors", "Education"] = "University"
print(data.Education.value_counts())