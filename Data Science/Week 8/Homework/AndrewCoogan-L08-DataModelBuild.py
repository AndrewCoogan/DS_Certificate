
# coding: utf-8

# In[1]:


#  Import statements
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 


# In[2]:


#  Import data from the UCI database
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data'
data = pd.read_csv(url, header = None)

#  These were grabbed from the above website where they define the variables
data.columns = ['Class', 'Age', 'Sex', 'Steroid', 'Antiviral', 'Fatigue', 'Malaise', 'Anorexia', 'Liver Big', 
               'Liver Firm', 'Spleen Palpable', 'Spiders', 'Ascites', 'Varices', 'Bilirubin', 'Alk', 'Sgot',
               'Albumin', 'Protime', 'Histology']

#  This is just to help me seperate the column names
categorical_data = ['Steroid', 'Antiviral', 'Fatigue', 'Malaise', 'Anorexia', 'Liver Big', 
               'Liver Firm', 'Spleen Palpable', 'Spiders', 'Ascites', 'Varices', 'Histology']

non_categorical_data = ['Age', 'Bilirubin', 'Alk', 'Sgot', 'Albumin', 'Protime']

#  We need to replace all '?' with None and convert all numbers to the same type
data = data.replace(to_replace="?", value=np.nan).apply(pd.to_numeric, errors = "coerse")


# In[3]:


data.head()


# In[4]:


#  What is the percentage of values that are missing?
data.apply(lambda l: np.isnan(l).sum()/data.shape[0])


# In[5]:


# This is going to be the removal of na categorical data
# This is fine as its a small percentage of the total data, I would rather have integrity.
# If we end up with poor performace later, we can revisit.
data = data.loc[~np.isnan(data.Steroid),:]
data = data.loc[~np.isnan(data.Fatigue),:]
data = data.loc[~np.isnan(data.Spiders),:]
data = data.loc[~np.isnan(data["Liver Big"]),:]
data = data.loc[~np.isnan(data["Liver Firm"]),:]


# In[6]:


# This is going to be the replacement of nan's with the median
data.Bilirubin.fillna(value = np.nanmedian(data.Bilirubin), inplace=True)
data.Alk.fillna(value = np.nanmedian(data.Alk), inplace=True)
data.Sgot.fillna(value = np.nanmedian(data.Sgot), inplace=True)
data.Albumin.fillna(value = np.nanmedian(data.Albumin), inplace=True)


# In[7]:


# I need to remove protime as it has almost half missing values
data.drop(columns = "Protime", inplace = True)
#  We need to also remove it from our column list
non_categorical_data.remove("Protime")


# In[8]:


# Where are we at when it comes to missing values?
data.apply(lambda l: np.isnan(l).sum()/data.shape[0])
# Sweet!  We have all values accounted for


# In[9]:


#  Here we are going to look for two standard deivation outliers
for col in non_categorical_data:
    sigma = np.std(data[col].astype(float))
    mu = np.mean(data[col].astype(float))
    lower_bound = mu - (2 * sigma)
    upper_bound = mu + (2 * sigma)
    pct_outliers = ((data[col] > upper_bound) | (data[col] < lower_bound)).sum() / data.shape[0]
    print(col + ' is %1.3f%% outliers' % (100 * pct_outliers))
#  2 Standard deivation contains ~95% of the data.  The concentration of outliers is in line with what is expected.


# In[10]:


#  What does the Bilirubin distribution look like?
plt.show(plt.hist(data.Bilirubin))
#  I am not too sure how I feel about this distribution
#  It is a small fraction of the components that are considered outliers, but they are kinda far out
#  Lets run with this for the time being and we can come back and change it later if the tests go poorly


# In[11]:


#  I am going to normalize these.
#  Normalizing will take care of the outlies.
data[non_categorical_data] = data[non_categorical_data].apply(lambda l: (l - np.mean(l)) / np.std(l))


# In[12]:


#  What does the normalized distribution look like now?
plt.show(plt.hist(data.Bilirubin))


# In[13]:


#  Here we are going to split the data into two randomly generated groups: training and testing
r = 0.7  # 70% training : 30% testing
N = len(data) # total number of rows in the datset
n = int(round(N*r)) # number of elements in training sample
nt = N - n # number of elements in testing sample

# What rows do we want to each one?
training_idx = random.sample(range(N), n)
test_idx = [x for x in range(N) if x not in training_idx]


# In[14]:


#  I am going to have the target be the class.  This means that all the symptoms provided are going to be features to 
#    determine if the patient will live or die.
target = 'Class'

target_train = data[target].iloc[training_idx]
target_test = data[target].iloc[test_idx]
feature_train = data[data.columns.difference(list(target))].iloc[training_idx]
feature_test = data[data.columns.difference(list(target))].iloc[test_idx]


# In[15]:


#  Note -- after looking at the documentation, most of the default parameters are fine.
#  Logictic Regression
clf = LogisticRegression()
clf.fit(feature_train, target_train)
lr_pred = clf.predict(feature_test)
print('Logistic Regression accuracy is %1.3f%%' % (100*(lr_pred == target_test).sum()/len(feature_test)))


# In[16]:


#  Gaussian Naive Bays 
nbc = GaussianNB()
nbc.fit(feature_train, target_train)
nb_pred = nbc.predict(feature_test)
print('Gaussian Naive Bays accuracy is %1.3f%%' % (100*(nb_pred == target_test).sum()/len(feature_test)))


# In[17]:


#  K Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors = 2) # 2 because the results are either dead or alive
knn.fit(feature_train, target_train)
knn_pred = knn.predict(feature_test)
print('K Nearest Neighbors accuracy is %1.3f%%' % (100*(knn_pred == target_test).sum()/len(feature_test)))


# In[18]:


#  Support Vector Classification
svc_mod = SVC()
svc_mod.fit(feature_train, target_train)
svc_pred = svc_mod.predict(feature_test)
print('SVC accuracy is %1.3f%%' % (100*(svc_pred == target_test).sum()/len(feature_test)))


# In[19]:


#  Decision Tree Classification
dtc = DecisionTreeClassifier()
dtc.fit(feature_train, target_train)
dtc_pred = dtc.predict(feature_test)
print('Decision Tree accuracy is %1.3f%%' % (100*(dtc_pred == target_test).sum()/len(feature_test)))


# In[20]:


#  Random Forest Classification
rf = RandomForestClassifier()
rf.fit(feature_train, target_train)
rf_pred = rf.predict(feature_test)
print('Random Forest accuracy is %1.3f%%' % (100*(rf_pred == target_test).sum()/len(feature_test)))


# In[21]:


#  Lets look at all of the results
results = pd.DataFrame()
results["Target"] = target_test
results["Logistic Regression"] = lr_pred
results["Naive Bayes"] = nb_pred
results["KNN"] = knn_pred
results["SVC"] = svc_pred
results["Decision Tree"] = dtc_pred
results["Random Forest"] = rf_pred


# In[22]:
results.head()