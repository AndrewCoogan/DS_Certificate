
# coding: utf-8

# In[1]:


# Import Statements
import numpy as np
import pandas as pd
from sklearn.metrics import *
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
import matplotlib.pyplot as plt


# In[2]:


#  Import data from the UCI database
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
data = pd.read_csv(url, header=None)
#data = pd.read_csv("AndrewCoogan-M02-Dataset.csv")
data.columns = ["Age", "Workclass", "FinlWeight", "Education", "EducationNum", "MaritalStatus", 
                "Occupation", "Relationship", "Race", "Sex", "CapitolGain", "CapitolLoss", 
                "HoursPerWeek", "NativeCountry", "Income"]


# In[3]:


#  Finalweight is not a valid factor
#  Education and Education Num are too highly correlated
data.drop("EducationNum", inplace = True, axis = 1)
data.drop("FinlWeight", inplace = True, axis = 1)


# In[4]:


#  This is just to help me seperate the column names
categorical_data = ['Workclass', 'Education', 'MaritalStatus', 'Occupation', 'Relationship', 
                    'Race', 'Sex', 'NativeCountry']

non_categorical_data = ['Age', 'CapitolGain', 'CapitolLoss', 'HoursPerWeek']
target = ['Income']


# In[5]:


#  Occupation, Workclass, NativeCountry all contain missing values 
#  We need to clean the data and see what we need to fix
data[categorical_data] = data[categorical_data].apply(lambda x : x.str.strip(), axis = 0)
data = data.replace(to_replace="?", value=np.nan)
data.apply(lambda x: x.isnull().sum(), axis=0)


# In[6]:


# Workclass and Occupation share the vast majort the missing values, so I am going to remove those
# Those missing values only account for 5% of the data, so that should be fine to trim
# I am going to keep the unknown NativeCountry as a dummy variable
data = data.loc[~data.Workclass.isnull(),:]
data = data.loc[~data.Occupation.isnull(),:]


# In[7]:


#  I am going to set all of the missing NativeCountry column entries to UnknownNC
#  I will allow the svc to detemine if knowing the country of origin is important or not
data.NativeCountry.loc[data.NativeCountry.isnull()] = "UnknownNC"


# In[8]:


#  This was for testing purposes.
#data_orig = data.copy()


# In[9]:


#  Get dummies breaks down the categeorical columns into binary columns.
#  This is going to make the dataframe go from 10 or so columns to well over 100 with
#     all the different combination of column elements.
data_f = pd.get_dummies(data[categorical_data])


# In[10]:


#  I am going to normalize all of this using mean variance
data_f[non_categorical_data] = data[non_categorical_data].apply(lambda l: (l-np.mean(l)) / np.std(l))


# In[11]:


#  This will break the pandas datatable into four new datatables:
# - Feature training set
# - Feature testing set
# - Target training set
# - Target testing set
feature_train, feature_test, target_train, target_test = train_test_split(
    data_f[data_f.columns.difference(target)], 
    data[target], 
    random_state=0,
    test_size = 0.7) 


# In[12]:


#  Replacing the categories with 0 if person makes less than 50k and 1 if a person makes more
target_test = target_test.Income.str.strip().replace(["<=50K", ">50K"], [0, 1])
target_train = target_train.Income.str.strip().replace(["<=50K", ">50K"], [0, 1])


# In[13]:


#  Lets start with a SVC Model because I think it is very robust.  SVC is a model
# that is routinely seen in finance, so I have a slght affinity to SVC.

svc_mod = SVC(probability=True)
svc_mod.fit(feature_train, target_train)
svc_pred = svc_mod.predict(feature_test)
svc_prob = svc_mod.predict_proba(feature_test)
prob_pos_svc = svc_prob[:,1]


# In[14]:

print('SVC Classifier')
#  Code to create the confusion matrix and outputs
CM_svc = confusion_matrix(target_test, svc_pred)
print ("\n\nConfusion matrix:\n", CM_svc)
tn_svc, fp_svc, fn_svc, tp_svc = CM_svc.ravel()
print ("\nTP, TN, FP, FN:", tp_svc, ",", tn_svc, ",", fp_svc, ",", fn_svc)
AR_svc = accuracy_score(target_test, svc_pred)
print ("\nAccuracy rate:", np.round(AR_svc,4))
ER_svc = 1.0 - AR_svc
print ("\nError rate:", np.round(ER_svc,2))
P_svc = precision_score(target_test, svc_pred)
print ("\nPrecision:", np.round(P_svc, 2))
R_svc = recall_score(target_test, svc_pred)
print ("\nRecall:", np.round(R_svc, 2))
F1_svc = f1_score(target_test, svc_pred)
print ("\nF1 score:", np.round(F1_svc, 2))


# In[15]:


#  Settings for plot and ROC Curve for the SVC model
LW = 1.5
LL = "lower right"
LC = "darkgreen"

fpr_svc, tpr_svc, th_svc = roc_curve(np.array(target_test), np.array(prob_pos_svc))
AUC_svc = auc(fpr_svc, tpr_svc)
print ("\nTP rates:", np.round(tpr_svc, 2))
print ("\nFP rates:", np.round(fpr_svc, 2))
print ("\nProbability thresholds:", np.round(th_svc, 2))
print ("\nAUC score (using auc function):", np.round(AUC_svc, 2))
print("\n")
plt.figure()
plt.title("Receiver Operating Characteristic (ROC) Curve - SVC")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("FALSE Positive Rate")
plt.ylabel("TRUE Positive Rate")
plt.plot(fpr_svc, tpr_svc, color=LC,lw=LW, label="ROC curve (area = %0.2f)" % AUC_svc)
plt.plot([0, 1], [0, 1], color="navy", lw=LW, linestyle='--')
plt.legend(loc=LL)
plt.show()


# In[16]:


#  For my second classifier I am going to use random forest classification
rf = RandomForestClassifier()
rf.fit(feature_train, target_train)
rf_prob = rf.predict_proba(feature_test)
rf_pred = rf.predict(feature_test)
prob_pos_rf = rf_prob[:,1]


# In[19]:
# I am going to give a random forest classifier a try as its a different approcah
# to classification.  It is going to make a conbination of weaker classiferies to 
# make strong classifiers.  Since we have so many classifiers, this should prove
# to be a good model.

print('Random Forest Classifier')
#  Code to create the confusion matrix and outputs
CM_rf = confusion_matrix(target_test, rf_pred)
print ("\n\nConfusion matrix:\n", CM_rf)
tn_rf, fp_rf, fn_rf, tp_rf = CM_rf.ravel()
print ("\nTP, TN, FP, FN:", tp_rf, ",", tn_rf, ",", fp_rf, ",", fn_rf)
AR_rf = accuracy_score(target_test, rf_pred)
print ("\nAccuracy rate:", np.round(AR_rf,4))
ER_rf = 1.0 - AR_rf
print ("\nError rate:", np.round(ER_rf,2))
P_rf = precision_score(target_test, rf_pred)
print ("\nPrecision:", np.round(P_rf, 2))
R_rf = recall_score(target_test, rf_pred)
print ("\nRecall:", np.round(R_rf, 2))
F1_rf = f1_score(target_test, rf_pred)
print ("\nF1 score:", np.round(F1_rf, 2))


# In[21]:


#  We will keep same colour scheme as the SVC model
#  This will output the ROC curve for the random forest model

fpr_rf, tpr_rf, th_rf = roc_curve(np.array(target_test), np.array(prob_pos_rf))
AUC_rf = auc(fpr_rf, tpr_rf)
print ("\nTP rates:", np.round(tpr_rf, 2))
print ("\nFP rates:", np.round(fpr_rf, 2))
print ("\nProbability thresholds:", np.round(th_rf, 2))
print ("\nAUC score (using auc function):", np.round(AUC_rf, 2))
print("\n")
plt.figure()
plt.title("Receiver Operating Characteristic (ROC) Curve - Random Forest")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("FALSE Positive Rate")
plt.ylabel("TRUE Positive Rate")
plt.plot(fpr_rf, tpr_rf, color=LC,lw=LW, label="ROC curve (area = %0.2f)" % AUC_rf)
plt.plot([0, 1], [0, 1], color="navy", lw=LW, linestyle='--')
plt.legend(loc=LL)
plt.show()

