# Import Statements
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data'
data = pd.read_csv(url, header = None)
data.head()

#  These were grabbed from the above website where they define the variables
data.columns = ['Class', 'Age', 'Sex', 'Steroid', 'Antiviral', 'Fatigue', 'Malaise', 'Anorexia', 'Liver Big', 
               'Liver Firm', 'Spleen Palpable', 'Spiders', 'Ascites', 'Varices', 'Bilirubin', 'Alk', 'Sgot',
               'Albumin', 'Protime', 'Histology']

categorical_data = ['Steroid', 'Antiviral', 'Fatigue', 'Malaise', 'Anorexia', 'Liver Big', 
               'Liver Firm', 'Spleen Palpable', 'Spiders', 'Ascites', 'Varices', 'Histology']

non_categorical_data = ['Age', 'Bilirubin', 'Alk', 'Sgot', 'Albumin', 'Protime']

#  We need to replace all '?' with None and convert all numbers to the same type
data = data.replace(to_replace="?", value=np.nan).apply(pd.to_numeric, errors = "coerse")
data.head()

#  1 is assigned if a patiend died, 2 if the patient lived
fig1, ax1 = plt.subplots()
ax1.pie(data.Class.value_counts(ascending=True), labels=['Die', 'Live'], autopct='%1.1f%%', startangle=90)
ax1.axis('equal')
plt.title("Patient Survival")
plt.show()

#  Plot all of the different columns
age = plt.hist(data.Age)
plt.title("Age")
plt.show()

#  1 is assigned if a patiend is male, 2 if the patient is female
## I choose pie chat to try out new ways of showing the binary data (ie, male or female)
fig2, ax2 = plt.subplots()
ax2.pie(data.Sex.value_counts(ascending=True), labels=['Male', 'Female'], autopct='%1.1f%%', startangle=90)
ax2.axis('equal')
plt.title("Sex")
plt.show()

# I am changing all of the 1's and 2's to No's and Yes's to make the charts look better
# I am also dropna()'ing all of these so that we don't have any ambiguity with the charts...
## ie, its either yes or no, we are not seeing the nan's in the data.
## We will look at the nan's later.  They also become more apparant in the stacked bar chart.
plt.hist(data.Steroid.dropna().replace(to_replace = [1, 2], value = ["No", "Yes"], inplace = False))
plt.title("Steroid Use")
plt.show()

plt.hist(data.Antiviral.replace(to_replace = [1, 2], value = ["No", "Yes"], inplace = False))
plt.title("Antivaral Use")
plt.show()

plt.hist(data.Fatigue.dropna().replace(to_replace = [1, 2], value = ["No", "Yes"], inplace = False))
plt.title("Did Patient Experience Fatigue")
plt.show()

plt.hist(data.Malaise.dropna().replace(to_replace = [1, 2], value = ["No", "Yes"], inplace = False))
plt.title("Did Patient Experience Malaise")
plt.show()

plt.hist(data.Anorexia.dropna().replace(to_replace = [1, 2], value = ["No", "Yes"], inplace = False))
plt.title("Did Patient Experience Anorexia")
plt.show()

plt.hist(data['Liver Big'].dropna().replace(to_replace = [1, 2], value = ["No", "Yes"], inplace = False))
plt.title("Did Patient Experience A Big Liver")
plt.show()

plt.hist(data['Liver Firm'].dropna().replace(to_replace = [1, 2], value = ["No", "Yes"], inplace = False))
plt.title("Did Patient Experience A Firm Liver")
plt.show()

plt.hist(data['Spleen Palpable'].dropna().replace(to_replace = [1, 2], value = ["No", "Yes"], inplace = False))
plt.title("Did Patient Experience A Palpable Spleen")
plt.show()

plt.hist(data.Spiders.dropna().replace(to_replace = [1, 2], value = ["No", "Yes"], inplace = False))
plt.title("Did Patient Experience Spiders")
plt.show()

plt.hist(data.Ascites.dropna().replace(to_replace = [1, 2], value = ["No", "Yes"], inplace = False))
plt.title("Did Patient Experience Ascites")
plt.show()

plt.hist(data.Varices.dropna().replace(to_replace = [1, 2], value = ["No", "Yes"], inplace = False))
plt.title("Did Patient Experience A Palpable Spleen")
plt.show()

plt.hist(data.Bilirubin.dropna())
plt.title("Bilirubin Amount")
plt.show()

plt.hist(data.Alk.dropna())
plt.title("Alk Phosphate Amount")
plt.show()

plt.hist(data.Sgot.dropna())
plt.title("Sgot Amount")
plt.show()

plt.hist(data.Albumin.dropna())
plt.title("Albumin Amount")
plt.show()

plt.hist(data.Protime.dropna())
plt.title("Protime Amount")
plt.show()

plt.hist(data.Histology.replace(to_replace = [1, 2], value = ["No", "Yes"], inplace = False))
plt.title("Histolgy")
plt.show()

#  I did not like having all of the data as individual histograms, so I made a stacked bar chart
#  Since all elements have the same number of elements, the delta is the number of nan's in each column
print("Here is the data for the stacked bar chart: ")
data_stacked = data[categorical_data].apply(lambda x: pd.Series.value_counts(x.dropna()), axis=0)
data_stacked.rename({1.0: "No", 2.0: "Yes"}, axis='index', inplace = True)
print(data_stacked)
colors = ["#BA4343","#4167BA"]
plt.show(data_stacked.transpose().plot.bar(stacked=True, color=colors, figsize=(14,11)))

#  Plot the scatter martix
pd.plotting.scatter_matrix(data.apply(lambda x: pd.to_numeric(x)), figsize=(14,11))

#  This is going to find the number of missing variables
number_missing = data.apply(lambda x: x.isnull().sum(), axis=0)
print('Number of missing variables: ')
print(number_missing)

#  I am defining outlier as two standard deviates from the mean.
#  This will look for elements outside that region and list them out.
print('The following columns have missing values:')
for col in non_categorical_data:
    upper_limit = 2.0 * np.std(data[col].dropna()) + np.nanmean(data[col])
    lower_limit = -2.0 * np.std(data[col].dropna()) + np.nanmean(data[col])
    violation = [str(vio) for vio in data[col] if vio < lower_limit or vio > upper_limit]
    if(len(violation) > 0): print(col + ' has outliers: ' + ', '.join(violation))