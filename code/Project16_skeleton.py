import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import scipy.stats as sts
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
from sklearn.cluster import DBSCAN
import pandas as pd



#Import data
data = pd.read_csv("./data/heart_2020_cleaned.csv")

#Inspect the data
print(data.head())
print("The dataset has ", data.shape[0], "rows ", data.shape[1], "columns and", data.shape[0]*data.shape[1], "entries.")

print(data.isna().sum()) #no missing data
print("******************Description of the data***************************")
print(data.describe())

for col in data.columns:
    unique_values = data[col].unique()
    unique_count = len(unique_values)
    print("Sum of unique values in", col, ":", unique_count)
columns = data.columns.values.tolist() #list of all column names
#print(columns)

#inspect the data types
num_cols = []
cat_cols = []
print(data.dtypes) #data types are float64 and object


#age and general health should be numerical and not categorical data!
for col in columns:
    if data[col].dtype == "float64":
       num_cols.append(col) 
    else:
        cat_cols.append(col)
cat_cols.remove('HeartDisease') #since we want to predict heart disease based on the other variables
print(num_cols)
print(cat_cols)

age_categories = data['AgeCategory'].unique()
age_categories.sort()
print(age_categories)

print("*******************Numerical Data*******************")

#check the data for normality in numerical data using the Shapiro-Wilk Test
for col in num_cols:    
    sampled_data =  data[col].sample(n = 5000, random_state = 42) #because of warning that above 5000 the test may not work
    p = sts.shapiro(sampled_data).pvalue 
    print(f"Test for normality of {col}: p={p:.10f}") 
    #all p values are below 0.05 --> none of the data is normally distributed 

#physical health, mental health and sleep time can be stored as integers rather than floats
data['PhysicalHealth'] = data['PhysicalHealth'].astype("int32") 
data['MentalHealth'] = data['MentalHealth'].astype("int32")
data['SleepTime'] = data['SleepTime'].astype("int32")
#plot the numerical data distributions


fig, axs = plt.subplots(2, 2, figsize=(12, 10))

print(num_cols)
for i in range (len(num_cols)):
    row = i // 2 
    col = i % 2
    col_name = num_cols[i]
    sns.histplot(data=data, x=col_name, ax=axs[row, col])
    axs[row, col].set_xlabel(col_name)
    axs[row, col].set_ylabel('Count')
    axs[row, col].set_title(col_name + ' Distribution')
#maybe rename the columns so that it is Mental Health and not MentalHealth
plt.savefig('./output/numerical_distributions.jpg')
plt.tight_layout()
plt.show()

#age is definitely not normally distributed
fig_age= sns.countplot(data, x = "AgeCategory", order = age_categories, hue='HeartDisease', dodge = False)
plt.savefig('./output/age_distribution.jpg')
plt.tight_layout()
plt.show()


#categorical data
print("*******************Categorical Data*******************")
#get the categories
print(cat_cols)
#is it better to do one-hot encoding or convert to booleans?



#one-hot encoding
print("*******************One-Hot Encoding*******************")
data_encoded = pd.get_dummies(data, columns=cat_cols, drop_first=True)
print(data_encoded.head())


'''for col in cat_cols:    
    print("Categories in ", col, ": ", data[col].unique()) #how many categories can the categorical variable take?
    if col == "Sex":
        data['Sex'] = data['Sex'].map({True: 'Male', False: 'Female'})
    elif data[col].nunique() == 2:
        data[col] = data[col].astype(bool).map({True: 'Yes', False: 'No'})


        
BMI_hist = sns.histplot(x = data["BMI"], kde = True)
BMI_hist.set_xlabel("BMI")
BMI_hist.set_xlabel("Count")
BMI_hist.set_title("BMI Distribution")
plt.savefig('../output/BMI.jpg')
plt.tight_layout()
#plt.show()

phys_hist = sns.countplot(data, x = "PhysicalHealth", hue='HeartDisease', dodge = False)
phys_hist.set_xlabel("Physical health within 30 days [score from 0-30]")
phys_hist.set_xlabel("Count")
phys_hist.set_title("Physical Health Distribution")
plt.savefig('../output/physical_health.jpg')
plt.tight_layout()
#plt.show()
'''
