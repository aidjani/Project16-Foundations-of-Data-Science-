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
from scipy.stats import chi2_contingency


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

'''
BMI recommendations: 
Underweight: < 18.5 
Normal weight: 18.5 – 24.9
Overweight: 25 – 29.9
Obesity: > = 30 

'''

for i in range (len(data)):
    if data['BMI'][i] < 18.5:
        data['BMI'][i] = 'underweight'
    
    elif data['BMI'][i] >= 18.5 and  data['BMI'][i] <= 24.9:
        data['BMI'][i] = 'normalweight'

    
    elif data['BMI'][i] == 2.5 or data['BMI'][i] <= 29.9: 
        data['BMI'][i] = 'overweight'

    else: data['BMI'][i] = 'obesity'

print(data.head())



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



#one-hot encoding
print("*******************One-Hot Encoding*******************")
#data_encoded = pd.get_dummies(data, columns=cat_cols, drop_first=True)
data_encoded = pd.get_dummies(data, columns=cat_cols, drop_first=False)

print(data_encoded.head())