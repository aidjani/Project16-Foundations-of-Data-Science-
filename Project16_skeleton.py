import pandas as pd

# Read the data from the csv file
data = pd.read_csv('./heart_2020_cleaned.csv')
# have a look at the data
print (data)
print (data.shape)
print (data.head())
print (data.describe())
print (data.dtypes)
print (data.isna().sum())
print (data.duplicated().sum())
