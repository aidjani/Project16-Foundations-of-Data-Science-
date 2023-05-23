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

#comment hellloooooooooooooooooooooooooooooooooooo

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

#put this part in the bar plot visualization want to keep here too tho?
''''
#age is definitely not normally distributed
fig_age= sns.countplot(data, x = "AgeCategory", order = age_categories, hue='HeartDisease', dodge = False)
plt.savefig('../output/age_distribution.jpg')
plt.tight_layout()
plt.show()
'''

#categorical data
print("*******************Categorical Data*******************")
#get the categories
print(cat_cols)
#is it better to do one-hot encoding or convert to booleans?



#one-hot encoding
print("*******************One-Hot Encoding*******************")
data_encoded = pd.get_dummies(data, columns=cat_cols, drop_first=True)
print(data_encoded.head())


#Visualization of data balance: Bar plot (categorical data) 
fig_race = sns.countplot(data=data, x="Race", hue='HeartDisease', dodge=False)
plt.xticks(rotation=45, ha = "right")
plt.savefig('../output/race_distribution.jpg')
plt.show()

fig_diabetic = sns.countplot(data=data, x="Diabetic", hue='HeartDisease', dodge=False)
plt.xticks(rotation=45, ha = "right")
plt.savefig('../output/diabetic_distribution.jpg')
plt.show()

fig_genhealth = sns.countplot(data=data, x="GenHealth", hue='HeartDisease', dodge=False)
plt.xticks(rotation=45, ha = "right")
plt.savefig('../output/genhealth_distribution.jpg')
plt.show()

fig_age= sns.countplot(data, x = "AgeCategory", order = age_categories, hue='HeartDisease', dodge = False) #definetely not normally distributed 
plt.xticks(rotation=45, ha = "right")
plt.savefig('../output/age_distribution.jpg')
plt.tight_layout()
plt.show()

''' #Tried to get all in one, but colours are off
for i, cat_var in enumerate(categorical_variables):
    ax = axes[i]  # Define the ax variable here

    # Calculate value counts for the current categorical variable
    category_counts = data[cat_var].value_counts()

    # Plot the bars for each category with different colors based on 'HeartDisease'
    bars = ax.bar(category_counts.index, category_counts.values, color=[heart_disease_colors.get(x, 'gray') for x in category_counts.index])

    ax.set_xlabel('Categories')
    ax.set_ylabel('Count')
    ax.set_title(f'Balance of {cat_var}')
    ax.tick_params(axis='x', labelrotation=45, labelsize=8)
    ax.margins(y=0.2)

plt.subplots_adjust(hspace=subplot_spacing, top=0.9)
plt.savefig('../output/balance_visualization.jpg')
plt.tight_layout()
plt.show()
'''
#Pearson and Spearman correlation between (HeartDisease, BMI) 
   #! now HeartDisease currently string and not numerical values so doesnt work yet 
''''
plt.figure()
scat = sns.scatterplot(x="HeartDisease", y="BMI", data=data)
scat.set_xlabel("Heart Disease (Yes/No)")
scat.set_ylabel("BMI")
scat.set_title("Heart Disease and BMI")
scat.text(
    x=1.5,
    y=30,
    s=r"Pearson's $\rho=$"
    + f"{sts.pearsonr(data['HeartDisease'], data['BMI']).correlation:.3f}",
    horizontalalignment="right",
)
scat.text(
    x=1.5,
    y=27,
    s=r"Spearman's $r=$"
    + f"{sts.spearmanr(data['HeartDisease'], data['BMI']).correlation:.3f}",
    horizontalalignment="right",
)
plt.savefig("../output/correlation-heartdisease-bmi.jpg")
plt.tight_layout()
plt.show()
print("Correlation between Heart Disease and BMI:")
print(
    f"\t Pearson:  {sts.pearsonr(data['HeartDisease'], data['BMI']).correlation:.3f}"
    + f" | p={sts.pearsonr(data['HeartDisease'], data['BMI']).pvalue:.10f}"
)
print(
    f"\t Spearman: {sts.spearmanr(data['HeartDisease'], data['BMI']).correlation:.3f}"
    + f" | p={sts.spearmanr(data['HeartDisease'], data['BMI']).pvalue:.10f}"
)
'''


#Heat Map 
# Calculate correlation matrix
corr_matrix = data_encoded.corr() #square matrix that shows the pairwise correlations between different variables( strength and direction of the linear relationship between variables)

# Create a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-0.5, vmax=0.5)
plt.title('Correlation Heatmap')
plt.xticks(rotation=45, ha = "right")

# Save the heatmap
plt.savefig('../output/correlation_heatmap.jpg')
plt.tight_layout()
plt.show()

#Loose ends 
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
