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
##Pearson and Spearman correlations
print("*******************Pearson and Spearman correlations*******************")
#Pearson and Spearman correlation between (HeartDisease, BMI)  
data_encoded_corr = pd.get_dummies(data, columns=cat_cols)
data_encoded_corr['HeartDisease'] = data_encoded_corr['HeartDisease'].map({'No': 0, 'Yes': 1})
data_encoded_corr['BMI'] = pd.to_numeric(data_encoded_corr['BMI'])

print(data_encoded_corr.head())

print("Correlation between Heart Disease and BMI:")
print(
    f"\t Pearson:  {sts.pearsonr(data_encoded_corr['HeartDisease'], data_encoded_corr['BMI'])[0]:.3f}"
    + f" | p={sts.pearsonr(data_encoded_corr['HeartDisease'], data_encoded_corr['BMI'])[1]:.10f}"
)
print(
    f"\t Spearman: {sts.spearmanr(data_encoded_corr['HeartDisease'], data_encoded_corr['BMI']).correlation:.3f}"
    + f" | p={sts.spearmanr(data_encoded_corr['HeartDisease'], data_encoded_corr['BMI']).pvalue:.10f}"
)

#Pearson and Spearman correlation between (HeartDisease, PhysicalHealth)  
data_encoded_corr['PhysicalHealth'] = pd.to_numeric(data_encoded_corr['PhysicalHealth'])

print("Correlation between Heart Disease and Physical Health:")
print(
    f"\t Pearson:  {sts.pearsonr(data_encoded_corr['HeartDisease'], data_encoded_corr['PhysicalHealth'])[0]:.3f}"
    + f" | p={sts.pearsonr(data_encoded_corr['HeartDisease'], data_encoded_corr['PhysicalHealth'])[1]:.10f}"
)
print(
    f"\t Spearman: {sts.spearmanr(data_encoded_corr['HeartDisease'], data_encoded_corr['PhysicalHealth']).correlation:.3f}"
    + f" | p={sts.spearmanr(data_encoded_corr['HeartDisease'], data_encoded_corr['PhysicalHealth']).pvalue:.10f}"
)

#Pearson and Spearman correlation between (HeartDisease, MentalHealth)
data_encoded_corr['MentalHealth'] = pd.to_numeric(data_encoded_corr['MentalHealth'])

print("Correlation between Heart Disease and Mental Health:")
print(
    f"\t Pearson:  {sts.pearsonr(data_encoded_corr['HeartDisease'], data_encoded_corr['MentalHealth'])[0]:.3f}"
    + f" | p={sts.pearsonr(data_encoded_corr['HeartDisease'], data_encoded_corr['MentalHealth'])[1]:.10f}"
)
print(
    f"\t Spearman: {sts.spearmanr(data_encoded_corr['HeartDisease'], data_encoded_corr['MentalHealth']).correlation:.3f}"
    + f" | p={sts.spearmanr(data_encoded_corr['HeartDisease'], data_encoded_corr['MentalHealth']).pvalue:.10f}"
)

#Pearson and Spearman correlation between (HeartDisease, SleepTime)
data_encoded_corr['SleepTime'] = pd.to_numeric(data_encoded_corr['SleepTime'])

print("Correlation between Heart Disease and Sleep Time:")
print(
    f"\t Pearson:  {sts.pearsonr(data_encoded_corr['HeartDisease'], data_encoded_corr['SleepTime'])[0]:.3f}"
    + f" | p={sts.pearsonr(data_encoded_corr['HeartDisease'], data_encoded_corr['SleepTime'])[1]:.10f}"
)
print(
    f"\t Spearman: {sts.spearmanr(data_encoded_corr['HeartDisease'], data_encoded_corr['SleepTime']).correlation:.3f}"
    + f" | p={sts.spearmanr(data_encoded_corr['HeartDisease'], data_encoded_corr['SleepTime']).pvalue:.10f}"
)


##Heatmap 
#cat_vars = ['HeartDisease', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth']
cat_vars = ['HeartDisease', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease', 'SkinCancer']
# Create a contingency table for each pair of categorical variables
contingency_table = pd.DataFrame(index=cat_vars, columns=cat_vars)

for var1 in cat_vars:
    for var2 in cat_vars:
        # Create a cross-tabulation between the variables
        cross_tab = pd.crosstab(data[var1], data[var2])
        
        # Perform chi-square test and extract the chi-square statistic
        chi2, _, _, _ = chi2_contingency(cross_tab)
        
        # Calculate Cramér's V statistic
        n = cross_tab.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * min(cross_tab.shape) - 1))
        
        # Assign the Cramér's V value to the contingency table
        contingency_table.loc[var1, var2] = cramers_v

# Create a heatmap for the contingency table
plt.figure(figsize=(10, 8))
sns.heatmap(contingency_table.astype(float), annot=False, cmap='coolwarm', fmt='.2f',vmin=0, vmax=0.2 )
plt.title("Categorical Variables Heatmap (Cramér's V)")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0) 
plt.tight_layout()
plt.savefig('../output/categorical_heatmap.jpg', dpi=300)
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
