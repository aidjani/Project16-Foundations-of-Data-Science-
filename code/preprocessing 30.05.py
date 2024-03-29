import pandas as pd
import seaborn as sns
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import scipy.stats as sts
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, confusion_matrix, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import os
from sklearn.cluster import DBSCAN
import pandas as pd
from scipy.stats import chi2_contingency
#Version martina 24.05.23, 22:30 Uhr
#code seems to compile but takes like 20 min until it computes all folds.
#Import data
data = pd.read_csv("../data/heart_2020_cleaned.csv")

#Remove SleepTime 
data = data.drop('SleepTime', axis=1)

#Inspect the data
print(data.head())
print("The dataset has ", data.shape[0], "rows ", data.shape[1], "columns and", data.shape[0]*data.shape[1], "entries.")


print("******************Description of the data***************************")
print(data.describe())


#Unique Values 
for col in data.columns:
    unique_values = data[col].unique()
    unique_count = len(unique_values)
    print("Sum of unique values in", col, ":", unique_count)
columns = data.columns.values.tolist() #list of all column names
#print(columns)
#change category names of diabetes column
data['Diabetic'] = data['Diabetic'].replace('No, borderline diabetes', 'Borderline diabetes')
data['Diabetic'] = data['Diabetic'].replace('Yes (during pregnancy)', 'During pregnancy')
print("*******************Cleaning the data*******************")
#data = data.drop(data[data['SleepTime'] > 20].index) #based on research, it is not possible to sleep more than 20 h per night
print(data.isna().sum()) #no missing data

print("*******************Data Type Inspection*******************")

#inspect the data types
num_cols = []
cat_cols = []
print(data.dtypes) #data types are float64 and object

#divide BMI into 4 categories (with WHO recommendations)
for i in range(len(data)):
    if data.loc[i, 'BMI'] < 18.5:
        data.loc[i, 'BMI'] = 'underweight'
    elif 18.5 <= data.loc[i, 'BMI'] <= 24.9:
        data.loc[i, 'BMI'] = 'normal'
    elif 25 <= data.loc[i, 'BMI'] <= 29.9:
        data.loc[i, 'BMI'] = 'overweight'
    else:
        data.loc[i, 'BMI'] = 'obesity'


#age and general health should be numerical and not categorical data!
for col in columns:
    if data[col].dtype == "float64":
       num_cols.append(col) 
    else:
        cat_cols.append(col)

#count gender imbalance
print(data['Sex'].value_counts())

#count racial imbalance
print(data['Race'].value_counts())
        


print(num_cols)
print(cat_cols)
#physical health, mental health and sleep time can be stored as integers rather than floats
data['PhysicalHealth'] = data['PhysicalHealth'].astype("int32") 
data['MentalHealth'] = data['MentalHealth'].astype("int32")
#data['SleepTime'] = data['SleepTime'].astype("int32") removed sleeptime 
for col in cat_cols:    
    print("Categories in ", col, ": ", data[col].unique()) #nr of categories the categorical variable has
    if col != "Sex" and data[col].nunique() == 2: #convert all binary categories to booleans except sex
        data[col] = data[col].map({'Yes': True, 'No': False})

#sort the age categories by age
age_categories = data['AgeCategory'].unique()
age_categories.sort()
print(age_categories)


#cat_cols.remove('HeartDisease') #since we want to predict heart disease based on the other variables
print(data.dtypes)
print(data.head())

#print heart disease prevalence
heart_disease = sns.countplot(data, x = "HeartDisease")
plt.title("Heart Disease Distribution")
plt.savefig('../output/heartdisease_distributions.jpg')
plt.tight_layout()
plt.show()


print("*******************Numerical Data*******************")

#check the data for normality in numerical data using the Shapiro-Wilk Test
for col in num_cols:    
    sampled_data =  data[col].sample(frac = 0.01, random_state = 42) #because of warning that above 5000 the test may not work
    p = sts.shapiro(sampled_data).pvalue 
    print(f"Test for normality of {col}: p={p:.10f}") 
    #all p values are below 0.05 --> none of the data is normally distributed 


'''
#plot the numerical data distributions
fig, axs = plt.subplots(2, 1, figsize=(12, 10))
#divide physical and mental health into smaller categories for visualization 
for j in (num_cols):
    for i in range(len(data)):
        if data.loc[i, j] < 5:
            data.loc[i, j] = 'below 5 days'
        elif data.loc[i, j] < 10:
            data.loc[i, j] = '5 - 10 days'
        elif data.loc[i, j] < 20:
            data.loc[i, j] = '10 - 20 days'
        else:
            data.loc[i, j] = '20 - 30 days'



# Define the order of categories
categories = ['below 5 days', '5 - 10 days', '10 - 20 days', '20 - 30 days']

# Plot countplot for physical health
sns.countplot(data=data, x="PhysicalHealth", order=categories, hue='HeartDisease', dodge=True, ax=axs[0])
axs[0].set_title('Physical Health Distribution')

# Plot countplot for mental health
sns.countplot(data=data, x="MentalHealth", order=categories, hue='HeartDisease', dodge=True, ax=axs[1])
axs[1].set_title('Mental Health Distribution')


#maybe rename the columns so that it is Mental Health and not MentalHealth
plt.savefig('../output/numerical_distributions.jpg')
plt.tight_layout()
plt.show()

'''



#statistical tests for categorical variables (in the list cat_cols)
#check what a good sample size is for the chi2 test
print("*******************Statistical testing for categorical variables (Chi-squared)*******************")

print("*******************sample size 10 %*******************")
alpha = 0.05 / len(cat_cols)  # multiple testing adjustment
print("alpha = ", alpha)
# Randomly sample 10 % observations from the column
for col in cat_cols:   
    sampled_data = data[col].sample(frac=0.1, random_state=42) #sample 10 % of the data
    
    # Perform chi-square test on the sampled data
    contingency_table = pd.crosstab(sampled_data, data["HeartDisease"])
    chi2, p, _, _ = sts.chi2_contingency(contingency_table)
    
    # Interpret the results
    if p > alpha:
        print(f"No statistical correlation between {col} and Heart Disease (p={p:.4f})." )
    else:
        print(f"Statistical correlation between {col} and Heart Disease (p={p:.4f}).")
print("*******************sample size 1 %*******************")
# Randomly sample 1% of observations from the column
for col in cat_cols:   
    sampled_data = data[col].sample(frac = 0.01, random_state=42)
    
    # Perform chi-square test on the sampled data
    contingency_table = pd.crosstab(sampled_data, data["HeartDisease"])
    chi2, p, _, _ = sts.chi2_contingency(contingency_table)
    
    # Interpret the results
    if p > alpha:
        print(f"No statistical correlation between {col} and Heart Disease (p={p:.4f})." )
    else:
        print(f"Statistical correlation between {col} and Heart Disease (p={p:.4f}).")

print("*******************original sample size*******************")
for col in cat_cols:
    contingency_table = pd.crosstab(data[col], data["HeartDisease"])
    chi2, p, _, _ = sts.chi2_contingency(contingency_table)
    if p > alpha:
            print(f"No statistical correlation between {col} and Heart Disease (p={p:.4f})." )
    else:
        print(f"Statistical correlation between {col} and Heart Disease (p={p:.4f}).")

print("*******************Statistical testing for numerical variables (Wilcoxon ranksums)*******************")
alpha = 0.05 / len(num_cols) #bonferroni correction
print("alpha = ", alpha)
for col in num_cols:
    p = sts.ranksums(
                data[data["HeartDisease"]][col],
                data[~data["HeartDisease"]][col],
            ).pvalue
    if p > alpha:
            print(f"No statistical correlation between {col} and Heart Disease (p={p:.4f})." )
    else:
        print(f"Statistical correlation between {col} and Heart Disease (p={p:.4f}).")

###############################numerical data visualization
fig, axs = plt.subplots(1, 2, figsize=(18, 6))


# Violin plot for Physical Health vs. Heart Disease
sns.violinplot(data=data, x="HeartDisease", y="PhysicalHealth", ax=axs[0])
axs[0].set_title('Physical Health vs. Heart Disease')
axs[0].set_xlabel('Heart Disease')
axs[0].set_ylabel('Physical Health')

# Violin plot for Mental Health vs. Heart Disease
sns.violinplot(data=data, x="HeartDisease", y="MentalHealth", ax=axs[1])
axs[1].set_title('Mental Health vs. Heart Disease')
axs[1].set_xlabel('Heart Disease')
axs[1].set_ylabel('Mental Health')

plt.tight_layout()
plt.savefig('../output/numerical_subplots_with.jpg')
plt.show()

############################################demographic data
##Create subplots with HD
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Countplot for Age vs. Heart Disease
sns.countplot(data=data, x="AgeCategory", order=age_categories, hue="HeartDisease", ax=axs[0, 0], dodge = True)
axs[0, 0].set_title('Age Distribution Plot')
axs[0, 0].set_xlabel('Age Category')
axs[0, 0].set_ylabel('Count')

# Countplot for Race vs. Heart Disease
sns.countplot(data=data, x="Race", hue="HeartDisease", ax=axs[0, 1], dodge = True)
axs[0, 1].set_title('Race Distribution Plot')
axs[0, 1].set_xlabel('Race')
axs[0, 1].set_ylabel('Count')

# countplot for BMI vs. Heart Disease
sns.countplot(data=data, x="BMI", hue="HeartDisease", ax=axs[1, 0], dodge = True)
axs[1, 0].set_title('BMI Distribution Plot')
axs[1, 0].set_xlabel('BMI')
axs[1, 0].set_ylabel('Count')

# Countplot for Sex vs. Heart Disease
sns.countplot(data=data, x="Sex", hue="HeartDisease", ax=axs[1, 1], dodge = True)
axs[1, 1].set_title('Gender Distribution Plot')
axs[1, 1].set_xlabel('Sex')
axs[1, 1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('../output/demographicdata_subplots.jpg')


#plt.show()

#age with HD
fig, axs = plt.subplots(1, 1, figsize=(10, 8))
fig_age= sns.countplot(data, x = "AgeCategory", order = age_categories, hue='HeartDisease', dodge = True)
fig_age.set_title("Age Distribution")
plt.tight_layout()
plt.savefig('../output/age_plot.jpg')

#plt.show()

#############################################general health
fig, axs = plt.subplots(1, 4, figsize=(25, 10))
genhealthcat = ['Poor', 'Fair', 'Good', 'Very good',  'Excellent']
# Plot countplot for DifficultyWalking column
sns.countplot(data=data, x="DiffWalking", hue="HeartDisease", dodge=True, ax=axs[0])
axs[0].set_title('Difficulty Walking Distribution Plot')
axs[0].set_xlabel('Difficulty Walking')
axs[0].set_ylabel('Count')

# Plot countplot for GeneralHealth column
sns.countplot(data=data, x="GenHealth", order = genhealthcat, hue="HeartDisease", dodge=True, ax=axs[1])
axs[1].set_title('General Health Distribution Plot')
axs[1].set_xlabel('General Health')
axs[1].set_ylabel('Count')

# Plot countplot for PhysicalActivity column
sns.countplot(data=data, x="PhysicalActivity", hue="HeartDisease", dodge=True, ax=axs[2])
axs[2].set_title('Physical Activity Distribution Plot')
axs[2].set_xlabel('Physical Activity')
axs[2].set_ylabel('Count')

sns.countplot(data=data, x="Smoking", hue="HeartDisease", dodge=True, ax=axs[3])
axs[3].set_title('Smoking Distribution Plot')
axs[3].set_xlabel('Smoking')
axs[3].set_ylabel('Count')



plt.tight_layout()
plt.savefig('../output/generalhealth_subplots_modified.jpg')

#plt.show()

fig, axs = plt.subplots(3, 2, figsize=(14, 16))

# Plot countplot for Diabetes column
sns.countplot(data=data, x="Diabetic", hue="HeartDisease", dodge=True, ax=axs[0, 0])
axs[0, 0].set_title('Diabetes Distribution Plot')
axs[0, 0].set_xlabel('Diabetes')
axs[0, 0].set_ylabel('Count')

# Plot countplot for Asthma column
sns.countplot(data=data, x="Asthma", hue="HeartDisease", dodge=True, ax=axs[0, 1])
axs[0, 1].set_title('Asthma Distribution Plot')
axs[0, 1].set_xlabel('Asthma')
axs[0, 1].set_ylabel('Count')

# Plot countplot for KidneyDisease column
sns.countplot(data=data, x="KidneyDisease", hue="HeartDisease", dodge=True, ax=axs[1, 0])
axs[1, 0].set_title('Kidney Disease Distribution Plot')
axs[1, 0].set_xlabel('Kidney Disease')
axs[1, 0].set_ylabel('Count')

# Plot countplot for alcohol column
sns.countplot(data=data, x="AlcoholDrinking", hue="HeartDisease", dodge=True, ax=axs[1, 1])
axs[1, 1].set_title('Alcohol Drinking Distribution Plot')
axs[1, 1].set_xlabel('Alcohol Drinking')
axs[1, 1].set_ylabel('Count')

# Plot countplot for SkinCancer column
sns.countplot(data=data, x="SkinCancer", hue="HeartDisease", dodge=True, ax=axs[2, 0])
axs[2, 0].set_title('Skin Cancer Distribution Plot')
axs[2, 0].set_xlabel('Skin Cancer')
axs[2, 0].set_ylabel('Count')

# Plot countplot for Stroke column
sns.countplot(data=data, x="Stroke", hue="HeartDisease", dodge=True, ax=axs[2, 1])
axs[2, 1].set_title('Stroke Distribution Plot')
axs[2, 1].set_xlabel('Stroke')
axs[2, 1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('../output/diseases_subplots_modified.jpg')



#Heatmap
# #cat_vars = ['HeartDisease', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth']
cat_vars = ['HeartDisease', 'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer']
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
print("*******************Feature Selection*******************")
#feature selection
data_selected = data.drop(["DiffWalking", "GenHealth", "Asthma", "BMI", "AlcoholDrinking", "Race", "MentalHealth"], axis=1)
print(data_selected.head())
#perform one-hot encoding
print("*******************One-Hot Encoding*******************")
y = data['HeartDisease'].replace({'True': 1, 'False': 0}).astype(int) #reconverted to integer from bool. MAYBE DO NOT CONVERT TO BOOL IN THE DATA PREPROCESSING
X_encoded = data.drop('HeartDisease', axis=1)
cat_cols.remove('HeartDisease')

print(cat_cols)
for col in cat_cols:
    X_encoded[col] = X_encoded[col].replace({'True': 1, 'False': 0})

X_encoded = pd.get_dummies(X_encoded, columns=cat_cols, drop_first=True)


print(X_encoded.head())
print(y.head())

