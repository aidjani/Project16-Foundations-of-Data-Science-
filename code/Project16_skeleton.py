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
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_curve, confusion_matrix, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
from sklearn.cluster import DBSCAN
import pandas as pd
from scipy.stats import chi2_contingency
#Version martina 25.05.23, 15:30 Uhr
#Import data
data = pd.read_csv("./data/heart_2020_cleaned.csv")

#Inspect the data
print(data.head())
print("The dataset has ", data.shape[0], "rows ", data.shape[1], "columns and", data.shape[0]*data.shape[1], "entries.")


print("******************Description of the data***************************")
print(data.describe())

for col in data.columns:
    unique_values = data[col].unique()
    unique_count = len(unique_values)
    print("Sum of unique values in", col, ":", unique_count)
columns = data.columns.values.tolist() #list of all column names
#print(columns)
print("*******************Cleaning the data*******************")
#data = data.drop(data[data['SleepTime'] > 20].index) #based on research, it is not possible to sleep more than 20 h per night
print(data.isna().sum()) #no missing data
print("*******************Data Type Inspection*******************")

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

print(num_cols)
print(cat_cols)
#physical health, mental health and sleep time can be stored as integers rather than floats
data['PhysicalHealth'] = data['PhysicalHealth'].astype("int32") 
data['MentalHealth'] = data['MentalHealth'].astype("int32")
data['SleepTime'] = data['SleepTime'].astype("int32")
for col in cat_cols:    
    print("Categories in ", col, ": ", data[col].unique()) #nr of categories the categorical variable has
    if col != "Sex" and data[col].nunique() == 2: #convert to booleans
        data[col] = data[col].map({'Yes': True, 'No': False})

#sort the age categories by age
age_categories = data['AgeCategory'].unique()
age_categories.sort()
print(age_categories)


cat_cols.remove('HeartDisease') #since we want to predict heart disease based on the other variables
print(data.dtypes)
print(data.head())
print("*******************Numerical Data*******************")

#check the data for normality in numerical data using the Shapiro-Wilk Test
for col in num_cols:    
    sampled_data =  data[col].sample(n = 5000, random_state = 42) #because of warning that above 5000 the test may not work
    p = sts.shapiro(sampled_data).pvalue 
    print(f"Test for normality of {col}: p={p:.10f}") 
    #all p values are below 0.05 --> none of the data is normally distributed 


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
#plt.savefig('./output/numerical_distributions.jpg')
plt.tight_layout()
#plt.show()

#put this part in the bar plot visualization want to keep here too tho?



#categorical data
print("*******************Categorical Data*******************")


#age is definitely not normally distributed
fig_age= sns.countplot(data, x = "AgeCategory", order = age_categories, hue='HeartDisease', dodge = False)
fig_age.set_title("Age Distribution with Heart Disease")
#plt.savefig('/output/age_distribution.jpg')
plt.tight_layout()
#plt.show()

#put all this back in again later

#Visualization of data balance: Bar plot (categorical data) 
fig_race = sns.countplot(data=data, x="Race", hue='HeartDisease', dodge=False)
fig_age.set_title("Race Distribution and Heart Disease")
plt.xticks(rotation=45, ha = "right")
#plt.savefig('../output/race_distribution.jpg')
#plt.show()

fig_diabetic = sns.countplot(data=data, x="Diabetic", hue='HeartDisease', dodge=False)
fig_age.set_title("Diabetic Stages and Heart Disease")
plt.xticks(rotation=45, ha = "right")
#plt.savefig('../output/diabetic_distribution.jpg')
#plt.show()

fig_genhealth = sns.countplot(data=data, x="GenHealth", hue='HeartDisease', dodge=False)
plt.xticks(rotation=45, ha = "right")
fig_age.set_title("General Health and Heart Disease")

#plt.savefig('../output/genhealth_distribution.jpg')
#plt.show()

#is it better to do one-hot encoding or convert to booleans?





#statistical tests for categorical variables (in the list cat_cols)
#check what a good sample size is for the chi2 test
print("*******************Statistical testing for categorical variables (Chi-squared)*******************")

print("*******************sample size 500*******************")
alpha = 0.05 / len(cat_cols)  # multiple testing adjustment
# Randomly sample 500 observations from the column
for col in cat_cols:   
    sampled_data = data[col].sample(n=500, random_state=42)
    
    # Perform chi-square test on the sampled data
    contingency_table = pd.crosstab(sampled_data, data["HeartDisease"])
    chi2, p, _, _ = sts.chi2_contingency(contingency_table)
    
    # Interpret the results
    if p > alpha:
        print(f"No statistical correlation between {col} and Heart Disease (p={p:.4f})." )
    else:
        print(f"Statistical correlation between {col} and Heart Disease (p={p:.4f}).")
print("*******************sample size 5000*******************")
# Randomly sample 5000 observations from the column
for col in cat_cols:   
    sampled_data = data[col].sample(n=5000, random_state=42)
    
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
alpha = 0.05 / len(num_cols)
for col in num_cols:
    p = sts.ranksums(
                data[data["HeartDisease"]][col],
                data[~data["HeartDisease"]][col],
            ).pvalue
    if p > alpha:
            print(f"No statistical correlation between {col} and Heart Disease (p={p:.4f})." )
    else:
        print(f"Statistical correlation between {col} and Heart Disease (p={p:.4f}).")


#KNN Classification

print("*******************K-Nearest Neighbour*******************")

def get_confusion_matrix(y,y_pred):
 # true/false pos/neg.
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # define positive and negative classes.
    
    for i in range(0, len(y)):
        if y[i] == 1:
            # positive class.
            if y_pred[i] == 1:
                tp += 1
            else:
                fn += 1
        else:
            # negative class.
            if y_pred[i] == 0:
                tn += 1
            else:
                fp += 1
    return tn, fp, fn, tp

def evaluation_metrics(clf, y, X, ax,legend_entry='my legendEntry'):
    # Get the label predictions
    y_test_pred    = clf.predict(X)

    # Calculate the confusion matrix given the predicted and true labels with your function
    tn, fp, fn, tp = get_confusion_matrix(y, y_test_pred)

    # Ensure that you get correct values - this code will divert to
    # sklearn if your implementation fails - you can ignore those lines
    tn_sk, fp_sk, fn_sk, tp_sk = confusion_matrix(y, y_test_pred).ravel()
    if np.sum([np.abs(tp-tp_sk) + np.abs(tn-tn_sk) + np.abs(fp-fp_sk) + np.abs(fn-fn_sk)]) >0:
        print('OWN confusion matrix failed!!! Reverting to sklearn.')
        tn = tn_sk
        tp = tp_sk
        fn = fn_sk
        fp = fp_sk
    else:
        print(':) Successfully implemented the confusion matrix!')

    precision   = tp / (tp + fp)
    specificity = tn / (tn + fp)
    accuracy    = (tp + tn) / (tp + fp + tn + fn)
    recall      = tp / (tp + fn)
    f1          = tp/(tp + 0.5*(fp+fn))

    # Get the roc curve using a sklearn function
    y_test_predict_proba  = clf.predict_proba(X)[:, 1]
    fp_rates, tp_rates, _ = roc_curve(y, y_test_predict_proba)

    # Calculate the area under the roc curve using a sklearn function
    roc_auc = auc(fp_rates, tp_rates)

    # Plot on the provided axis
    ax.plot(fp_rates, tp_rates,label = legend_entry)


    return [accuracy,precision,recall,specificity,f1, roc_auc]

#perform one-hot encoding
print("*******************One-Hot Encoding*******************")
X = data.drop('HeartDisease', axis = 1)
X.head()
y = data['HeartDisease']
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
print(X_encoded.head())

#5-fold crossvalidation

n_splits = 5
df_performance = pd.DataFrame(columns = ['fold','accuracy','precision','recall',
                                         'specificity','F1','roc_auc'])
df_KNN_normcoef = pd.DataFrame(index = X_encoded.columns, columns = np.arange(n_splits))
fold = 0
fig,axs = plt.subplots(1,2,figsize=(9, 4))
skf = StratifiedKFold(n_splits)


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2)

# loop over every split  
for train_index, test_index in skf.split(X_encoded, y):

    # Get the relevant subsets for training and testing
    X_test  = X_encoded.iloc[test_index]
    y_test  = y.iloc[test_index]
    X_train = X_encoded.iloc[train_index]
    y_train = y.iloc[train_index]


    # Standardize only the numerical features 
    sc = StandardScaler()
    for col in [num_cols]: 
        X_train_sc = sc.fit_transform(X_train)
    X_test_sc  = sc.transform(X_test)

    #train model using the euclidian distance metric
    knn = NearestNeighbors(n_neighbors= 5, metric = 'euclidean')
    knn.fit(X_train, y_train)

    distances, indices = knn.kneighbors(X_test)
    y_pred = []
    #evaluation for k = 5
    eval_metrics = evaluation_metrics(y_test, X_test_sc, axs[0],legend_entry=str(fold))
    df_performance.loc[len(df_performance)-1,:] = [fold,'KNN']+eval_metrics


    # increase counter for folds
    fold += 1


'''
#iterate over several k to identify the best k
k_values = [i for i in range (1,31)]
scores = []

scaler = StandardScaler()
X = scaler.fit_transform(X)

for k in k_values:
    knn = NearestNeighbors(n_neighbors = k)
    #score = accuracy score for k = 5
    scores.append(np.mean())
#plot every k against the accuracy
sns.lineplot(x = k_values, y = scores, marker = 'o')
plt.xlabel("K Values")
plt.ylabel("Accuracy Score")

'''

