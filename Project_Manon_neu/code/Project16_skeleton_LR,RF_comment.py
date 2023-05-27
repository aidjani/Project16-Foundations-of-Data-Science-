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

print ('################################################################## Data Preprocessing ##################################################################')

#Inspect the data
print(data.head())
print("The dataset has ", data.shape[0], "rows ", data.shape[1], "columns and", data.shape[0]*data.shape[1], "entries.")

print(data.isna().sum()) #no missing data
print("******************Description of the data***************************")
print(data.describe())

#divide BMI into 4 categories (with WHO recommendations)
for i in range (len(data)):
    if data['BMI'][i] < 18.5:
        data['BMI'][i] = 'underweight'
    
    elif data['BMI'][i] >= 18.5 and  data['BMI'][i] <= 24.9:
        data['BMI'][i] = 'normalweight'
    
    elif data['BMI'][i] == 25 or data['BMI'][i] <= 29.9: 
        data['BMI'][i] = 'overweight'

    else: data['BMI'][i] = 'obesity'

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
#plt.show()

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
#data_encoded = pd.get_dummies(data, columns=cat_cols, drop_first=True)
data_encoded = pd.get_dummies(data, columns=cat_cols, drop_first=False)

print(data_encoded.head())


#Visualization of data balance: Bar plot (categorical data) 
fig_race = sns.countplot(data=data, x="Race", hue='HeartDisease', dodge=False)
plt.xticks(rotation=45, ha = "right")
plt.savefig('./output/race_distribution.jpg')
#plt.show()

fig_diabetic = sns.countplot(data=data, x="Diabetic", hue='HeartDisease', dodge=False)
plt.xticks(rotation=45, ha = "right")
plt.savefig('./output/diabetic_distribution.jpg')
#plt.show()

fig_genhealth = sns.countplot(data=data, x="GenHealth", hue='HeartDisease', dodge=False)
plt.xticks(rotation=45, ha = "right")
plt.savefig('./output/genhealth_distribution.jpg')
#plt.show()

fig_age= sns.countplot(data, x = "AgeCategory", order = age_categories, hue='HeartDisease', dodge = False) #definetely not normally distributed 
plt.xticks(rotation=45, ha = "right")
plt.savefig('./output/age_distribution.jpg')
plt.tight_layout()
#plt.show()

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
cat_vars = ['HeartDisease', 'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease', 'SkinCancer']
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
plt.savefig('./output/categorical_heatmap.jpg', dpi=300)
#plt.show()


#Loose ends 
    for col in cat_cols:    
    print("Categories in ", col, ": ", data[col].unique()) #how many categories can the categorical variable take?
    if col == "Sex":
        data['Sex'] = data['Sex'].map({True: 'Male', False: 'Female'})
    elif data[col].nunique() == 2:
        data[col] = data[col].astype(bool).map({True: 'Yes', False: 'No'})


        
BMI_hist = sns.histplot(x = data["BMI"], kde = True)
BMI_hist.set_xlabel("BMI")
BMI_hist.set_xlabel("Count")
BMI_hist.set_title("BMI Distribution")
plt.savefig('./output/BMI.jpg')
plt.tight_layout()
#plt.show()

phys_hist = sns.countplot(data, x = "PhysicalHealth", hue='HeartDisease', dodge = False)
phys_hist.set_xlabel("Physical health within 30 days [score from 0-30]")
phys_hist.set_xlabel("Count")
phys_hist.set_title("Physical Health Distribution")
plt.savefig(./output/physical_health.jpg')
plt.tight_layout()
#plt.show()
'''



print("################################################################## Logistic Regression & Random Forest & K Nearest Neighbour ##################################################################")


''' Import packages that are needed '''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, confusion_matrix, auc
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


''' get confusion matrix of a classifier yielding predictions (y_pred) for the true class labels (y) '''
def get_confusion_matrix(y, y_pred):
   
    # true/false pos/neg.
    tp = 0 # tp = true positive
    fp = 0 # fp = false positive
    tn = 0 # tn = true negative
    fn = 0 # fn = false negative

    # summarize positive and negative classes
    for i in range (len(y)):
        if y[i] == 1 and y_pred[i] == 1:
            tp += 1                        
        elif y[i] == 0 and y_pred[i] == 1:
            fp += 1                         
        elif y[i] == 0 and y_pred[i] == 0:
            tn += 1
        elif y[i] == 1 and y_pred[i] == 0:
            fn += 1
    return tn, fp, fn, tp


''' compute the evaltuation metrics for the provided classifier (given: true labels, input features) '''
def evaluation_metrics(clf, y, X, ax,legend_entry='my legendEntry'):

    # calculate labels prediction
    y_test_pred = clf.predict(X)
    
    #convert y into an array-> same dimension as y_test_pred (=> needed for def get_confusion_matrix)
    y = y.values

    # Calculate the confusion matrix
    tn, fp, fn, tp = get_confusion_matrix(y, y_test_pred)

    # Ensurance to get the correct confusion matrix
    tn_sk, fp_sk, fn_sk, tp_sk = confusion_matrix(y, y_test_pred).ravel()
    if np.sum([np.abs(tp-tp_sk) + np.abs(tn-tn_sk) + np.abs(fp-fp_sk) + np.abs(fn-fn_sk)]) >0:
        print(' Confusion matrix failed. ')
        tn = tn_sk
        tp = tp_sk
        fn = fn_sk
        fp = fp_sk
    else:
        print(' Confusion matrix implementation succeded. ')

    # Calculate evaluation metrics
    precision   = tp / (tp + fp)
    specificity = tn / (tn + fp)
    accuracy    = (tp + tn) / (tp + fp + tn + fn)
    recall      = tp / (tp + fn)
    f1          = tp/(tp + 0.5*(fp+fn))

    # Get the roc curve
    y_test_predict_proba  = clf.predict_proba(X)[:, 1]
    fp_rates, tp_rates, _ = roc_curve(y, y_test_predict_proba)

    # Calculate the area under the roc curve (with auc)
    roc_auc = auc(fp_rates, tp_rates)

    # Plot on the provided axis
    ax.step(fp_rates, tp_rates, where='post', label=legend_entry + ' (AUC = {:.2f})'.format(roc_auc))

    return [accuracy, precision, recall, specificity, f1, roc_auc]


''' definining x und y for the following machine learnings '''
data_encoded['HeartDisease'] = data_encoded['HeartDisease'].replace({'Yes': 1, 'No': 0}) # already converted into boolean BUT (True/Fasle)
X  = data_encoded.copy().drop('HeartDisease', axis = 1)
y  = data_encoded['HeartDisease']


''' prepare the splits (perform a 5-fold crossvalidation) '''
n_splits = 5
skf      = StratifiedKFold(n_splits=n_splits)


''' Prepare performance overview data frame '''
data_encoded_performance = pd.DataFrame(columns = ['fold','clf','accuracy','precision','recall','specificity','F1','roc_auc'])
data_encoded_LR_normcoef = pd.DataFrame(index = X.columns, columns = np.arange(n_splits))


''' save performance metrics for each crossvalidation fold, plot roc curve for each model '''
fold = 0
fig,axs = plt.subplots(1,2, figsize=(9, 4)) # with KNN: fig,axs = plt.subplots(1,3, figsize=(9, 4))

for train_index, test_index in skf.split(X, y):

    # subsets for training and testing
    X_test  = X.iloc[test_index]
    y_test  = y.iloc[test_index]
    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]

    # Standardize numerical features (with training set statistics)
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc  = sc.transform(X_test)

    # Logistic regression
    LR_clf = LogisticRegression(random_state=1)
    LR_clf.fit(X_train_sc, y_train)

    # Get all features that contribute to classification
    data_encoded_this_LR_coefs = pd.DataFrame(zip(X_train.columns, np.transpose(LR_clf.coef_[0])), columns=['features', 'coef'])
    data_encoded_LR_normcoef.iloc[:, fold] = data_encoded_this_LR_coefs['coef'].values / data_encoded_this_LR_coefs['coef'].abs().sum()

    # Random forest
    RF_clf = RandomForestClassifier(random_state=1)
    RF_clf.fit(X_train_sc, y_train)

 
    #KNN = 








    # Evaluate the classifiers Logistic Regression, Random Forest and K Nearest Neighbour
    eval_metrics_LR = evaluation_metrics(LR_clf, y_test, X_test_sc, axs[0], legend_entry=str(fold)) # Logistic Regression 
    data_encoded_performance.loc[len(data_encoded_performance)-1,:] = [fold,'LR']+eval_metrics_LR
    eval_metrics_RF = evaluation_metrics(RF_clf, y_test, X_test_sc, axs[1], legend_entry=str(fold)) # Random Forest 
    data_encoded_performance.loc[len(data_encoded_performance)-1, :] = [fold, 'RF'] + eval_metrics_RF
    # K Nearest Neighbour 

    # increase counter for folds
    fold += 1

''' plotting ROC_Curves '''
model_names = ['Logistic Regresssion (LR)', 'Random Forest (RF)'] #add KNN
for i, ax in enumerate(axs):
    ax.set_xlabel('False positive rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.plot([0, 1], [0, 1], color='r', linestyle='--', label='random\nclassifier')
    ax.set_title(model_names[i])
    ax.legend(title='Fold',loc='lower right')
    ax.grid(True)
plt.tight_layout()
plt.savefig('./output/roc_curves.png')


''' group performance metrics by classifier type '''
df_by_clf = data_encoded_performance.groupby('clf')


''' table with mean and standard deviation for each metric and classifier type'''
mean_acc = df_by_clf['accuracy'].mean() # accuracy
std_acc = df_by_clf['accuracy'].std()
mean_prec = df_by_clf['precision'].mean() #precision 
std_prec = df_by_clf['precision'].std()
mean_recall = df_by_clf['recall'].mean() #recall
std_recall = df_by_clf['recall'].std()
mean_spec = df_by_clf['specificity'].mean() #specificity
std_spec = df_by_clf['specificity'].std()
mean_F1 = df_by_clf['F1'].mean() #F1
std_F1 = df_by_clf['F1'].std()
mean_roc_auc = df_by_clf['roc_auc'].mean() #roc_auc
std_roc_auc = df_by_clf['roc_auc'].std()

table_data = {'Classifier': ['Logistic Regression', 'Random Forest'],
              'Accuracy (mean ± std)': [f"{mean_acc['LR']:.3f} ± {std_acc['LR']:.3f}", f"{mean_acc['RF']:.3f} ± {std_acc['RF']:.3f}"],
              'Precision (mean ± std)': [f"{mean_prec['LR']:.3f} ± {std_prec['LR']:.3f}", f"{mean_prec['RF']:.3f} ± {std_prec['RF']:.3f}"],
              'Recall (mean ± std)': [f"{mean_recall['LR']:.3f} ± {std_recall['LR']:.3f}", f"{mean_recall['RF']:.3f} ± {std_recall['RF']:.3f}"],
              'Specificity (mean ± std)': [f"{mean_spec['LR']:.3f} ± {std_spec['LR']:.3f}", f"{mean_spec['RF']:.3f} ± {std_spec['RF']:.3f}"],
              'F1 score (mean ± std)': [f"{mean_F1['LR']:.3f} ± {std_F1['LR']:.3f}", f"{mean_F1['RF']:.3f} ± {std_F1['RF']:.3f}"],
              'ROC AUC score (mean ± std)': [f"{mean_roc_auc['LR']:.3f} ± {std_roc_auc['LR']:.3f}", f"{mean_roc_auc['RF']:.3f} ± {std_roc_auc['RF']:.3f}"]}

table = pd.DataFrame(table_data)
table.to_csv('./output/table_metrics.csv', index=False)


''' get top features (coefficients across five folds) and visualization '''
data_encoded_LR_normcoef['importance_mean'] = data_encoded_LR_normcoef.mean(axis =1)
data_encoded_LR_normcoef['importance_std']  = data_encoded_LR_normcoef.std(axis =1)
data_encoded_LR_normcoef['importance_abs_mean'] = data_encoded_LR_normcoef.abs().mean(axis =1)
data_encoded_LR_normcoef.sort_values('importance_abs_mean', inplace = True, ascending=False)

data_encoded_LR_normcoef['mean'] = data_encoded_LR_normcoef.mean(axis=1) #mean of normalized coefficients 
data_encoded_LR_normcoef_sorted = data_encoded_LR_normcoef.sort_values(by='mean', ascending=False) # sort coefficients by mean 

fig, ax = plt.subplots(figsize=(8,6)) # visualization of each feature by normalized coefficient
data_encoded_LR_normcoef_sorted.plot(kind='bar', y='mean', yerr=data_encoded_LR_normcoef_sorted.std(axis=1), legend=False, ax=ax) # give the values in the plot
ax.set_title(' Feature importance of different classification models ') # Logistic Regression, Random Forest, K Nearest Neigbour Importance
ax.set_xlabel('Features')
ax.set_ylabel('Normalized Coefficient')
plt.tight_layout()
plt.savefig('./output/importance.png')

coefficients = pd.DataFrame(zip(X_train.columns, np.transpose(LR_clf.coef_[0])), columns=['features', 'coef']) # using a table to visualize the coefficients
coefficients = coefficients.sort_values(by='coef', ascending=False) 
coefficients.to_csv('./output/table_with_coefficients.csv', index=False)

''' save performance table to csv '''
data_encoded_performance.to_csv('./output/table_with_performance.csv', index=False)