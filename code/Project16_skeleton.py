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




print("##################################################################Logistic Regression & Random Forest##################################################################")

# Import packages that are needed
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, confusion_matrix, auc
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Utility function to plot the diagonal line
def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes

# Functions you are asked to complete
def get_confusion_matrix(y, y_pred):
    """
    compute the confusion matrix of a classifier yielding
    predictions y_pred for the true class labels y
    :param y: true class labels
    :type y: numpy array

    :param y_pred: predicted class labels
    :type y_pred: numpy array

    :return: comfusion matrix comprising the
             true positives (tp),
             true negatives  (tn),
             false positives (fp),
             and false negatives (fn)
    :rtype: four integers
    """

    # true/false pos/neg.
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # define positive and negative classes.
    ''''
    for i in range (len(y)):
        if y[i] == 'Yes' and y_pred[i] == 'Yes':
            tp += 1
        elif y[i] == 'No' and y_pred[i] == 'Yes':
            fp += 1
        elif y[i] == 'No' and y_pred[i] == 'No':
            tn += 1
        elif y[i] == 'Yes' and y_pred[i] == 'No':
            fn += 1

    return tn, fp, fn, tp
    '''

    for i in range (len(y)):
        if y[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif y[i] == 0 and y_pred[i] == 0:
            tn += 1
        elif y[i] == 1 and y_pred[i] == 0:
            fn += 1


    '''
    for i in range (0, len(y)):
        if y[i, 1] == 1 and y_pred[i] == 1:
            tp += 1
        elif y[i, 1] == 0 and y_pred[i] == 1:
            fp += 1
        elif y[i, 1] == 0 and y_pred[i] == 0:
            tn += 1
        elif y[i, 1] == 1 and y_pred[i] == 0:
            fn += 1
    '''
    print(tn)

    return tn, fp, fn, tp

def evaluation_metrics(clf, y, X, ax,legend_entry='my legendEntry'):
    """
    compute multiple evaluation metrics for the provided classifier given the true labels
    and input features. Provides a plot of the roc curve on the given axis with the legend
    entry for this plot being specified, too.

    :param clf: true class labels
    :type clf: numpy array

    :param y: true class labels
    :type y: numpy array

    :param X: feature matrix
    :type X: numpy array

    :param ax: matplotlib axis to plot on
    :type legend_entry: matplotlib Axes

    :param legend_entry: the legend entry that should be displayed on the plot
    :type legend_entry: string

    :return: comfusion matrix comprising the
             true positives (tp),
             true negatives  (tn),
             false positives (fp),
             and false negatives (fn)
    :rtype: four integers
    """

    # Get the label predictions
    y_test_pred = clf.predict(X)
    

    #convert y into an array-> same dimension as y_test_pred (=> needed for def get_confusion_matrix)
    y = y.values


    print(len(y))
    print(y)
    print(y.shape)
    print(len(y_test_pred))
    print(y_test_pred)

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

    # Calculate the evaluation metrics
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


    return [accuracy, precision, recall, specificity, f1, roc_auc]

#Import data
#data = pd.read_csv("./data/heart_2020_cleaned.csv")

data_encoded['HeartDisease'] = data_encoded['HeartDisease'].replace({'Yes': 1, 'No': 0}) #-> already converted into boolean BUT (True/Fasle)
X  = data_encoded.copy().drop('HeartDisease', axis = 1)
y  = data_encoded['HeartDisease']

#print(X)
#print(y)


# Perform a 5-fold crossvalidation - prepare the splits
n_splits = 5
skf      = StratifiedKFold(n_splits=n_splits)


# Prepare the performance overview data frame
data_encoded_performance = pd.DataFrame(columns = ['fold','clf','accuracy','precision','recall','specificity','F1','roc_auc'])
data_encoded_LR_normcoef = pd.DataFrame(index = X.columns, columns = np.arange(n_splits))

# Use this counter to save your performance metrics for each crossvalidation fold
# also plot the roc curve for each model and fold into a joint subplot
fold = 0
fig,axs = plt.subplots(1,2, figsize=(9, 4))
# with KNN: fig,axs = plt.subplots(1,3, figsize=(9, 4))



# Loop over all splits
for train_index, test_index in skf.split(X, y):

    # Get the relevant subsets for training and testing
    X_test  = X.iloc[test_index]
    y_test  = y.iloc[test_index]
    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]

    #print('X_test', X_test, 'y_test', y_test, 'X_train', X_train, 'y_train', y_train)
    

    # Standardize the numerical features using training set statistics
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc  = sc.transform(X_test)

    #print('X_train_sc', X_train_sc)
    #print('X_test_sc', X_test_sc)

    # Creat prediction models and fit them to the training data

    # Logistic regression
    LR_clf = LogisticRegression(random_state=1)
    LR_clf.fit(X_train_sc, y_train)

    #print('LR_clf', clf)


    # Get the top 5 features that contribute most to the classification
    data_encoded_this_LR_coefs = pd.DataFrame(zip(X_train.columns, np.transpose(LR_clf.coef_[0])), columns=['features', 'coef'])
    data_encoded_LR_normcoef[data_encoded_this_LR_coefs['features']] = data_encoded_this_LR_coefs['coef'].values / data_encoded_this_LR_coefs['coef'].abs().sum()
    #data_encoded_LR_normcoef.loc[:,fold] = data_encoded_this_LR_coefs['coef'].values/data_encoded_this_LR_coefs['coef'].abs().sum()
    #data_encoded_this_LR_coefs = pd.DataFrame(zip(X_train.columns, np.transpose(clf.coef_)), columns=['features', 'coef'])
    #data_encoded_LR_normcoef.loc[:,fold] = data_encoded_this_LR_coefs['coef'].values



    #print('top 5 features', data_encoded_this_LR_coefs, data_encoded_LR_normcoef)

    # Random forest
    RF_clf = RandomForestClassifier(random_state=1)
    RF_clf.fit(X_train_sc, y_train)

    #print('RF_clf', RF_clf)

 
    #KNN = 








    # Evaluate the classifiers Logistic Regression, Random Forest and KNN
    eval_metrics = evaluation_metrics(LR_clf, y_test, X_test_sc, axs[0], legend_entry=str(fold))
    data_encoded_performance.loc[len(data_encoded_performance)-1,:] = [fold,'LR']+eval_metrics

    eval_metrics_RF = evaluation_metrics(RF_clf, y_test, X_test_sc, axs[1], legend_entry=str(fold))
    data_encoded_performance.loc[len(data_encoded_performance)-1, :] = [fold, 'RF'] + eval_metrics_RF

    print('eval metrics LR', eval_metrics)
    
    print('eval metrics RF', eval_metrics_RF)

    '''
    eval_metrics_KNN = evaluation_metrics(KNN_clf, y_test, X_test_sc, axs[2], legend_entry=str(fold))
    data_encoded_performance.loc[len(data_encoded_performance)-1, :] = [fold, 'RF'] + eval_metrics_RF
    '''

    # increase counter for folds
    fold += 1

''''
    print(data_encoded_performance.loc[len(data_encoded_performance)-1,:])
    print(data_encoded_performance.loc[len(data_encoded_performance)-1, :])
'''

model_names = ['LR','RF'] #add 'KNN'
for i,ax in enumerate(axs):
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    add_identity(ax, color="r", ls="--",label = 'random\nclassifier')
    ax.legend()
    ax.title.set_text(model_names[i])
plt.tight_layout()
plt.savefig('./output/roc_curves.png')


# Summarize the folds
print(data_encoded_performance.groupby(by = 'LR_clf').mean())
print(data_encoded_performance.groupby(by = 'LR_clf').std())


# Get the top features - evaluate the coefficients across the five folds
data_encoded_LR_normcoef['importance_mean'] = data_encoded_LR_normcoef.mean(axis =1)
data_encoded_LR_normcoef['importance_std']  = data_encoded_LR_normcoef.std(axis =1)
data_encoded_LR_normcoef['importance_abs_mean'] = data_encoded_LR_normcoef.abs().mean(axis =1)
data_encoded_LR_normcoef.sort_values('importance_abs_mean', inplace = True, ascending=False)

# Visualize the normalized feature importance across the five folds and add error bar to indicate the std
fig, ax = plt.subplots(figsize=(9, 6))
ax.bar(np.arange(15), data_encoded_LR_normcoef['importance_abs_mean'][:15], yerr=data_encoded_LR_normcoef['importance_std'][:15])
#ax.set_xticklabels(data_encoded_LR_normcoef.index.tolist()[:15], rotation=90)
ax.set_xticks(np.arange(15), data_encoded_LR_normcoef.index.tolist()[:15], rotation=90)
ax.set_title("Normalized feature importance for LR across 5 folds", fontsize=20)
plt.xlabel('Features', fontsize=16)
plt.ylabel("Normalized feature importance", fontsize=16)
plt.tight_layout()
plt.savefig('./output/importance.png')

# Get the two most important features and the relevant sign:
data_encoded_LR_normcoef.index[:2]
data_encoded_LR_normcoef['importance_mean'][:2]


print ("****************** Zusatz Annika ***************************")
# Get the coefficients for all features
coefficients = pd.DataFrame(zip(X_train.columns, np.transpose(LR_clf.coef_[0])), columns=['features', 'coef'])
coefficients = coefficients.sort_values(by='coef', ascending=False)

# Print all features and their coefficients
print(coefficients)
coefficients.to_csv('./output/table_with_coefficients.csv', index=False)
'''
positive Koeffizienten: höheres Risiko für Herzkrankheiten
- Alterskategorien: höchsten positiven Koeffizienten --> höheres Alter = erhöhtes Risiko für Herzkrankheiten verbunden (Koeffizienten nehmen mit steigendem Alter ab --> Risiko mit höherem Alter nimmt geringfügig ab)
- Allgemeine Gesundheit (GenHealth) --> Kategorien "Good" und "Fair" einen höheren Einfluss haben als "Poor" und "Very good" --> schlechtere allgemeine Gesundheit = höheres Risiko für Herzkrankheiten 
- Geschlecht (Sex_Male) --> Männer haben ein etwas höheres Risiko für Herzkrankheiten als Frauen
- Schlaganfall (Stroke_Yes), Rauchen (Smoking_Yes), Diabetes (Diabetic_Yes), Nierenerkrankungen (KidneyDisease_Yes) und Asthma (Asthma_Yes) --> erhöhtes Risiko für Herzkrankheiten 
- BMI, geistige Gesundheit (MentalHealth), körperliche Gesundheit (PhysicalHealth) geringere positive Koeffizienten --> geringeren Einfluss auf das Herzkrankheitsrisiko 
Negative Koeffizeinten: geringeren Risiko für Herzkrankheiten 
- Rassenzugehörigkeit (Race)  
- Hautkrebs (SkinCancer_Yes) 
'''
data_encoded_performance.to_csv('./output/table_with_performance.csv', index=False)
'''
- Accuracy: Prozentsatz der korrekt vorhergesagten Werte insgesamt --> LR-Classifier durchschnittliche Accuracy über die Folds: 0.916, RF-Classifier: 0.905 l --> LR-Classifier weist tendenziell eine etwas höhere Gesamtgenauigkeit auf
- Precision: Anteil der korrekt positiv vorhergesagten Werte (true positives) unter allen positiven Vorhersagen --> LR-Classifier durchschnittliche Precision: 0.544, RF-Classifier: 0.341 aufweist --> LR-Classifier tendenziell eine bessere Fähigkeit hat, echte positive Fälle zu identifizieren
- Recall (Sensitivität): Anteil der korrekt positiv vorhergesagten Werte (true positives) unter allen tatsächlich positiven Fällen --> LR-Classifier: 0.107, RF-Classifier: 0.120 --> RF-Classifier hat tendenziell eine etwas bessere Fähigkeit hat, tatsächlich positive Fälle zu erkennen
- Specificity: Anteil der korrekt negativ vorhergesagten Werte (true negatives) unter allen tatsächlich negativen Fällen an --> LR-Classifier: 0.992, RF-Classifier: 0.978 --> LR-Classifier hat eine höhere Fähigkeit, echte negative Fälle zu identifizieren
- F1-Score: harmonische Mittel aus Precision und Recall und gibt einen kombinierten Maßstab für die Modellleistung --> LR-Classifier durchschnittlicher F1-Score: 0.178, RF-Classifier: 0.178 --> ähnliche F1-Scores
- ROC-AUC: Fähigkeit des Modells beschreibt, zwischen positiven und negativen Fällen zu unterscheiden --> LR-Classifier durchschnittliche ROC-AUC: 0.840, RF-Classifier: 0.787 --> LR-Classifier tendenziell eine bessere Fähigkeit, zwischen positiven und negativen Fällen zu unterscheiden
'''







