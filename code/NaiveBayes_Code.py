import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

# Load and Prepare the Data
data = pd.read_csv("../data/heart_2020_cleaned.csv")

# Drop rows with missing values
data.dropna(inplace=True)

# Convert "yes" and "no" to binary values
data.replace({"Yes": 1, "No": 0}, inplace=True)

# Convert "Female" and "Male" to binary values
data.replace({"Female": 0, "Male": 1}, inplace=True)

# Select the desired features
# Feature Selection use: 'Smoking', 'Stroke', 'Diabetic', 'PhysicalActivity', 'KidneyDisease', 'SkinCancer','Sex'
# No Feature Selection use: 'Smoking', 'AlcoholDrinking','Stroke', 'DiffWalking', 'Diabetic', 'PhysicalActivity', 'Asthma','KidneyDisease', 'SkinCancer'
selected_features = ['Smoking', 'Stroke', 'Diabetic', 'PhysicalActivity', 'KidneyDisease', 'SkinCancer','Sex']
X = data[selected_features]
y = data['HeartDisease']

# Convert the categorical columns to strings
X = X.astype(str)

# Perform one-hot encoding for categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(drop='first', sparse=False)
X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]))

# Concatenate the encoded features with the numeric features
X = pd.concat([X.drop(categorical_cols, axis=1), X_encoded], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform undersampling to balance the classes
indices_class_0 = y_train[y_train == 0].index
indices_class_1 = y_train[y_train == 1].index

undersampled_indices_class_0 = np.random.choice(indices_class_0, size=sum(y_train == 1), replace=False)
undersampled_indices = np.concatenate((undersampled_indices_class_0, indices_class_1))

X_train_undersampled = X_train.loc[undersampled_indices]
y_train_undersampled = y_train.loc[undersampled_indices]

# Handle missing values using imputation
imputer = SimpleImputer(strategy='most_frequent')
X_train_imputed = imputer.fit_transform(X_train_undersampled)
X_test_imputed = imputer.transform(X_test)

# Train the Naive Bayes model
naive_bayes = GaussianNB()
naive_bayes.fit(X_train_imputed, y_train_undersampled)

# Make predictions on the test set
y_pred = naive_bayes.predict(X_test_imputed)

# Obtain predicted probabilities for the positive class
predicted_probabilities = naive_bayes.predict_proba(X_test_imputed)[:, 1]

# Calculate evaluation metrics
classification_report_result = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, predicted_probabilities)

# Print the evaluation metrics
print("Classification Report:")
print(classification_report_result)
print("ROC AUC Score:", roc_auc)

# Calculate specificity
tn = sum((y_test == 0) & (y_pred == 0))  # True negatives
fp = sum((y_test == 0) & (y_pred == 1))  # False positives
specificity = tn / (tn + fp)
print("Specificity:", specificity)

# Plot the ROC curve
# Perform stratified k-fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store the true positive rates (TPR) and false positive rates (FPR) for each fold
tpr_list = []
fpr_list = []

# Initialize list to store the AUC scores for each fold
auc_scores = []

# Plot ROC curve for each fold
for i, (train_index, test_index) in enumerate(kfold.split(X_train, y_train)):
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    # Train the Naive Bayes model
    naive_bayes_fold = GaussianNB()
    naive_bayes_fold.fit(X_train_fold, y_train_fold)

    # Obtain predicted probabilities for the positive class
    predicted_probabilities_fold = naive_bayes_fold.predict_proba(X_test_fold)[:, 1]

    # Calculate the false positive rate, true positive rate, and thresholds
    fpr, tpr, thresholds = roc_curve(y_test_fold, predicted_probabilities_fold)

    # Calculate the area under the ROC curve
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.plot(fpr, tpr, label='Fold {} (AUC = {:.2f})'.format(i, roc_auc), linewidth=1)

# Plot the random classifier (dotted red line)
plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier', linewidth=1)

# Set plot title and labels
plt.title('Receiver Operating Characteristic (ROC) Curve for Naive Bayes')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# Add legend with different colors for each fold
plt.legend()

# Save the plot as a PDF
#change the name of the jpg depending on featute selection (roc_auc_plot_Featureselection.jpg vs. roc_auc_plot_NoFeatureselection.jpg)
plt.savefig('C:/Users/ginah/Desktop/roc_auc_plot_Featureselection.jpg', format='jpg')

# Show the plot
plt.show()
