import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Set the random seed as my Last 2 Student ID Digits
random_seed = 96

# Specify the column names based on the dataset documentation
column_names = [
    'Class', 'Age', 'Menopause', 'Tumor_Size', 'Node_Caps', 'Degree_Malign', 'Breast', 'Breast_Quad', 'Irradiat'
]

# Load the dataset
data = pd.read_csv('../data/breast-cancer.data', header=None, names=column_names, na_values="?")

# Display the number of cases and attributes before processing
num_cases_before = len(data)
num_attributes = len(data.columns)

# Remove cases with invalid values
data = data.dropna()

# Display the number of cases and attributes after processing
num_cases_after = len(data)
print(f"Number of cases before processing: {num_cases_before}")
print(f"Number of attributes in the dataset: {num_attributes}")
print(f"Number of cases after processing: {num_cases_after}")

# Encode target/class values
label_encoder = LabelEncoder()
data['Class'] = label_encoder.fit_transform(data['Class'])

# Remove the least populated class
class_counts = data['Class'].value_counts()
min_class_count = class_counts.min()
min_class = class_counts[class_counts == min_class_count].index[0]
data = data[data['Class'] != min_class]

# Display the unique encoded values in the 'Class' column
print("Encoded values in 'Class' column:", data['Class'].unique())

# Perform train-test split with stratification
X = data.drop('Class', axis=1)  # Features
y = data['Class']  # Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_seed, stratify=y
)

# One-hot encode the features
encoder = OneHotEncoder()
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)

# Configure Decision Tree classifiers for the ensemble
dt1 = DecisionTreeClassifier(criterion="entropy", max_depth=1, max_features=10, random_state=random_seed)
dt2 = DecisionTreeClassifier(criterion="entropy", max_depth=3, max_features=10, random_state=random_seed)
dt3 = DecisionTreeClassifier(criterion="entropy", max_depth=10, max_features=10, random_state=random_seed)

# Train the classifiers on the training set
dt1.fit(X_train_encoded, y_train)
dt2.fit(X_train_encoded, y_train)
dt3.fit(X_train_encoded, y_train)

# Make predictions on the testing set with each classifier
y_pred_dt1 = dt1.predict(X_test_encoded)
y_pred_dt2 = dt2.predict(X_test_encoded)
y_pred_dt3 = dt3.predict(X_test_encoded)

# Function to perform majority voting
def majority_vote(predictions):
    return np.argmax(np.bincount(predictions))

# Initialize an array to store ensemble predictions
ensemble_predictions = np.zeros_like(y_pred_dt1)

# Iterate through each testing case and perform majority voting
for i in range(len(X_test)):
    predictions = [y_pred_dt1[i], y_pred_dt2[i], y_pred_dt3[i]]
    ensemble_predictions[i] = majority_vote(predictions)

# Create a DataFrame to store information for each testing case
ensemble_results = pd.DataFrame({
    'case_id': range(1, len(X_test) + 1),
    'actual_class_label': y_test.values,
    'DT1_predict': y_pred_dt1,
    'DT2_predict': y_pred_dt2,
    'DT3_predict': y_pred_dt3,
    'ensemble_predict': ensemble_predictions
})

# Print to console and write to CSV file
print(ensemble_results)

ensemble_results.to_csv('part_2c_ensemble_out.csv', index=False)

# Compute and print the accuracy score of the ensemble
ensemble_accuracy = np.sum(ensemble_predictions == y_test.values) / len(y_test)
print(f"Ensemble Accuracy: {ensemble_accuracy}")
