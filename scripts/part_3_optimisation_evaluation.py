from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set the random seed as my Last 2 Student ID Digits
random_seed = 96

# Load the MNIST dataset
digits = load_digits()
X, y = digits.data, digits.target

# Split the data into training_validation (85%) and testing (15%) sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=random_seed, stratify=y
)

# Define the number of folds for cross-validation
num_folds = 10

# Define the cross-validation strategy
cv_strategy = StratifiedKFold(n_splits=num_folds, shuffle=False)  # Set shuffle to False

# Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=random_seed)
rf_scores = cross_val_score(rf_classifier, X_train_val, y_train_val, cv=cv_strategy, scoring='accuracy')
rf_mean_accuracy = np.mean(rf_scores)

# K-Nearest Neighbors Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_scores = cross_val_score(knn_classifier, X_train_val, y_train_val, cv=cv_strategy, scoring='accuracy')
knn_mean_accuracy = np.mean(knn_scores)

# Support Vector Machine Classifier
svm_classifier = SVC(kernel='rbf', C=1, gamma='scale', random_state=random_seed)
svm_scores = cross_val_score(svm_classifier, X_train_val, y_train_val, cv=cv_strategy, scoring='accuracy')
svm_mean_accuracy = np.mean(svm_scores)

# Print mean cross-validated accuracies directly to console
# Random Forest
print("Random Forest Classifier:")
print(f"Mean Cross-validated Accuracy: {rf_mean_accuracy}")
rf_classifier.fit(X_train_val, y_train_val)
rf_test_predictions = rf_classifier.predict(X_test)
rf_test_accuracy = accuracy_score(y_test, rf_test_predictions)
print(f"Testing Accuracy: {rf_test_accuracy}\n")

# K-Nearest Neighbors
print("K-Nearest Neighbors Classifier:")
print(f"Mean Cross-validated Accuracy: {knn_mean_accuracy}")
knn_classifier.fit(X_train_val, y_train_val)
knn_test_predictions = knn_classifier.predict(X_test)
knn_test_accuracy = accuracy_score(y_test, knn_test_predictions)
print(f"Testing Accuracy: {knn_test_accuracy}\n")

# Support Vector Machine
print("Support Vector Machine Classifier:")
print(f"Mean Cross-validated Accuracy: {svm_mean_accuracy}")
svm_classifier.fit(X_train_val, y_train_val)
svm_test_predictions = svm_classifier.predict(X_test)
svm_test_accuracy = accuracy_score(y_test, svm_test_predictions)
print(f"Testing Accuracy: {svm_test_accuracy}\n")

# Table for Cross-Validation Results
cv_results = pd.DataFrame({
    'Classifier': ['Random Forest', 'K-Nearest Neighbors', 'Support Vector Machine'],
    'Mean Cross-Validation Accuracy': [rf_mean_accuracy, knn_mean_accuracy, svm_mean_accuracy],
    'Testing Accuracy': [rf_test_accuracy, knn_test_accuracy, svm_test_accuracy]
})

print("Cross-Validation Results:")
print(cv_results)

# Bar Plot for Cross-Validation Results
plt.figure(figsize=(8, 6))
plt.bar(cv_results['Classifier'], cv_results['Mean Cross-Validation Accuracy'], label='Mean Cross-Validation Accuracy')
plt.bar(cv_results['Classifier'], cv_results['Testing Accuracy'], label='Testing Accuracy', alpha=0.5)
plt.title("Cross-Validation and Testing Results - Accuracy")
plt.ylim(0, 1)
plt.legend()
plt.show()

# Testing Performance
# Random Forest
rf_conf_matrix = confusion_matrix(y_test, rf_test_predictions)
print("Testing Performance - Random Forest Classifier:")
print(f"Testing Accuracy: {rf_test_accuracy}")
print("Confusion Matrix:")
print(rf_conf_matrix)

# Visualise Confusion Matrix for Random Forest
plt.figure(figsize=(8, 6))
plt.imshow(rf_conf_matrix, cmap='Blues', interpolation='nearest')
plt.title("Confusion Matrix - Random Forest Classifier (Testing)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.colorbar()
plt.show()
