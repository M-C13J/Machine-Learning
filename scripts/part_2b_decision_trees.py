import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt

# Set the random seed as my Last 2 Student ID Digits
random_seed = 96

# Specify the column names based on the dataset documentation
column_names = [
    'Class', 'Age', 'Menopause', 'Tumor_Size', 'Node_Caps', 'Degree_Malign', 'Breast', 'Breast_Quad', 'Irradiat'
]

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

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=y
    )
    print(f"Number of cases in training set: {len(X_train)}")
    print(f"Number of cases in testing set: {len(X_test)}")

    # One-hot encode the features
    encoder = OneHotEncoder()
    X_train_encoded = encoder.fit_transform(X_train)
    X_test_encoded = encoder.transform(X_test)

    # Train and assess decision tree classifiers with varying max_depth
    max_depths = list(range(1, 11))
    results = []

    for depth in max_depths:
        # Configure Decision Tree classifier
        dt_classifier = DecisionTreeClassifier(criterion="entropy", max_depth=depth)

        # Train the classifier
        dt_classifier.fit(X_train_encoded, y_train)

        # Evaluate the classifier on the testing set
        accuracy = dt_classifier.score(X_test_encoded, y_test)
        balanced_accuracy = balanced_accuracy_score(y_test, dt_classifier.predict(X_test_encoded))

        # Print and store results
        result = {
            'Model_id': depth,
            'Max_depth': depth,
            'Accuracy': accuracy,
            'Balanced_Accuracy': balanced_accuracy
        }
        results.append(result)

        print(f"Model id: {depth}, max depth: {depth}, accuracy: {accuracy}, balanced accuracy: {balanced_accuracy}")

    # Save results to a text file
    with open('part_2b_decision_trees_max_depths_out.txt', 'w') as file:
        file.write("Model id [1…10], max depth: [value], accuracy: [score], balanced accuracy: [score]\n")
        for result in results:
            file.write(f"Model id: {result['Model_id']}, max depth: {result['Max_depth']}, "
                       f"accuracy: {result['Accuracy']}, balanced accuracy: {result['Balanced_Accuracy']}\n")

    # Generate a plot
    plt.figure(figsize=(10, 6))
    plt.plot(max_depths, [result['Accuracy'] for result in results], label='Accuracy', marker='o')
    plt.plot(max_depths, [result['Balanced_Accuracy'] for result in results], label='Balanced Accuracy', marker='o')
    plt.title('Decision Tree Classifier Performance vs. Max Depth')
    plt.xlabel('Max Depth')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('../output/part_2b_decision_trees_max_depths_plot.png')
    plt.show()

except ValueError as e:
    print(f"Error: {e}")
