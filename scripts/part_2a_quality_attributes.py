import numpy as np

def get_information_gain(contingency_table):
    """Calculates the IG.

    Args:
        contingency_table: A 2x2 contingency table, where the rows are the different
          attribute values and the columns are the different class labels.

    Returns:
        The information gain.
    """
    # Calculate the entropy of the target variable.
    entropy_target = _calculate_entropy(contingency_table[:, 1] / np.sum(contingency_table[:, 1]))

    # Calculate the weighted entropy of the target variable for each attribute value.
    weighted_entropy_attribute = []
    for i in range(contingency_table.shape[0]):
        probabilities = contingency_table[i, 1:] / np.sum(contingency_table[i, 1:])
        weighted_entropy_attribute.append(_calculate_entropy(probabilities) *
                                          contingency_table[i, 0] / np.sum(contingency_table[:, 0]))

    # Calculate the information gain.
    information_gain = entropy_target - np.sum(weighted_entropy_attribute)

    return information_gain

def get_gini_impurity(contingency_table):
    """Calculates the Gini impurity.

    Args:
        contingency_table: A 2x2 contingency table, where the rows are the different
          attribute values and the columns are the different class labels.

    Returns:
        The Gini impurity.
    """
    # Calculate the Gini impurity of the target variable.
    gini_impurity_target = _calculate_gini_impurity(contingency_table[:, 1] / np.sum(contingency_table[:, 1]))

    # Calculate the weighted Gini impurity of the target variable for each attribute
    # value.
    weighted_gini_impurity_attribute = []
    for i in range(contingency_table.shape[0]):
        probabilities = contingency_table[i, 1:] / np.sum(contingency_table[i, 1:])
        weighted_gini_impurity_attribute.append(
            _calculate_gini_impurity(probabilities) *
            contingency_table[i, 0] / np.sum(contingency_table[:, 0]))

    # Calculate the Gini impurity.
    gini_impurity = gini_impurity_target - np.sum(weighted_gini_impurity_attribute)

    return gini_impurity

def get_chi2(contingency_table):
    """Calculates the Chi-squared.

    Args:
        contingency_table: A 2x2 contingency table, where the rows are the different
          attribute values and the columns are the different class labels.

    Returns:
        The Chi-squared statistic.
    """
    # Calculate the expected values for each cell in the contingency table.
    expected_values = np.outer(
        np.sum(contingency_table, axis=1), np.sum(contingency_table, axis=0)) / np.sum(
            contingency_table)

    # Calculate the Chi-squared statistic.
    chi2 = np.sum((contingency_table - expected_values)**2 / expected_values)

    return chi2

def _calculate_entropy(probabilities):
    """Calculates the entropy.

    Args:
        probabilities: A NumPy array containing the probabilities of each class.

    Returns:
        The entropy.
    """
    entropy = np.sum(-probabilities * np.log2(probabilities + 1e-10))
    return entropy

def _calculate_gini_impurity(probabilities):
    """Calculates the Gini .

    Args:
        probabilities: A NumPy array containing the probabilities of each class.

    Returns:
        The Gini impurity.
    """
    gini_impurity = 1 - np.sum(probabilities**2)
    return gini_impurity

if __name__ == '__main__':
    # Create the contingency tables for the Headache, Spots, and Stiff-Neck attributes
    # in terms of Diagnosis using the dataset in Table 1.
    headache_contingency_table = np.array([[3, 1], [2, 4]])
    spots_contingency_table = np.array([[3, 2], [1, 4]])
    stiff_neck_contingency_table = np.array([[1, 3], [5, 2]])

    stiff_neck_information_gain = get_information_gain(stiff_neck_contingency_table)
    stiff_neck_gini_impurity = get_gini_impurity(stiff_neck_contingency_table)
    stiff_neck_chi2 = get_chi2(stiff_neck_contingency_table)

    headache_information_gain = get_information_gain(headache_contingency_table)
    headache_gini_impurity = get_gini_impurity(headache_contingency_table)
    headache_chi2 = get_chi2(headache_contingency_table)

    spots_information_gain = get_information_gain(spots_contingency_table)
    spots_gini_impurity = get_gini_impurity(spots_contingency_table)
    spots_chi2 = get_chi2(spots_contingency_table)

    # Print the results for each attribute.
    print("Stiff neck:")
    print("Contingency Table:", stiff_neck_contingency_table)
    print("Information Gain:", stiff_neck_information_gain)
    print("Gini Impurity:", stiff_neck_gini_impurity)
    print("Chi-squared statistic:", stiff_neck_chi2)

    print("\nHeadache:")
    print("Contingency Table:", headache_contingency_table)
    print("Information Gain:", headache_information_gain)
    print("Gini Impurity:", headache_gini_impurity)
    print("Chi-squared statistic:", headache_chi2)

    print("\nSpots:")
    print("\nContingency Table:", spots_contingency_table)
    print("Information Gain:", spots_information_gain)
    print("Gini Impurity:", spots_gini_impurity)
    print("Chi-squared statistic:", spots_chi2)
