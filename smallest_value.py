import numpy as np

def extract_smallest_value_from_csv(file_path):
    """
    Extracts the smallest value from a CSV file containing BERT features.

    :param file_path: Path to the CSV file.
    :type file_path: str
    :return: The smallest value in the CSV file.
    :rtype: float
    """
    # Load the CSV file into a numpy array
    data = np.loadtxt(file_path, delimiter=',')
    
    # Find the smallest value in the array
    smallest_value = np.min(data)
    
    return smallest_value

# Paths to the CSV files
regular_features_path = 'bert_encoded_features_original.csv'
drifted_features_path = 'bert_encoded_features_drifted.csv'

# Extract the smallest values from the CSV files
smallest_value_regular = extract_smallest_value_from_csv(regular_features_path)
smallest_value_drifted = extract_smallest_value_from_csv(drifted_features_path)

print(f"Smallest value in regular features: {smallest_value_regular}")
print(f"Smallest value in drifted features: {smallest_value_drifted}")