import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import string

def tokenize_dataset_enron(data, save_file=True):

    # Combine 'Subject' and 'Message' into one 'message' column
    data['message'] = data['Subject'].fillna('') + " " + data['Message'].fillna('')
    # Convert text to lowercase to maintain consistency
    data['message'] = data['message'].str.lower()
    # Remove punctuation
    data['message'] = data['message'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    # Tokenize messages
    data['message'] = data['message'].apply(word_tokenize)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))

    data['message'] = data['message'].apply(lambda x: [word for word in x if word not in stop_words])

    if save_file:
        data[['message', 'Spam/Ham']].to_csv("tokenized_data.csv", index=False)
    
    return data

def tokenize_dataset_other(data, save_file=False):

    # Combine 'Subject' and 'Message' into one 'message' column
    data['message'] = data['Subject'].fillna('') + " " + data['Message'].fillna('')
    # Convert text to lowercase to maintain consistency
    data['message'] = data['message'].str.lower()
    # Remove punctuation
    data['message'] = data['message'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    # Tokenize messages
    data['message'] = data['message'].apply(word_tokenize)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    data['message'] = data['message'].apply(lambda x: [word for word in x if word not in stop_words])

    if save_file:
        data[['message', 'Spam/Ham']].to_csv("tokenized_data.csv", index=False)
    
    return data


# Needed for creating dataset after concept drift
def mix_datasets(df1, df2, blend_ratio):
    n_samples = len(df1)
    n_samples_from_df2 = int(n_samples * blend_ratio)
    n_samples_from_df1 = n_samples - n_samples_from_df2
    
    sampled_df1 = df1.sample(n=n_samples_from_df1, random_state=42)
    sampled_df2 = df2.sample(n=n_samples_from_df2, random_state=42)
    
    mixed_df = pd.concat([sampled_df1, sampled_df2], ignore_index=True)
    return mixed_df

# Example usage
data1 = pd.read_csv("enron_spam_data.csv")
data2 = pd.read_csv("enron_spam_data.csv")

# Preprocess the datasets
data1_preprocessed = tokenize_dataset_enron(data1)
data2_preprocessed = tokenize_dataset_other(data2)

blend_ratio = 0.3

mixed_dataset = mix_datasets(data1_preprocessed, data2_preprocessed, blend_ratio)
original_dataset = data1_preprocessed.copy()

print("Original Dataset:")
print(original_dataset)
print("Mixed Dataset:")
print(mixed_dataset)
