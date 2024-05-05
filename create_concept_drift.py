import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import os

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def tokenize_dataset_enron(data):
    data['message'] = data['Subject'].fillna('') + " " + data['Message'].fillna('')
    data['message'] = data['message'].str.lower()
    data['message'] = data['message'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    data['message'] = data['message'].apply(word_tokenize)
    stop_words = set(stopwords.words('english'))
    data['message'] = data['message'].apply(lambda x: [word for word in x if word not in stop_words])
    return data

def mix_datasets(df1, df2, blend_ratio):
    n_samples = len(df1)
    n_samples_from_df2 = int(n_samples * blend_ratio)
    n_samples_from_df1 = n_samples - n_samples_from_df2
    sampled_df1 = df1.sample(n=n_samples_from_df1, random_state=42)
    sampled_df2 = df2.sample(n=n_samples_from_df2, random_state=42)
    mixed_df = pd.concat([sampled_df1, sampled_df2], ignore_index=True)
    return mixed_df

def prepare_mix_dataset(output_folder):
    data1 = pd.read_csv("enron_spam_data.csv")
    data2 = pd.read_csv("enron_spam_data_2.csv")

    data1_preprocessed = tokenize_dataset_enron(data1)
    data2_preprocessed = tokenize_dataset_enron(data2)

    blend_ratio = 0.3
    mixed_dataset = mix_datasets(data1_preprocessed, data2_preprocessed, blend_ratio)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    save_path = os.path.join(output_folder, "mixed_data.csv")
    mixed_dataset.to_csv(save_path, index=False)
    return save_path

