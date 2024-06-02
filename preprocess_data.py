import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


def tokenize_dataset_enron(data):
    data["message"] = data["Subject"].fillna("") + " " + data["Message"].fillna("")
    data["message"] = data["message"].str.lower()
    data["message"] = data["message"].apply(
        lambda x: x.translate(str.maketrans("", "", string.punctuation))
    )
    data["message"] = data["message"].apply(word_tokenize)
    stop_words = set(stopwords.words("english"))
    data["message"] = data["message"].apply(
        lambda x: [word for word in x if word not in stop_words]
    )
    return data


def tokenize_dataset_processed(data):
    data["message"] = data["subject"].fillna("") + " " + data["message"].fillna("")
    data["message"] = data["message"].str.lower()
    data["message"] = data["message"].apply(
        lambda x: x.translate(str.maketrans("", "", string.punctuation))
    )
    data["message"] = data["message"].apply(word_tokenize)
    stop_words = set(stopwords.words("english"))
    data["message"] = data["message"].apply(
        lambda x: [word for word in x if word not in stop_words]
    )
    return data


def bert_tokenize_dataset_enron(data):
    data["message"] = data["Subject"].fillna("") + " " + data["Message"].fillna("")
    data["message"] = data["message"].str.lower()
    data["message"] = data["message"].apply(
        lambda x: x.translate(str.maketrans("", "", string.punctuation))
    )
    return data


def bert_tokenize_dataset_processed(data):
    data["message"] = data["subject"].fillna("") + " " + data["message"].fillna("")
    data["message"] = data["message"].str.lower()
    data["message"] = data["message"].apply(
        lambda x: x.translate(str.maketrans("", "", string.punctuation))
    )
    return data


def prepare_data(data_path, drift_data_path):
    df = pd.read_csv(data_path)
    df = tokenize_dataset_enron(df)
    df["Spam/Ham"] = df["Spam/Ham"].apply(lambda x: 1 if x == "spam" else 0)
    df.rename(columns={"Spam/Ham": "label"}, inplace=True)
    df.drop(columns=["Subject", "Message", "Message ID", "Date"], inplace=True)
    print(df.head())

    df_after_drift = pd.read_csv(drift_data_path)
    df_after_drift = tokenize_dataset_processed(df_after_drift)
    df_after_drift.drop(columns=["subject", "email_to", "email_from"], inplace=True)
    print(df_after_drift.head())

    df.to_pickle("enron_spam_data.pkl")
    df_after_drift.to_pickle("processed_data.pkl")

    return df, df_after_drift


def bert_prepare_data(data_path, drift_data_path):
    df = pd.read_csv(data_path)
    df = bert_tokenize_dataset_enron(df)
    df["Spam/Ham"] = df["Spam/Ham"].apply(lambda x: 1 if x == "spam" else 0)
    df.rename(columns={"Spam/Ham": "label"}, inplace=True)
    df.drop(columns=["Subject", "Message", "Message ID", "Date"], inplace=True)
    print(df.head())

    df_after_drift = pd.read_csv(drift_data_path)
    df_after_drift = bert_tokenize_dataset_processed(df_after_drift)
    df_after_drift.drop(columns=["subject", "email_to", "email_from"], inplace=True)
    print(df_after_drift.head())

    df.to_pickle("bert_enron_spam_data.pkl")
    df_after_drift.to_pickle("bert_processed_data.pkl")

    return df, df_after_drift
