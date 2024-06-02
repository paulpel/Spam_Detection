import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_classifiers(data):
    """
    Train classifiers on the given dataset.

    This function splits the data into training and testing sets, vectorizes the text data,
    and trains Naive Bayes classifiers on the training data. It then evaluates the trained
    classifiers on the test data.

    :param data: The dataset containing messages and labels.
    :type data: pandas.DataFrame
    :return: A tuple containing the trained models, their scores, and the vectorizer.
    :rtype: tuple(dict, dict, CountVectorizer)
    """
    # Vectorizer for transforming text data to numerical data
    vectorizer = CountVectorizer(analyzer=lambda x: x)

    # Split the data into training and testing sets
    X = data["message"]
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit and transform the data using the vectorizer
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Initialize the classifiers
    classifiers = {
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
    }

    trained_models = {}
    scores = {}

    # Train each classifier and evaluate
    for name, clf in classifiers.items():
        clf.fit(X_train_vec, y_train)
        y_pred = clf.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        trained_models[name] = clf
        scores[name] = {"accuracy": acc, "report": report}
        print(f"Classifier: {name}")
        print(f"Accuracy: {acc}")
        print(f"Classification Report:\n{report}")

    return trained_models, scores, vectorizer


def evaluate_models(trained_models, vectorizer, data):
    """
    Evaluate trained classifiers on a new dataset.

    This function transforms the text data using the provided vectorizer and evaluates
    each trained classifier on the data.

    :param trained_models: A dictionary of trained models.
    :type trained_models: dict
    :param vectorizer: The vectorizer used to transform the text data.
    :type vectorizer: CountVectorizer
    :param data: The dataset containing messages and labels.
    :type data: pandas.DataFrame
    :return: A dictionary of scores for each classifier.
    :rtype: dict
    """
    # Extract messages and labels from data
    X = data["message"]
    y = data["label"]

    # Transform the data using the provided vectorizer
    X_vec = vectorizer.transform(X)

    scores = {}

    # Evaluate each trained model on the data
    for name, model in trained_models.items():
        y_pred = model.predict(X_vec)
        acc = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred)
        scores[name] = {"accuracy": acc, "report": report}
        print(f"Classifier: {name}")
        print(f"Accuracy: {acc}")
        print(f"Classification Report:\n{report}")

    return scores


def extract_features_with_bert(
    data, model_name="bert-base-uncased", max_length=128, batch_size=32
):
    """
    Extract features from text data using a pre-trained BERT model.

    This function tokenizes and encodes the text data using a BERT tokenizer, processes it in batches,
    and extracts features using the BERT model.

    :param data: The dataset containing messages.
    :type data: pandas.DataFrame
    :param model_name: The name of the pre-trained BERT model to use.
    :type model_name: str
    :param max_length: The maximum length of the tokenized sequences.
    :type max_length: int
    :param batch_size: The batch size for processing the data.
    :type batch_size: int
    :return: A numpy array containing the extracted features.
    :rtype: np.ndarray
    """
    # Check if GPU is available and set the device accordingly
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load pre-trained BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)

    # Tokenize and encode the messages
    encoded_inputs = tokenizer(
        data["message"].tolist(),  # Directly use the list of strings
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # Create a DataLoader for the encoded inputs
    dataset = torch.utils.data.TensorDataset(
        encoded_inputs["input_ids"], encoded_inputs["attention_mask"]
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # Extract features
    features = []
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting BERT features"):
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            # Use the CLS token embedding as the sentence embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            features.append(cls_embeddings)

    # Concatenate all features into a single array
    features = np.concatenate(features, axis=0)

    return features


def train_classifiers_with_bert_features(data, features):
    """
    Train classifiers on BERT-extracted features.

    This function splits the BERT features and labels into training and testing sets, trains
    several classifiers on the training data, and evaluates them on the test data.

    :param data: The dataset containing labels.
    :type data: pandas.DataFrame
    :param features: The BERT-extracted features.
    :type features: np.ndarray
    :return: A tuple containing the trained models and their scores.
    :rtype: tuple(dict, dict)
    """
    labels = data["label"].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    # Initialize the classifiers
    classifiers = {
        "Naive Bayes": MultinomialNB(),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
    }

    trained_models = {}
    scores = {}

    # Train each classifier and evaluate
    for name, clf in classifiers.items():
        print(f"Training {name} classifier...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        trained_models[name] = clf
        scores[name] = {"accuracy": acc, "report": report}
        print(f"Classifier: {name}")
        print(f"Accuracy: {acc}")
        print(f"Classification Report:\n{report}")
    return trained_models, scores


def bert_evaluate_models(trained_models, features, data):
    """
    Evaluate trained classifiers on BERT-extracted features.

    This function evaluates each trained classifier on the provided BERT features and labels.

    :param trained_models: A dictionary of trained models.
    :type trained_models: dict
    :param features: The BERT-extracted features.
    :type features: np.ndarray
    :param data: The dataset containing labels.
    :type data: pandas.DataFrame
    :return: A dictionary of scores for each classifier.
    :rtype: dict
    """
    labels = data["label"].values

    scores = {}

    # Evaluate each trained model on the data
    for name, model in trained_models.items():
        y_pred = model.predict(features)
        acc = accuracy_score(labels, y_pred)
        report = classification_report(labels, y_pred)
        scores[name] = {"accuracy": acc, "report": report}
        print(f"Classifier: {name}")
        print(f"Accuracy: {acc}")
        print(f"Classification Report:\n{report}")
    return scores
