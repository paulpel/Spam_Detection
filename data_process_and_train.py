import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_classifiers(data):
    # Vectorizer for transforming text data to numerical data
    vectorizer = CountVectorizer(analyzer=lambda x: x)
    
    # Split the data into training and testing sets
    X = data['message']
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit and transform the data using the vectorizer
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Initialize the classifiers
    classifiers = {
        "Naive Bayes": MultinomialNB(),
        # "KNN": KNeighborsClassifier(),
        # "Random Forest": RandomForestClassifier(),
        # "Gradient Boosting": GradientBoostingClassifier()
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
    # Extract messages and labels from data
    X = data['message']
    y = data['label']
    
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


def extract_features_with_bert(data, model_name='bert-base-uncased', max_length=128, batch_size=32):
    # Check if GPU is available and set the device accordingly
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load pre-trained BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)
    
    # Tokenize and encode the messages
    encoded_inputs = tokenizer(
        data['message'].tolist(),  # Directly use the list of strings
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

    # Create a DataLoader for the encoded inputs
    dataset = torch.utils.data.TensorDataset(
        encoded_inputs['input_ids'],
        encoded_inputs['attention_mask']
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
    labels = data['label'].values
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Initialize the classifiers
    classifiers = {
        "Naive Bayes": MultinomialNB(),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
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
    labels = data['label'].values
    
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

