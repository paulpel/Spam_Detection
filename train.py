import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


class DataPreprocessor:
    def __init__(self, filepath, is_bert=False):
        self.data = pd.read_csv(filepath)
        self.is_bert = is_bert
        self.label_encoder = LabelEncoder() if not is_bert else None

    def preprocess(self):
        if not self.is_bert:
            self.data["Messages"] = (
                self.data["Subject"].fillna("") + " " + self.data["Message"].fillna("")
            )
            self.data.drop(
                columns=["Subject", "Message", "Message ID", "Date"], inplace=True
            )
            self.data["Messages"] = (
                self.data["Messages"]
                .str.lower()
                .replace([":", ",", ".", "-"], " ", regex=True)
            )
            self.data["Spam/Ham"] = self.label_encoder.fit_transform(
                self.data["Spam/Ham"]
            )
        else:
            # If BERT features are used, we expect the data to have two columns: features and labels
            self.data.rename(columns={"label": "Spam/Ham"}, inplace=True)
        return self.data


class ModelTrainer:
    def __init__(self, data, is_bert=False):
        self.data = data
        self.is_bert = is_bert
        self.vectorizer = None if is_bert else TfidfVectorizer()
        self.models = {
            "Naive Bayes": MultinomialNB(),
            "KNN": KNeighborsClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
        }
        self.results = {}

    def train_and_evaluate(self):
        if not self.is_bert:
            X_train, X_test, y_train, y_test = train_test_split(
                self.data["Messages"], self.data["Spam/Ham"], test_size=0.5
            )
            count_train = self.vectorizer.fit_transform(X_train)
            count_test = self.vectorizer.transform(X_test)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                self.data.drop("Spam/Ham", axis=1), self.data["Spam/Ham"], test_size=0.5
            )
            count_train = X_train
            count_test = X_test

        rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=2652124)
        for name, model in self.models.items():
            model.fit(count_train, y_train)
            scores = cross_val_score(
                model, count_train, y_train, scoring="accuracy", cv=rkf, n_jobs=-1
            )
            y_pred = model.predict(count_test)
            self.results[name] = {
                "accuracy_mean": np.mean(scores),
                "accuracy_std": np.std(scores),
                "confusion_matrix": confusion_matrix(y_test, y_pred),
                "classification_report": classification_report(
                    y_test, y_pred, output_dict=True
                ),
            }

    def get_results(self):
        return self.results


class ResultsVisualizer:
    def __init__(self, results):
        self.results = results

    def display_results(self):
        for name, result in self.results.items():
            print(
                f"{name} Model Accuracy: {result['accuracy_mean']:.3f} (+/- {result['accuracy_std']:.3f})"
            )
            sns.heatmap(result["confusion_matrix"], annot=True, fmt="d")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"{name} Confusion Matrix")
            plt.show()
            print(f"{name} Classification Report:")
            print(
                classification_report(
                    None, None, output_dict=result["classification_report"]
                )
            )


def main(filepath, is_bert):
    data_preprocessor = DataPreprocessor(filepath, is_bert)
    data = data_preprocessor.preprocess()
    model_trainer = ModelTrainer(data, is_bert)
    model_trainer.train_and_evaluate()
    results_visualizer = ResultsVisualizer(model_trainer.get_results())
    results_visualizer.display_results()
