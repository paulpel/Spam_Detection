import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Configuration for BERT processing
class Config:
    MAX_LEN = 128
    BATCH_SIZE = 16
    BERT_MODEL = "bert-base-uncased"


# Custom dataset for loading text
class TextDataset(Dataset):
    def __init__(self, messages, tokenizer, max_len):
        self.messages = messages
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, item):
        message = str(self.messages[item])
        encoding = self.tokenizer.encode_plus(
            message,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }


# Load data
def load_data(filepath):
    data = pd.read_csv(filepath)
    data["Messages"] = data["Subject"].fillna("") + " " + data["Message"].fillna("")
    data.drop(["Subject", "Message", "Message ID", "Date"], axis=1, inplace=True)
    label_encoder = LabelEncoder()
    data["Spam/Ham"] = label_encoder.fit_transform(data["Spam/Ham"])
    return data


# Create data loader for processing
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = TextDataset(
        messages=df.Messages.to_numpy(), tokenizer=tokenizer, max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=4)


# Feature extraction function
def extract_features(data_loader, model, device):
    model = model.eval()
    features = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            features.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
    return np.concatenate(features)


# Function to process and extract features that can be imported and reused
def process_and_extract_features(data_filepath, output_filepath):
    config = Config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL)
    model = BertModel.from_pretrained(config.BERT_MODEL)
    model.to(device)

    data = load_data(data_filepath)
    data_loader = create_data_loader(data, tokenizer, config.MAX_LEN, config.BERT_SIZE)
    features = extract_features(data_loader, model, device)

    # Combine features with labels and save
    features_df = pd.DataFrame(features)
    features_df["label"] = data["Spam/Ham"].values
    features_df.to_csv(output_filepath, index=False)


# process_and_extract_features("/path/to/input.csv", "/path/to/output.csv")
