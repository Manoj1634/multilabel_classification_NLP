import os
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.metrics import f1_score
import argparse
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Parameters
nickname = 'mj'
max_length = 128
batch_size = 16
num_labels = 6
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Text Preprocessing Functions
def expand_contractions(text, contractions_dict):
    for contraction, expansion in contractions_dict.items():
        text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text)
    return text

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    return ' '.join([word for word in words if word not in stop_words])

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    return ' '.join([lemmatizer.lemmatize(word) for word in words])

def clean_row(text):
    contractions_dict = {
        "don't": "do not", "it's": "it is", "how's": "how is", "i'm": "i am", "you're": "you are",
        "we're": "we are", "they're": "they are", "won't": "will not", "can't": "cannot",
        "didn't": "did not", "hasn't": "has not", "haven't": "have not", "isn't": "is not",
        "aren't": "are not", "wasn't": "was not", "weren't": "were not", "shouldn't": "should not",
        "couldn't": "could not", "wouldn't": "would not", "let's": "let us", "she's": "she is",
        "he's": "he is", "that's": "that is", "there's": "there is"
    }
    text = text.lower()
    text = expand_contractions(text, contractions_dict)
    text = preprocess_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return re.sub(r'\s+', ' ', text).strip()

# Dataset
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label)
        }

# Model
class DistilBERTForSequenceClassification(nn.Module):
    def __init__(self, pretrained_model_name, num_labels, dropout_prob=0.2):
        super().__init__()
        config = AutoConfig.from_pretrained(pretrained_model_name)
        config.hidden_dropout_prob = dropout_prob
        self.distilbert = AutoModel.from_pretrained(pretrained_model_name, config=config)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_output))

# Evaluation
def eval_model(model, test_loader, criterion):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        total_val_loss = 0
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            total_val_loss += loss.item()

            predictions = torch.sigmoid(logits) >= 0.5
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(test_loader)
        print(f"Test Loss: {avg_val_loss:.4f}")
    return np.vstack(all_targets), np.vstack(all_preds)

def metrics(all_targets, all_preds):
    return f1_score(all_targets, all_preds, average='macro')

# Main Inference
def main(path, train_df, model_path, split):
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    df = pd.read_excel(path + train_df, engine='openpyxl')
    df['text'] = df['text'].apply(clean_row)

    labels_list = ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']
    test_df = df[df['split'] == split]
    texts = test_df['text'].tolist()
    labels = test_df[labels_list].values

    test_dataset = CustomDataset(texts, labels, tokenizer, max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = DistilBERTForSequenceClassification("distilbert-base-uncased", num_labels=num_labels).to(device)
    model.load_state_dict(torch.load(f'{model_path}/{nickname}_model-weights.pt'))
    criterion = nn.BCEWithLogitsLoss()

    all_targets, all_preds = eval_model(model, test_loader, criterion)
    np.save(f'{model_path}/{nickname}-{split}_predictions.npy', all_preds)
    f1 = metrics(all_targets, all_preds)
    print(f"F1 Score: {f1:.4f}")

# Entry Point
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data file')
    parser.add_argument('--train_df', type=str, required=True, help='Excel file name')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--split', type=str, default='test', help='Split to evaluate (default: test)')
    args = parser.parse_args()

    main(args.data_path, args.train_df, args.model_path, args.split)
