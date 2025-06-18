import os

# List of packages to install
packages = [
    "numpy",
    "pandas",
    "scikit-learn",
    "tqdm",
    "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
    "transformers",
    "openpyxl",
    "torch",
    "nltk"
]

for package in packages:
    os.system(f"pip3 install {package}")

import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

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
    # First handle contractions
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
    
    # Then apply the NLTK-based preprocessing
    text = preprocess_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    
    # Final cleanup
    return re.sub(r'\s+', ' ', text).strip()

# Parameters
nickname = 'mj'
file_path = '../Data/train.xlsx'
batch_size = 16
num_epochs = 10
learning_rate = 5e-5
num_labels = 6
max_length = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
df = pd.read_excel(file_path, engine='openpyxl')
df['text'] = df['text'].apply(clean_row)

labels_list = ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']
labels = df[labels_list].values

# Compute pos_weight for imbalance handling
label_counts = labels.sum(axis=0)
total_samples = labels.shape[0]
pos_weight = torch.tensor((total_samples - label_counts) / (label_counts + 1e-6), dtype=torch.float).to(device)

# Optional: Print stats
print("\nLabel Distribution and pos_weights:")
for i, label in enumerate(labels_list):
    print(f"{label:25s} | Positives: {int(label_counts[i])} | pos_weight: {pos_weight[i]:.2f}")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

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

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].values, labels, test_size=0.2, random_state=42
)

train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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

model = DistilBERTForSequenceClassification("distilbert-base-uncased", num_labels=num_labels).to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
def train_epoch(model, data_loader, criterion, optimizer, scheduler, device):
    model.train()
    losses = []
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    return np.mean(losses)

# Evaluation loop
def eval_model(model, data_loader, criterion, device):
    model.eval()
    losses = []
    preds = []
    true_labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            losses.append(loss.item())
            preds.append(torch.sigmoid(logits).cpu().numpy())
            true_labels.append(labels.cpu().numpy())

    preds = np.vstack(preds)
    true_labels = np.vstack(true_labels)
    f1_macro = f1_score(true_labels, preds >= 0.5, average='macro')
    return np.mean(losses), f1_macro

# Training
best_val_loss = float('inf')
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
    val_loss, val_f1_macro = eval_model(model, val_loader, criterion, device)
    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, F1 Macro: {val_f1_macro:.4f}")

    if val_loss < best_val_loss:
        print(f"Saving model with improved validation loss: {val_loss:.4f}")
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"{nickname}_model-weights.pt")
