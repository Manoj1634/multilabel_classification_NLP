# multilabel_classification_NLP

# Multi-Label Text Classification with DistilBERT

This project aims to perform multi-label classification abstracts into six scientific domains using DistilBERT. Each paper can belong to multiple categories, and the model is trained to predict all applicable labels.

---

## 📌 Problem Statement

Given a dataset of academic abstracts with labels indicating the fields they belong to (e.g., Computer Science, Physics), the goal is to build a deep learning model that can assign **multiple relevant labels** to each abstract.

---

## 🧩 Dataset Overview

The dataset consists of three main columns:

- `text`: the abstract content
- six binary label columns for each category:
  - `Computer Science`
  - `Physics`
  - `Mathematics`
  - `Statistics`
  - `Quantitative Biology`
  - `Quantitative Finance`
- `split`: indicating whether the row belongs to the train or test set

---

## ⚙️ Preprocessing Pipeline

We perform the following steps to clean and prepare text:

1. **Expand contractions** (e.g., "don't" → "do not")
2. **Lowercase text**
3. **Remove punctuation and non-letter characters**
4. **Remove English stopwords** using NLTK
5. **Lemmatize words** to get base forms

---

## 🔍 Tokenization

We use the `bert-base-uncased` tokenizer from HuggingFace Transformers to tokenize the cleaned abstracts, truncating/padding to a max length of 128.

---

## 🧠 Model Architecture

- **Base model**: DistilBERT (a lighter version of BERT)
- **Classification Head**: A linear layer with output dimension equal to the number of labels (6)
- **Loss Function**: `BCEWithLogitsLoss` with `pos_weight` for handling class imbalance
- **Optimizer**: AdamW
- **Scheduler**: Linear warmup and decay using `get_linear_schedule_with_warmup`

---

## ⚖️ Handling Class Imbalance

The dataset is highly imbalanced across labels:

| Label                  | Positives | Pos_Weight |
|------------------------|-----------|------------|
| Computer Science       | 7295      | 1.44       |
| Physics                | 5101      | 2.49       |
| Mathematics            | 4764      | 2.74       |
| Statistics             | 4413      | 3.04       |
| Quantitative Biology   | 493       | 35.16      |
| Quantitative Finance   | 220       | 80.03      |

To mitigate this, we compute the `pos_weight` for each label as:

pos_weight = (total_samples - positives) / (positives + 1e-6)


This ensures that **rare classes** (e.g., Quantitative Finance) are not ignored by the model.

---

## 🚀 Training Configuration

- **Epochs**: 10
- **Batch size**: 16
- **Learning rate**: 5e-5
- **Train/Validation Split**: 80/20

---

## 📈 Evaluation Metrics

After training, we obtained:

- **Validation Loss**: `0.4350`
- **Macro F1 Score**: `0.8657`

### Why These Metrics?

- **Validation Loss** (`BCEWithLogitsLoss`): Measures the average difference between predicted probabilities and true labels. Lower is better.
- **F1 Macro**: Averages the F1 score for each class equally, regardless of how frequent the class is. This is crucial in multi-label and imbalanced settings, as it gives fair weight to rare classes like `Quantitative Finance`.

---

## 🧪 Usage

To train the model:
```bash
python train.py

```
To run inference on a split:
```bash
python inference.py --data_path ../Data/ --train_df train.xlsx --model_path ./ --split test
```

##📚 Dependencies
Python 3.6+

PyTorch

Transformers (HuggingFace)

NLTK

pandas, numpy, scikit-learn, tqdm, openpyxl

This project demonstrates an end-to-end pipeline for handling multi-label text classification with:

Balanced loss for rare labels

Transformer-based deep learning

Interpretable and robust evaluation
