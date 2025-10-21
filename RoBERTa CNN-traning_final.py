import os
import random
import string

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
import torchvision
import torchvision.transforms as transforms
from transformers import BertTokenizer, RobertaTokenizer, RobertaModel, RobertaForSequenceClassification, DistilBertForSequenceClassification
from tqdm.auto import tqdm
import wandb

nltk.download('wordnet')
nltk.download('stopwords')

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Define the hybrid model with RoBERTa and a CNN
class RoBERTaCNN(nn.Module):
    def __init__(self, roberta, num_classes, cnn_out_channels=100, cnn_kernel_size=2, cnn_stride=1):
        super(RoBERTaCNN, self).__init__()
        self.roberta = roberta
        self.embedding_dim = roberta.config.hidden_size
        self.cnn = nn.Conv1d(self.embedding_dim, cnn_out_channels, kernel_size=cnn_kernel_size, stride=cnn_stride)
        self.fc = nn.Linear(cnn_out_channels, num_classes)

    def forward(self, input_ids, attention_mask):
        # Get embeddings from RoBERTa
        embeddings = self.roberta(input_ids, attention_mask).last_hidden_state
        embeddings = embeddings.permute(0, 2, 1) # swap dimensions for CNN

        # Apply CNN
        cnn_features = self.cnn(embeddings)
        cnn_features = nn.functional.relu(cnn_features)
        cnn_features = nn.functional.max_pool1d(cnn_features, kernel_size=cnn_features.shape[-1])
        cnn_features = cnn_features.squeeze(dim=-1)

        # Apply connected layer
        output = self.fc(cnn_features)

        return output
    
# Define dataset class
class SuicideDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Data Augmentation
def normalize_text(text):
    # Define stopwords and stemmer
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # Stemming
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text 

# Replace synonyms
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def synonym_replacement(sentence, num_replacements=1):
    if not isinstance(sentence, str):
        print(f"Unexpected input type: {type(sentence)}, value: {sentence}")
        return sentence  # Return the input as is if it's not a string

    words = sentence.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words]))
    random.shuffle(random_word_list)
    num_replaced = 0

    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= num_replacements:
            break

    sentence = ' '.join(new_words)
    return sentence

def evaluate_model(model, loader, device):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    
    return accuracy, precision, recall, f1, auc

def train(model, train_loader, val_loader, optimizer, loss_fn, device, epochs):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct_preds = 0
        total_preds = 0
        
        for batch in train_loader:
            # Move data to the specified device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            # Compute loss and backpropagate
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()
        
        # Validation at the end of the epoch
        val_accuracy, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader)}, Train Accuracy: {correct_preds/total_preds}")
        # Log metrics using WandB
        wandb.log({
            "Train Loss": train_loss / len(train_loader),
            "Train Accuracy": correct_preds / total_preds,
            "Validation Accuracy": val_accuracy,
            "Validation Precision": val_precision,
            "Validation Recall": val_recall,
            "Validation F1": val_f1,
            "Validation AUC": val_auc
        })

def main():
# Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    # Initialize WandB
    wandb.init(project='pytorch-RoBERTa',
           config={
               'epochs': int(args.epochs),
               'classes': 2,
               'kernels': [16, 32],
               'batch_size': int(args.batch_size),
               'learning_rate': float(args.lr),
               'dataset': "data",
               'architecture': "CNN",
           })
    
    # Data Preprocessing
    data = pd.read_csv('Suicide_Dataset.csv')
    data['class'] = data['class'].map({'non suicide': 0, 'non-suicide': 0, 'suicide': 1})

    # Rename the 'class' column to 'labels'
    data.rename(columns={'class': 'label'}, inplace=True)

    # Apply the normalization function on the 'text' column of the dataset
    data['text'] = data['text'].apply(normalize_text)

    # Apply replace sysnonym
    data['text'] = data['text'].apply(synonym_replacement)

    # Back-translation for Text Classification
    # Apply augmentation on the data
    augmented_texts = []
    augmented_labels = []

    for idx, row in data.iterrows():
        augmented_text = synonym_replacement(row['text'])
    
        # Only append if the augmented text is different from the original
        if augmented_text != row['text']:
            augmented_texts.append(augmented_text)
            augmented_labels.append(row['label'])  # Assuming 'label' is the column name for labels

    # Combine the original and augmented data
    all_texts = data['text'].tolist() + augmented_texts
    all_labels = data['label'].tolist() + augmented_labels

    combined_data = pd.DataFrame({
        'text': all_texts,
        'label': all_labels  
    })

    # Load the necessary libraries and the RoBERTa model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    roberta = RobertaModel.from_pretrained('roberta-base')

    num_classes = 2 #2 classes, suicide and non-suicide
    model = RoBERTaCNN(roberta, num_classes)

    # Filter out rows with label value 2
    filtered_data = combined_data[combined_data['label'] != 2]

    # Split data into train, validation, and test sets
    train_data, temp_data = train_test_split(filtered_data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Convert labels to integers
    label_encoder = LabelEncoder()
    label_encoder.fit(filtered_data['label'])

    train_labels = label_encoder.transform(train_data['label'])  
    val_labels = label_encoder.transform(val_data['label'])
    test_labels = label_encoder.transform(test_data['label'])


    # Convert labels to tensors
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    val_labels = torch.tensor(val_labels, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    
    # Check unique values in labels
    unique_train_labels = torch.unique(train_labels)
    unique_val_labels = torch.unique(val_labels)
    unique_test_labels = torch.unique(test_labels)

    # Get texts
    train_texts = train_data['text'].tolist()
    val_texts = val_data['text'].tolist()
    test_texts = test_data['text'].tolist()

    # Create datasets
    max_len = 512
    train_dataset = SuicideDataset(train_texts, train_labels, tokenizer, max_len)
    val_dataset = SuicideDataset(val_texts, val_labels, tokenizer, max_len)
    test_dataset = SuicideDataset(test_texts, test_labels, tokenizer, max_len)

    # DataLoader
    batch_size = wandb.config.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define the device, loss function, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    
    # Call the train function
    train(model, train_loader, val_loader, optimizer, loss_fn, device, wandb.config.epochs)

    wandb.finish()

    # Evaluation on test set
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            test_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    print("Test Report:")
    print(classification_report(test_labels, test_preds))


if __name__ == '__main__':
    main()