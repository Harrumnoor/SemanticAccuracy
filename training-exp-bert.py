import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import json
import os

# Function to load data
def load_data(correct_file_path, incorrect_file_path):
    with open(correct_file_path, 'r') as f:
        correct_data = json.load(f)
    with open(incorrect_file_path, 'r') as f:
        incorrect_data = json.load(f)

    for item in correct_data:
        item['correctness'] = 1
    for item in incorrect_data:
        item['correctness'] = 0

    return correct_data + incorrect_data

# Custom dataset class for dual encoding
class SQLCorrectnessDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sql_query = item.get('query', item.get('predicted_parse', ''))
        inputs = self.tokenizer.encode_plus(
            item['question'] + " [SEP] " + sql_query,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(item['correctness'], dtype=torch.long)
        }

# Model with a classification layer on top of BERT
class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output[1]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)


# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertClassifier()

# Load and split data
data = load_data('correct_training_data.json', 'incorrect_training_data.json')
train_data, val_data = train_test_split(data, test_size=0.1)

train_dataset = SQLCorrectnessDataset(train_data, tokenizer)
val_dataset = SQLCorrectnessDataset(val_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training and evaluation
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 3  # Assuming 3 epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

for epoch in range(3):
    model.train()
    total_train_loss = 0
    total_train_accuracy = 0
    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(batch['input_ids'], batch['attention_mask'])
        loss = nn.CrossEntropyLoss()(outputs, batch['labels'])
        total_train_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        total_train_accuracy += (preds == batch['labels']).cpu().numpy().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_train_accuracy = total_train_accuracy / len(train_loader)
    print(f'Epoch {epoch+1} | Average Training Loss: {avg_train_loss} | Average Training Accuracy: {avg_train_accuracy}')

    # Validation
    model.eval()
    total_eval_loss = 0
    total_eval_accuracy = 0
    for batch in tqdm(val_loader, desc='Validation'):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(batch['input_ids'], batch['attention_mask'])
            loss = nn.CrossEntropyLoss()(outputs, batch['labels'])
            total_eval_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            total_eval_accuracy += (preds == batch['labels']).cpu().numpy().mean()

    avg_val_loss = total_eval_loss / len(val_loader)
    avg_val_accuracy = total_eval_accuracy / len(val_loader)
    print(f'Validation Loss: {avg_val_loss} | Validation Accuracy: {avg_val_accuracy}')

# Save the model and tokenizer
model_dir = './my_trained_model_final'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
