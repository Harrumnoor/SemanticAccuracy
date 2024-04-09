import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
import json
import os
from tqdm import tqdm

# Function to load data
def load_data(correct_file_path, incorrect_file_path):
    with open(correct_file_path, 'r') as f:
        correct_data = json.load(f)
    with open(incorrect_file_path, 'r') as f:
        incorrect_data = json.load(f)

    # Add correctness labels
    for item in correct_data:
        item['correctness'] = 1
    for item in incorrect_data:
        item['correctness'] = 0

    return correct_data + incorrect_data  # Combine data

# Custom dataset class for dual encoding
class SQLCorrectnessDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):  # Half max_len for each part
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Encode question
        question_encoding = self.tokenizer.encode_plus(
            item['question'],
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        # Encode SQL
        # Check if 'predicted_parse' is available; otherwise, use 'query'
        sql_text = item.get('predicted_parse', item.get('query', ''))
        sql_encoding = self.tokenizer.encode_plus(
            sql_text,  # Use either 'predicted_parse' or 'query'
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        # Concatenate question and SQL encodings
        input_ids = torch.cat([question_encoding['input_ids'], sql_encoding['input_ids']], dim=1)
        attention_mask = torch.cat([question_encoding['attention_mask'], sql_encoding['attention_mask']], dim=1)

        return {
            'input_ids': input_ids.squeeze(0),
            'attention_mask': attention_mask.squeeze(0),
            'labels': torch.tensor(item['correctness'], dtype=torch.long)
        }

# Model with a classification layer on top of BERT
class BertClassifier(nn.Module):
    def __init__(self, max_len=512):  # Add max_len as a parameter with a default value
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)  # Binary classification
        self.max_len = max_len  # Set max_len as an attribute

    def forward(self, input_ids, attention_mask):
        # Ensure input_ids and attention_mask are within max_len
        input_ids = input_ids[:, :self.max_len]
        attention_mask = attention_mask[:, :self.max_len]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        pooled_output = outputs[1]  # Get the pooled output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertClassifier()

# Load data
data = load_data('correct_training_data.json', 'incorrect_training_data.json')

# Split data and create dataloaders
train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
train_dataset = SQLCorrectnessDataset(train_data, tokenizer)
val_dataset = SQLCorrectnessDataset(val_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader)}")


# Define the directory where you want to save your model
model_dir = './my_trained_model_v2'

# Check if the directory exists, and if not, create it
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Now it's safe to save your model's state dictionary
torch.save(model.state_dict(), os.path.join(model_dir, 'model_state_dict.pt'))

# Also, save your tokenizer in the same directory
tokenizer.save_pretrained(model_dir)

print("Model state dict and tokenizer have been saved.")

# Evaluation loop
model.eval()
total_eval_accuracy = 0
total_eval_loss = 0
criterion = torch.nn.CrossEntropyLoss()  # Initialize the loss function

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Get model outputs (logits)
        logits = model(input_ids, attention_mask)

        # Compute loss
        loss = criterion(logits, labels)
        total_eval_loss += loss.item()

        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        correct_predictions = torch.sum(preds == labels)
        total_eval_accuracy += correct_predictions.item()

# Compute the average loss and accuracy over the validation set
avg_val_loss = total_eval_loss / len(val_loader)
avg_val_accuracy = total_eval_accuracy / len(val_loader.dataset)

print(f"Validation Loss: {avg_val_loss}")
print(f"Validation Accuracy: {avg_val_accuracy}")