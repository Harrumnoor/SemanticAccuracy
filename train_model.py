import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Function to load data from JSON files
def load_data(correct_file_path, incorrect_file_path):
    with open(correct_file_path, 'r') as f:
        correct_data = json.load(f)
        for item in correct_data:
            item['correctness'] = 1  # Label for correct examples

    with open(incorrect_file_path, 'r') as f:
        incorrect_data = json.load(f)
        for item in incorrect_data:
            item['correctness'] = 0  # Label for incorrect examples

    return correct_data + incorrect_data  # Combine and return the data

# Custom dataset class
class SQLCorrectnessDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer.encode_plus(
            item['question'],
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
            'labels': torch.tensor(item['correctness'], dtype=torch.long)
        }

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

# Load data
data = load_data('correct_training_data.json', 'incorrect_dataset_all.json')  # Update paths

# Split data
train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

# Create datasets and loaders
train_dataset = SQLCorrectnessDataset(train_data, tokenizer)
val_dataset = SQLCorrectnessDataset(val_data, tokenizer)
# Adjust batch size
#train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Reduced from 16 to 8
#val_loader = DataLoader(val_dataset, batch_size=8)  # Reduced from 16 to 8

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = AdamW(model.parameters(), lr=2e-5)
# Reduce the learning rate
#optimizer = AdamW(model.parameters(), lr=1e-5)  # Reduced from 2e-5 to 1e-5

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss / len(train_loader)}")

# Evaluation loop
model.eval()
total_eval_accuracy = 0
for batch in tqdm(val_loader, desc="Evaluating"):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    accuracy = (predictions == batch['labels']).cpu().numpy().mean()
    total_eval_accuracy += accuracy

# Save the trained model and tokenizer
model.save_pretrained('./my_trained_model')
tokenizer.save_pretrained('./my_trained_model')

print("Model and tokenizer have been saved.")

print(f"Validation Accuracy: {total_eval_accuracy / len(val_loader)}")
