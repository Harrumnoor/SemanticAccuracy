import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Function to load and prepare data
def load_data(correct_file, incorrect_file):
    with open(correct_file, 'r') as f:
        correct_data = [{'input_text': item['question'], 'target_text': 'correct'} for item in json.load(f)]

    with open(incorrect_file, 'r') as f:
        incorrect_data = [{'input_text': item['question'], 'target_text': 'incorrect'} for item in json.load(f)]

    return correct_data + incorrect_data

# Custom dataset class
class SQLCorrectnessDataset(Dataset):
    def __init__(self, data, tokenizer, source_max_token_len=512, target_max_token_len=32):
        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_row = self.data[idx]

        source_encoding = self.tokenizer(
            data_row['input_text'],
            max_length=self.source_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        target_encoding = self.tokenizer(
            data_row['target_text'],
            max_length=self.target_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        labels = target_encoding['input_ids']
        labels[labels == 0] = -100  # To make sure we ignore padding in the loss computation

        return dict(
            input_ids=source_encoding['input_ids'].flatten(),
            attention_mask=source_encoding['attention_mask'].flatten(),
            labels=labels.flatten()
        )

# Initialize tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Load data and create datasets
data = load_data('correct_training_data.json', 'incorrect_dataset_all.json') 
train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
train_dataset = SQLCorrectnessDataset(train_data, tokenizer)
val_dataset = SQLCorrectnessDataset(val_data, tokenizer)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Prepare for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3 
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch: {epoch + 1}, Loss: {avg_train_loss:.2f}")

    # Validation loop - Implement this if needed

# Save the model and tokenizer
model.save_pretrained('./fine_tuned_T5')
tokenizer.save_pretrained('./fine_tuned_T5')
