import torch
import json
from transformers import RobertaTokenizer, RobertaForSequenceClassification  
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset

class ValidationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer.encode_plus(
            item["question"],
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=False,  # RoBERTa does not use token type IDs
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(item['correctness'], dtype=torch.long)
        }

# Load the tokenizer and model for RoBERTa
tokenizer = RobertaTokenizer.from_pretrained('my_trained_model_roberta') 
model = RobertaForSequenceClassification.from_pretrained('my_trained_model_roberta')
model.eval()  # Set the model to evaluation mode

with open('F1.json', 'r') as file: 
    validation_data = json.load(file)

val_dataset = ValidationDataset(validation_data, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

predictions = []
true_labels = []

# Evaluation loop
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Compute and display Precision, Recall, and F1 Score
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
