import torch
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset

# Custom Dataset class for PyTorch
class T5ValidationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Format the input text as a question about the correctness of the SQL query
        input_text = f"question: {item['question']} SQL: {item['query']} Is this SQL correct?"
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        target_text = "correct" if item["correctness"] == 1 else "incorrect"
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }

tokenizer = T5Tokenizer.from_pretrained('finetuned_T5')
model = T5ForConditionalGeneration.from_pretrained('finetuned_T5')
model.eval()  # Set the model to evaluation mode

with open('F1.json', 'r') as file: 
    validation_data = json.load(file)

val_dataset = T5ValidationDataset(validation_data, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)  

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

        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
        decoded_preds = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]
        decoded_labels = [tokenizer.decode(ids, skip_special_tokens=True) for ids in labels]

        predictions.extend(decoded_preds)
        true_labels.extend(["correct" if label == 1 else "incorrect" for label in decoded_labels])  # Adjust according to your labels

# Convert predictions and true labels to binary
binary_predictions = [1 if pred.lower() == "correct" else 0 for pred in predictions]
binary_true_labels = [1 if label.lower() == "correct" else 0 for label in true_labels]

# Compute Accuracy, Precision, Recall, and F1 Score
accuracy = accuracy_score(binary_true_labels, binary_predictions)
precision = precision_score(binary_true_labels, binary_predictions)
recall = recall_score(binary_true_labels, binary_predictions)
f1 = f1_score(binary_true_labels, binary_predictions)

# Display the evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
