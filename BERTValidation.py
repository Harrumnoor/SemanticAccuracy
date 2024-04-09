import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertConfig
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset
import json

# Custom Dataset class for PyTorch
class ValidationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sql_query = item.get('query') or item.get('predicted_parse')  # Adjusted for potential keys in your data
        combined_input = f"{item['question']} [SEP] {sql_query}"  # Combined input with question and SQL query

        inputs = self.tokenizer.encode_plus(
            combined_input,
            add_special_tokens=True,
            max_length=self.max_length,
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

config = BertConfig.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification(config)
model.load_state_dict(torch.load('./my_trained_model_v2/model_state_dict.pt'))
model.eval()
# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('./my_trained_model_v2')
#model = BertForSequenceClassification.from_pretrained('./my_trained_model_v2')
#model.eval()  # Set the model to evaluation mode

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

# Compute Precision, Recall, and F1 Score
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

# Display the evaluation metrics
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
