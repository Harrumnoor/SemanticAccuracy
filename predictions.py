import tensorflow as tf
from transformers import BertTokenizer
import json
import numpy as np
import pandas as pd

def load_data_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    # Map each example into input features
    def gen():
        for example in examples:
            inputs = tokenizer.encode_plus(
                example['question'],
                example['predicted_parse'],
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_token_type_ids=True
            )
            yield {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask'],
                'token_type_ids': inputs['token_type_ids']
            }

    return tf.data.Dataset.from_generator(
        gen,
        output_signature={
            'input_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'attention_mask': tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'token_type_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32)
        }
    )

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

# Load and preprocess the test dataset
test_data = load_data_from_json('F1.json')
test_dataset = convert_examples_to_tf_dataset(test_data, tokenizer)
test_dataset = test_dataset.batch(32)  # Adjust the batch size if necessary

# Load the fine-tuned model
model_directory = 'my_trained_model/'
model = tf.keras.models.load_model(model_directory)

# Run the model to get predictions
predictions = model.predict(test_dataset)


if isinstance(predictions, dict):
    predicted_logits = predictions['logits']
else:
    predicted_logits = predictions

predicted_classes = np.argmax(predicted_logits, axis=1)

output_df = pd.DataFrame({
    'question': [example['question'] for example in test_data],
    'predicted_parse': [example['predicted_parse'] for example in test_data],
    'predicted_class': predicted_classes
})

output_df.to_csv('predictions.csv', index=False)

print("Predictions have been saved to 'predictions.csv'")