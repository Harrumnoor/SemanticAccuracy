#baseline
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data(correct_file_path, incorrect_file_path):
    with open(correct_file_path, 'r') as f:
        correct_data = json.load(f)
        correct_questions = [item['question'] for item in correct_data]
        correct_labels = [1] * len(correct_data)  # Correct examples have label 1

    with open(incorrect_file_path, 'r') as f:
        incorrect_data = json.load(f)
        incorrect_questions = [item['question'] for item in incorrect_data]
        incorrect_labels = [0] * len(incorrect_data)  # Incorrect examples have label 0

    # Combine correct and incorrect data
    questions = correct_questions + incorrect_questions
    labels = correct_labels + incorrect_labels
    return questions, labels

correct_file_path = 'correct_training_data.json'
incorrect_file_path = 'incorrect_dataset_all.json'

X, y = load_data(correct_file_path, incorrect_file_path)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create feature vectors
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train_vectors, y_train)

# Predict on the test set
y_pred = model.predict(X_test_vectors)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
