# Here is the code of the baseline
# The minimum standard

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

train_data = pd.read_csv('processed_train_data.csv')
test_data = pd.read_csv('processed_test_data.csv')

X_test = test_data['Text']
y_test = test_data['Sentiment']


test_data = test_data.dropna(subset=['Sentiment']).reset_index(drop=True)
y_test = test_data['Sentiment']


random_predictions = np.random.choice([1, 0, -1], size=len(y_test))

# Evaluating the performance of random classifiers
accuracy = accuracy_score(y_test, random_predictions)
precision = precision_score(y_test, random_predictions, average='weighted', zero_division=0)
recall = recall_score(y_test, random_predictions, average='weighted', zero_division=0)
f1 = f1_score(y_test, random_predictions, average='weighted', zero_division=0)

# 5. eval
print(f"Random Classifier - accuracy: {accuracy:.4f}")
print(f"Random Classifier - precision: {precision:.4f}")
print(f"Random Classifier - recall: {recall:.4f}")
print(f"Random Classifier - F1 score: {f1:.4f}")

