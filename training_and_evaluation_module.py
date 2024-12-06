#This is the training and evaluation module
#Accuraccy,precision,recall,f1

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data_file_train = 'processed_train_data.csv'
data_file_test = 'processed_test_data.csv'

train_data = pd.read_csv(data_file_train)
test_data = pd.read_csv(data_file_test)

X_train_raw = train_data['Text']
y_train_raw = train_data['Sentiment']
X_test_raw = test_data['Text']
y_test_raw = test_data['Sentiment']

train_data_cleaned = train_data.dropna(subset=['Text', 'Sentiment']).reset_index(drop=True)
test_data_cleaned = test_data.dropna(subset=['Text', 'Sentiment']).reset_index(drop=True)

X_train_cleaned = train_data_cleaned['Text']
y_train_cleaned = train_data_cleaned['Sentiment']
X_test_cleaned = test_data_cleaned['Text']
y_test_cleaned = test_data_cleaned['Sentiment']

X_train = X_train_cleaned
y_train = y_train_cleaned
X_test = X_test_cleaned
y_test = y_test_cleaned

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_tfidf, y_train)

predictions = logistic_model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')

accuracy_result = round(accuracy, 4)
precision_result = round(precision, 4)
recall_result = round(recall, 4)
f1_result = round(f1, 4)

accuracy_message = f"accuracy: {accuracy_result}"
precision_message = f"precision: {precision_result}"
recall_message = f"recall: {recall_result}"
f1_message = f"f1 score: {f1_result}"

results = [
    accuracy_message,
    precision_message,
    recall_message,
    f1_message
]

for result in results:
    print(result)

metrics = {
    "Accuracy": accuracy_result,
    "Precision": precision_result,
    "Recall": recall_result,
    "F1 Score": f1_result
}

for metric, value in metrics.items():
    print(f"{metric}: {value}")


for result in results:
    print(result)
