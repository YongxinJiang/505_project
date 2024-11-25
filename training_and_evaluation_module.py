#This is the training and evaluation module
#Accuraccy,precision,recall,f1

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


train_data = pd.read_csv('processed_train_data.csv')
test_data = pd.read_csv('processed_test_data.csv')

X_train = train_data['Text']
y_train = train_data['Sentiment']
X_test = test_data['Text']
y_test = test_data['Sentiment']


train_data = train_data.dropna(subset=['Text', 'Sentiment']).reset_index(drop=True)
test_data = test_data.dropna(subset=['Text', 'Sentiment']).reset_index(drop=True)


X_train = train_data['Text']
y_train = train_data['Sentiment']
X_test = test_data['Text']
y_test = test_data['Sentiment']


vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)


y_pred = model.predict(X_test_tfidf)

# eval
# 1) accurancy
accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy: {accuracy:.4f}")

#  precision
precision = precision_score(y_test, y_pred, average='weighted')
print(f"precision: {precision:.4f}")

#  recall
recall = recall_score(y_test, y_pred, average='weighted')
print(f"recall: {recall:.4f}")

#  F1
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"f1 score: {f1:.4f}")
