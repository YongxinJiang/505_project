#This is the data_processing.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import re


file_path = 'twitter_training.csv'
data = pd.read_csv(file_path, delimiter=',', header=None)


data.columns = ['ID', 'Topic', 'Sentiment', 'Text']
data = data[['Sentiment', 'Text']]

data = data.dropna()


label_mapping = {'Positive': 1, 'Negative': -1, 'Neutral': 0}
data['Sentiment'] = data['Sentiment'].map(label_mapping)


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

data['Text'] = data['Text'].apply(preprocess_text)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['Text'])
y = data['Sentiment']

# 80% 20%
X_train, X_test, y_train, y_test = train_test_split(data['Text'], data['Sentiment'], test_size=0.2, random_state=42)

train_data = pd.DataFrame({'Text': X_train, 'Sentiment': y_train})
test_data = pd.DataFrame({'Text': X_test, 'Sentiment': y_test})

train_data.to_csv('processed_train_data.csv', index=False)
print("training set is saved in processed_train_data.csv")

test_data.to_csv('processed_test_data.csv', index=False)
print("test set is saved in processed_test_data.csv")
