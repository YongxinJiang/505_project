import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

train_data_path = 'processed_train_data.csv'
test_data_path = 'processed_test_data.csv'

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

X_test_raw = test_data['Text']
y_test_raw = test_data['Sentiment']

non_null_subset = ['Sentiment']
cleaned_test_data = test_data.dropna(subset=non_null_subset)
cleaned_test_data = cleaned_test_data.reset_index(drop=True)

X_test_cleaned = cleaned_test_data['Text']
y_test_cleaned = cleaned_test_data['Sentiment']

X_test = X_test_cleaned
y_test = y_test_cleaned

possible_classes = [1, 0, -1]
number_of_samples = len(y_test)

random_predictions = np.random.choice(possible_classes, size=number_of_samples)

accuracy = accuracy_score(y_test, random_predictions)
accuracy_result = round(accuracy, 4)
accuracy_message = f"Random Classifier - Accuracy: {accuracy_result}"

precision = precision_score(y_test, random_predictions, average='weighted', zero_division=0)
precision_result = round(precision, 4)
precision_message = f"Random Classifier - Precision: {precision_result}"

recall = recall_score(y_test, random_predictions, average='weighted', zero_division=0)
recall_result = round(recall, 4)
recall_message = f"Random Classifier - Recall: {recall_result}"

f1 = f1_score(y_test, random_predictions, average='weighted', zero_division=0)
f1_result = round(f1, 4)
f1_message = f"Random Classifier - F1 Score: {f1_result}"

metrics = {
    "Accuracy": accuracy_result,
    "Precision": precision_result,
    "Recall": recall_result,
    "F1 Score": f1_result
}

metric_messages = {
    "Accuracy": accuracy_message,
    "Precision": precision_message,
    "Recall": recall_message,
    "F1 Score": f1_message
}

output_messages = []
output_messages.append(metric_messages["Accuracy"])
output_messages.append(metric_messages["Precision"])
output_messages.append(metric_messages["Recall"])
output_messages.append(metric_messages["F1 Score"])

for message in output_messages:
    print(message)

detailed_output = []
for metric, value in metrics.items():
    metric_message = f"{metric}: {value}"
    detailed_output.append(metric_message)

for detail in detailed_output:
    print(detail)

accuracy_metric = metrics["Accuracy"]
precision_metric = metrics["Precision"]
recall_metric = metrics["Recall"]
f1_metric = metrics["F1 Score"]

accuracy_display = f"Accuracy metric: {accuracy_metric}"
precision_display = f"Precision metric: {precision_metric}"
recall_display = f"Recall metric: {recall_metric}"
f1_display = f"F1 Score metric: {f1_metric}"

additional_messages = [
    accuracy_display,
    precision_display,
    recall_display,
    f1_display
]

for additional_message in additional_messages:
    print(additional_message)

final_messages = output_messages + detailed_output + additional_messages

for final_message in final_messages:
    print(final_message)


