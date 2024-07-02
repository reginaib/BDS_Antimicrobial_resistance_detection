import numpy as np
import pandas as pd
import time
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, average_precision_score


def evaluate_model(model, x_train, y_train, x_valid, y_valid, task=None):
    model.fit(x_train, y_train)
    
    # Predict on training and validation data
    y_train_pred = model.predict(x_train)
    y_valid_pred = model.predict(x_valid)
    
    # training data
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_balanced_accuracy = balanced_accuracy_score(y_train, y_train_pred)
    train_precision_weighted = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
    train_recall_weighted = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
    train_f1_weighted = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
    
    if task == 1:
        train_auprc = average_precision_score(y_train, model.predict_proba(x_train), average='weighted')
    elif task == 2:
        y_train_probabilities = model.predict_proba(x_train)
        train_auprc = np.mean([average_precision_score((y_train == c).astype(int), y_train_probabilities[:, c]) 
                        for c in range(y_train_probabilities.shape[1])])
    # validation data
    valid_accuracy = accuracy_score(y_valid, y_valid_pred)
    valid_balanced_accuracy = balanced_accuracy_score(y_valid, y_valid_pred)
    valid_precision_weighted = precision_score(y_valid, y_valid_pred, average='weighted', zero_division=0)
    valid_recall_weighted = recall_score(y_valid, y_valid_pred, average='weighted', zero_division=0)
    valid_f1_weighted = f1_score(y_valid, y_valid_pred, average='weighted', zero_division=0)

    if task == 1:
        valid_auprc = average_precision_score(y_valid, model.predict_proba(x_valid), average='weighted')
    elif task == 2:
        y_valid_probabilities = model.predict_proba(x_valid)
        valid_auprc = np.mean([average_precision_score((y_valid == c).astype(int), y_valid_probabilities[:, c]) 
                        for c in range(y_valid_probabilities.shape[1])])
    
    results = {
        'Metric': ['Accuracy', 'Balanced Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'F1 Score (Weighted)', 'AUPRC'],
        'Train Score': [
            train_accuracy, train_balanced_accuracy, train_precision_weighted,
            train_recall_weighted, train_f1_weighted, train_auprc
        ],
        'Validation Score': [
            valid_accuracy, valid_balanced_accuracy, valid_precision_weighted,
            valid_recall_weighted, valid_f1_weighted, valid_auprc
        ]
    }
    
    return pd.DataFrame(results)


def format_time(seconds):
    minutes = int(seconds // 60)
    seconds_remaining = round(seconds % 60)  
    return f'{minutes} min {seconds_remaining} sec'


def evaluate_model_with_time(model, x_train, y_train, x_valid, y_valid, task=None):
    start_time = time.time() 
    results = evaluate_model(model, x_train, y_train, x_valid, y_valid, task=task) 
    end_time = time.time()

    execution_time_seconds = end_time - start_time 
    execution_time_formatted = format_time(execution_time_seconds)
    time_df = pd.DataFrame({'Execution Time': [execution_time_formatted]})
    
    final_df = pd.concat([results, time_df], axis=1) 
    final_df['Execution Time'] = final_df['Execution Time'].ffill()
    return final_df