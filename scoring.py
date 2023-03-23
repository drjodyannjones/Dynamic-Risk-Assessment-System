from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
from training import preprocess_data
from diagnostics import model_predictions

#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)

test_data_path = config['test_data_path']
model_path = config['output_model_path']


#################Function for model scoring
def score_model(predictions, y_true):
    # Load test data
    test_data_csv_path = os.path.join(test_data_path, 'testdata.csv')
    test_data = pd.read_csv(test_data_csv_path)

    # Preprocess test data
    test_data = preprocess_data(test_data)

    # Split test data into features (X) and labels (y)
    X_test = test_data.drop('exited', axis=1)
    y_test = test_data['exited']

    # Calculate the F1 score
    f1_score = metrics.f1_score(y_true, predictions)

    # Write the result to the latestscore.txt file in the output_model_path directory
    latest_score_file_path = os.path.join(model_path, 'latestscore.txt')
    with open(latest_score_file_path, 'w') as f:
        f.write(f'F1 Score: {f1_score:.2f}\n')

    print(f'F1 Score: {f1_score:.2f}')

if __name__ == '__main__':
    # Get model predictions and load test labels
    predictions = model_predictions()
    test_data = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    y_true = test_data['exited']

    # Score the model on the test data
    score_model(predictions, y_true)
