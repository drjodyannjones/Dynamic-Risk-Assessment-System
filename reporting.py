import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from training import preprocess_data

###############Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')
output_model_path = config['output_model_path']

##############Function for reporting
def score_model():
    # Load the deployed model
    model_path = os.path.join(output_model_path, 'trainedmodel.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Read test data
    test_data = pd.read_csv(test_data_path)
    test_data = preprocess_data(test_data)

    # Calculate predictions and actual values
    X_test = test_data.drop('exited', axis=1)
    y_test = test_data['exited']
    predictions = model.predict(X_test)

    # Calculate the confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_test, predictions)

    # Visualize the confusion matrix as a heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap="YlGnBu")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save the heatmap to a file
    plt.savefig('confusion_matrix.png')

if __name__ == '__main__':
    score_model()
