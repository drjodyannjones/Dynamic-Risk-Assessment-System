import pandas as pd
import numpy as np
import time
import os
import json
import pickle
from sklearn.metrics import f1_score
import subprocess
import sys
from training import preprocess_data
import subprocess

##################Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')
output_model_path = config['output_model_path']

##################Function to get model predictions
def model_predictions():
    # Read the deployed model
    model_path = os.path.join(output_model_path, 'trainedmodel.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Read test data
    test_data = pd.read_csv(test_data_path)

    # Preprocess test data
    test_data = preprocess_data(test_data)

    # Calculate predictions
    X_test = test_data.drop('exited', axis=1)
    y_test = test_data['exited']
    predictions = model.predict(X_test)

    return predictions.tolist()

##################Function to get summary statistics
def dataframe_summary():
    # Read the test data
    test_data = pd.read_csv(test_data_path)

    # Calculate summary statistics
    summary_statistics = test_data.describe().to_dict()

    return summary_statistics

##################Function to get timings
def execution_time():
    # Timing training.py and ingestion.py
    start_time = time.time()
    os.system("python training.py")
    training_time = time.time() - start_time

    start_time = time.time()
    os.system("python ingestion.py")
    ingestion_time = time.time() - start_time

    return [training_time, ingestion_time]

##################Function to check dependencies
def outdated_packages_list():
    # Get a list of outdated packages
    outdated_packages = []

    try:
        result = subprocess.check_output(["pip", "list", "--outdated", "--format=json"]).decode("utf-8")
        outdated_packages = json.loads(result)
    except Exception as e:
        print(f"Error occurred while fetching outdated packages: {e}")

    return outdated_packages

def na_percentage():
    # Read the test data
    test_data = pd.read_csv(test_data_path)

    # Calculate the percentage of NA values in each numeric column
    na_percent = test_data.select_dtypes(include=[np.number]).isna().mean().round(4) * 100

    return na_percent.to_dict()


if __name__ == '__main__':
    print("Model Predictions:", model_predictions())
    print("\nDataframe Summary:", dataframe_summary())
    print("\nExecution Time:", execution_time())
    print("\nOutdated Packages List:", outdated_packages_list())
    print("\nPercentage of NA values in Numeric Columns:", na_percentage())
