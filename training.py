import os
import json
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)

output_folder_path = config['output_folder_path']
output_model_path = config['output_model_path']

def read_finaldata_csv():
    # Set the path to finaldata.csv
    finaldata_path = os.path.join(output_folder_path, 'finaldata.csv')

    # Read the CSV file into a pandas dataframe
    df = pd.read_csv(finaldata_path)

    # Return the dataframe
    return df


#################Function for training the model
def train_model():

    # Set the path to finaldata.csv
    finaldata_path = os.path.join(output_folder_path, 'finaldata.csv')

    # Read finaldata.csv into a pandas dataframe
    df = read_finaldata_csv()

    # Split the data into features (X) and labels (y)
    X = df.drop('label', axis=1)
    y = df['label']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train a logistic regression model
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='warn', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f'Accuracy: {accuracy:.2f}')

    # Write the trained model to a file using pickle
    if not os.path.exists(output_model_path):
        os.makedirs(output_model_path)
    model_path = os.path.join(output_model_path, 'trainedmodel.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
