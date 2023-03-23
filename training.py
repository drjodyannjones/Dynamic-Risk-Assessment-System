import os
import json
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)

output_folder_path = config['output_folder_path']
output_model_path = config['output_model_path']

def read_finaldata_csv():
    finaldata_path = os.path.join(output_folder_path, 'finaldata.csv')
    df = pd.read_csv(finaldata_path)
    return df

def preprocess_data(df):
    # Drop the 'corporation' column
    df.drop(columns=['corporation'], inplace=True)

    for column in df.columns:
        # Convert non-numeric values to NaN
        df[column] = pd.to_numeric(df[column], errors='coerce')

        # If the column is not the 'label' column, replace NaN values with the mean of the column
        if column != 'exited':
            df[column].fillna(df[column].mean(), inplace=True)

    # Drop rows with NaN values in the 'label' column
    df.dropna(subset=['exited'], inplace=True)

    return df


#################Function for training the model
def train_model():
    finaldata_path = os.path.join(output_folder_path, 'finaldata.csv')
    df = read_finaldata_csv()

    # Preprocess the data
    df = preprocess_data(df)

    # Split the data into features (X) and labels (y)
    X = df.drop('exited', axis=1)
    y = df['exited']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train a logistic regression model
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    model.fit(X_train, y_train)

    # Evaluate the model on the test set using F1 score
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(f'F1 Score: {f1:.2f}')

    # Write the trained model to a file using pickle
    if not os.path.exists(output_model_path):
        os.makedirs(output_model_path)
    model_path = os.path.join(output_model_path, 'trainedmodel.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    return y_pred, y_test

if __name__ == '__main__':
    predictions, y_true = train_model()
