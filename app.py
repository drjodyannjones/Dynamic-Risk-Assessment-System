from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os

# Import the necessary functions from the respective files
from training import preprocess_data
from diagnostics import execution_time, outdated_packages_list, na_percentage
from scoring import score_model

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

@app.route("/", methods=['GET'])
def index():
    return "Welcome to the Flask API!"

with open('config.json', 'r') as f:
    config = json.load(f)

prod_deployment_path = os.path.join(config['prod_deployment_path'])

# Load the trained model
model_path = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
with open(model_path, 'rb') as f:
    prediction_model = pickle.load(f)

#######################Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame(data)
    df_processed = preprocess_data(df)
    predictions = prediction_model.predict(df_processed)
    return jsonify(predictions.tolist())

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET', 'OPTIONS'])
def scoring():
    data = request.get_json(force=True)
    y_true = data['true']
    y_pred = data['pred']
    f1_score = score_model(y_pred, y_true)
    return jsonify({'f1_score': f1_score})

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def summary_stats():
    data = request.get_json(force=True)
    df = pd.DataFrame(data)
    summary_statistics = df.describe().to_dict()
    return jsonify(summary_statistics)

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnostics():
    timing = execution_time()
    outdated_packages = outdated_packages_list()
    diagnostics = {'timing': timing, 'outdated_packages': outdated_packages, 'na_percentage': na_percentage}
    return jsonify(diagnostics)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
