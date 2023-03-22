from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os

# Import the necessary functions from the respective files
from training import *
from diagnostics import *
from scoring import *

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])

prediction_model = None

#######################Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    predictions = model_predictions()
    return jsonify(predictions)

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET', 'OPTIONS'])
def scoring():
    f1_score = score_model()
    return jsonify({'f1_score': f1_score})

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def summary_stats():
    summary_statistics = dataframe_summary()
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
