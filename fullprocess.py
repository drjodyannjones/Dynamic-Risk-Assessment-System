import os
import json
import pandas as pd
from sklearn.metrics import f1_score
import pickle
import scoring
from training import preprocess_data


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

sourcedata_folder_path = os.path.join(config['input_folder_path'])
ingesteddata_folder_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
latestscore_path = os.path.join(prod_deployment_path, 'latestscore.txt')

# Check for new data
with open(os.path.join(prod_deployment_path, 'ingestedfiles.txt'), 'r') as f:
    ingested_files = set(f.read().splitlines())

all_files = set(os.listdir(sourcedata_folder_path))
new_files = all_files - ingested_files


if len(new_files) == 0:
    print("No new data to process")
else:
    print("New data detected:", new_files)

    for file in new_files:
        file_path = os.path.join(sourcedata_folder_path, file)
        df_new = pd.read_csv(file_path)

        # Preprocess new data
        df_new = preprocess_data(df_new)

        # Load the trained model
        model_path = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Make predictions on new data
        X_new = df_new.drop('exited', axis=1)
        y_new = df_new['exited']
        predictions = model.predict(X_new)

        # Calculate F1 score for new predictions
        f1score = f1_score(y_new, predictions)

        # Check for model drift
        with open(latestscore_path, 'r') as f:
            latestscore = float(f.read().strip().split(':')[1])

        if f1score < latestscore:
            print("Model drift has occurred.")

        # Write new data to ingestedfiles.txt
        with open(os.path.join(prod_deployment_path, 'ingestedfiles.txt'), 'a') as f:
            f.write(file + "\n")

        # Update latestscore.txt
        with open(latestscore_path, 'w') as f:
            f.write(f"F1 Score: {f1score:.2f}")

        # Run scoring.py on new predictions
        score = scoring.score_model(predictions, y_new)

        # Print summary statistics
        print(f"New data summary statistics:\n{df_new.describe()}")
        print(f"F1 Score: {f1score:.2f}")
        print(f"Score: {score:.2f}\n")
