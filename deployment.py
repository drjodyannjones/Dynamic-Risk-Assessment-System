import os
import json
import shutil


##################Load config.json and correct path variable
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
output_model_path = config['output_model_path']
prod_deployment_path = os.path.join(config['prod_deployment_path'])
output_folder_path = os.path.join(config['output_folder_path'])


####################function for deployment
def store_model_into_pickle():
    # Create the production_deployment directory if it doesn't already exist
    if not os.path.exists(prod_deployment_path):
        os.makedirs(prod_deployment_path)

    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory

    # Copy the latest pickle file
    source_pickle_file = os.path.join(output_model_path, 'trainedmodel.pkl')
    dest_pickle_file = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
    shutil.copyfile(source_pickle_file, dest_pickle_file)

    # Copy the latestscore.txt value
    source_score_file = os.path.join(output_model_path, 'latestscore.txt')
    dest_score_file = os.path.join(prod_deployment_path, 'latestscore.txt')
    shutil.copyfile(source_score_file, dest_score_file)

    # Copy the ingestfiles.txt file
    source_ingest_file = os.path.join(output_folder_path, 'ingestedfiles.txt')
    dest_ingest_file = os.path.join(prod_deployment_path, 'ingestedfiles.txt')
    shutil.copyfile(source_ingest_file, dest_ingest_file)

    print('Files successfully copied to deployment directory.')


if __name__ == '__main__':
    store_model_into_pickle()
