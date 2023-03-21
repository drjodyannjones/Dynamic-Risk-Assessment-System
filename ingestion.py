import pandas as pd
import os
import json
from datetime import datetime

# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)
input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

# Define function for data ingestion
def merge_multiple_dataframe():
    # Get the list of files in the input folder
    files = os.listdir(input_folder_path)

    # Filter the list to only include CSV files
    csv_files = [f for f in files if f.endswith('.csv')]

    # Read each CSV file into a pandas dataframe and concatenate them
    dataframes = []
    for csv_file in csv_files:
        csv_path = os.path.join(input_folder_path, csv_file)
        dataframe = pd.read_csv(csv_path)
        dataframes.append(dataframe)
    merged_dataframe = pd.concat(dataframes)

    # Write the merged dataframe to a new CSV file
    output_path = os.path.join(output_folder_path, 'finaldata.csv')
    merged_dataframe.to_csv(output_path, index=False)

    # Record the ingestion in a log file
    log_path = os.path.join(output_folder_path, 'ingestedfiles.txt')
    with open(log_path, 'a') as f:
        for csv_file in csv_files:
            ingestion_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_entry = f'File {csv_file} ingested at {ingestion_time}\n'
            f.write(log_entry)

if __name__ == '__main__':
    merge_multiple_dataframe()
