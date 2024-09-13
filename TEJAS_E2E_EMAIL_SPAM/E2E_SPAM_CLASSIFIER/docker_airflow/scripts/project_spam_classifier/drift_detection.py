import os
import json
import pandas as pd
from evidently.metrics import TextDescriptorsDriftMetric, ColumnDriftMetric
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report

def check_drift():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the 'dataset' folder
    dataset_dir = os.path.join(script_dir,'dataset')
    # Construct the full paths to the CSV files
    reference_path = os.path.join(dataset_dir, 'spam_assassin.csv')
    valid_disturbed_path = os.path.join(dataset_dir, 'spam_emails.csv')
   
    # Read the CSV files using the full paths
    reference = pd.read_csv(reference_path)
    valid_disturbed = pd.read_csv(valid_disturbed_path)

    # Set up column mapping
    column_mapping = ColumnMapping()
    column_mapping.target = 'target'
    # column_mapping.prediction = 'predict_proba'    Required if NaiveBayes model is used. I am using xgboost.
    column_mapping.text_features = ['text']

    # List features so text field is not treated as a regular feature
    column_mapping.numerical_features = []
    column_mapping.categorical_features = []

    data_drift_report = Report(
        metrics=[
            ColumnDriftMetric('target'),
           # ColumnDriftMetric('predict_proba'),  Required if NaiveBayes model is used. I am using xgboost.
            TextDescriptorsDriftMetric(column_name='text'),
        ]
    )
    data_drift_report.run(reference_data=reference,
                           current_data=valid_disturbed,
                           column_mapping=column_mapping)

    report_json = json.loads(data_drift_report.json())
    dataset_drift_check = report_json['metrics'][1]['result']['dataset_drift']

    print(dataset_drift_check)
    return dataset_drift_check


if __name__ == "__main__":
    check_drift()
  