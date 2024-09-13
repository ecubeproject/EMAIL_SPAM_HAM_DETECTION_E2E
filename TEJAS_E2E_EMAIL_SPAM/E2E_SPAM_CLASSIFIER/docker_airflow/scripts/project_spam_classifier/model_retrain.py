import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBClassifier  # Import XGBoost classifier
import pickle


def retrain_model():
    # Get the AIRFLOW_HOME environment variable or default to '/opt/airflow' for Linux
    base_dir = os.getenv('AIRFLOW_HOME', '/opt/airflow')
    print(f"AIRFLOW_HOME: {base_dir}")

    # Construct the file path dynamically based on the environment
    dataset_dir = os.path.join(base_dir, 'scripts', 'project_spam_classifier', 'dataset')
    file_path = os.path.join(dataset_dir, 'training_data.csv')
    file_path = os.path.normpath(file_path)  # Normalize the path to use correct separators
    print(f"Dataset Path: {file_path}")

    # Load the data
    df = pd.read_csv(file_path)
    print(df.head())  # Debugging output

    # Create pipeline
    vectorizer = CountVectorizer()
    xgboost_classifier = XGBClassifier()

    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', xgboost_classifier)
    ])

    # Fit the model
    pipeline.fit(df['text'], df['target'])

    # Update the output directory to match the volume path in Docker
    output_dir = os.path.join(base_dir, 'app')
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    output_file = 'spam_classifier_pipeline.pkl'
    output_path = os.path.join(output_dir, output_file)
    output_path = os.path.normpath(output_path)  # Normalize the path
    print(f"Full Path for Model Save: {output_path}")

    # Save the pipeline
    with open(output_path, 'wb') as file:
        pickle.dump(pipeline, file)


if __name__ == "__main__":
    retrain_model()
