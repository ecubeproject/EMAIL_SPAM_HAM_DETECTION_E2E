#server.py contents

import pandas as pd
import numpy as np
import pickle
from fastapi import FastAPI


filename = 'app/spam_classifier_pipeline.pkl'

with open(filename, 'rb') as file:
    model = pickle.load(file)

class_names = np.array(['Ham','Spam'])

# Initialize the app as a FastAPI class instance
app = FastAPI()

@app.get('/')
def read_root():
    return {'message': 'Spam Model Classifier API'}

@app.post('/predict')
def predict(data: dict) -> str:
    """
    Predicts the class of a given set of features.

    Args:
        data (dict): A dictionary containing the email to predict.
        e.g. {"text": "What is this email?"}

    Returns:
        dict: A dictionary containing the predicted class.
    """        
    
    prediction = model.predict(pd.Series(data['text']))
    class_name = class_names[prediction][0]
    return class_name