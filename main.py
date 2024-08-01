# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
from typing import List

from config.core import config

# Load the trained model
model_path = config.get('model_save_path')
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

app = FastAPI()

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class Prediction(BaseModel):
    species: str

@app.post('/predict', response_model=List[Prediction])
def predict(data: List[IrisData]):
    # Convert the incoming data to a DataFrame
    input_data = pd.DataFrame([item.dict() for item in data])
    
    # Make predictions
    predictions = model.predict(input_data)
    
    # Return predictions
    return [Prediction(species=pred) for pred in predictions]

# Example to test
@app.get('/')
def read_root():
    return {"message": "Welcome to the Iris Prediction API"}
