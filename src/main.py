# src/main.py
from fastapi import FastAPI
import pickle
import pandas as pd
import uvicorn
from pydantic import BaseModel
from typing import List
import numpy as np


# Load the saved model globally when the service starts
try:
    # Load the model artifact
    with open("models/insurance_model.pkl", "rb") as f:
        model = pickle.load(f) 
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None # Handle case where model cannot be loaded


# Define the input data format for prediction
# class InsuranceData(BaseModel):
#    features: List[float]

# Initialize FastAPI app
app = FastAPI()


# 3. Define the prediction endpoint
@app.post("/predict")
def predict_insurance(data: List[float]):
    if model is None:
        return {"error": "Model not available"}, 503
        
    try:
        # Convert incoming JSON data into a format the model expects (e.g., DataFrame)
        # input_df = pd.DataFrame([data])

        # Model expects a 2D structure: [[feature_1, feature_2, ..., feature_86]]
        input_array = np.array([data])
        
        # Convert to DataFrame (without column names, which preserves positional index)
        input_df = pd.DataFrame(input_array)
        
        # Make the prediction
        prediction = model.predict(input_df).tolist()
        
        # Return the result
        return {"prediction": prediction}
        
    except Exception as e:
        return {"error": str(e)}, 400


# Define a root endpoint
@app.get("/")
def read_root():
    return {"message": "Insurance model API"}

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)