from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
import numpy as np
import tensorflow as tf

app = FastAPI()

# Load the Keras model
model = tf.keras.models.load_model("./best_model_diabetes.h5")  # change filename if needed


class PredictionRequest(BaseModel):
    inputs: Dict[str, Any]


@app.post("/predict")
async def predict(request: PredictionRequest):
    inputs = request.inputs

    try:
        # Example: extract features in the correct order expected by your model
        feature_order = ['bmi', 'age', 'income', 'education', 'genhlth', 'physhlth', 'menthlth']

        print("Feature values:", [(f, inputs.get(f)) for f in feature_order])
        features = [float(inputs.get(f, 0)) for f in feature_order]
        input_array = np.array([features])  # Shape (1, n_features)


        if len(features) != 7:
            raise HTTPException(status_code=400, detail=f"Expected 7 features, got {len(features)}")

        # Predict
        prediction = model.predict(input_array)
        diabetes_score = float(prediction[0][0])  # Assuming binary classification

        return [[
            {"label": "non-diabetic", "score": round(1 - diabetes_score, 2)},
            {"label": "diabetic", "score": round(diabetes_score, 2)},
        ]]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

