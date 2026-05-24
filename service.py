import bentoml
import pandas as pd
from bentoml.io import JSON

# Load model
model_ref = bentoml.sklearn.get("iris_classifier:latest")

# Create runner
runner = model_ref.to_runner()

# Create BentoML service
svc = bentoml.Service(
    "iris_service",
    runners=[runner]
)

# Prediction API
@svc.api(input=JSON(), output=JSON())
async def predict(input_data):

    # Convert input to DataFrame
    df = pd.DataFrame([input_data])

    # Predict
    prediction = await runner.predict.async_run(df)

    # Return result
    return {
        "prediction": prediction.tolist()
    }