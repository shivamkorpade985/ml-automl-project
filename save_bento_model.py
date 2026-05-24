import bentoml 
from joblib import load 
 
# Load trained model 
model = load("best_iris_model.pkl") 
 
# Save model 
bentoml.sklearn.save_model( 
    "iris_classifier", 
    model 
) 
 
print("Model saved successfully") 
