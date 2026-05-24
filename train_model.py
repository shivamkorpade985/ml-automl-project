from pycaret.classification import *
import pandas as pd
import mlflow
import mlflow.sklearn

# Load dataset
data = pd.read_csv("data/dataset.csv")

# Remove spaces from column names
data.columns = data.columns.str.strip()

print(data.head())
print(data['species'].value_counts())

# Setup
clf = setup(
    data=data,
    target='species',
    session_id=123,
    verbose=False
)

# Train only stable models
best_model = compare_models(
    include=['lr', 'dt', 'rf', 'knn']
)

print(best_model)

# Finalize
final_model = finalize_model(best_model)

# Save model
save_model(final_model, "best_iris_model")

print("Model Trained Successfully")

# MLflow
mlflow.set_experiment("Iris_AutoML_Experiment")

with mlflow.start_run():

    mlflow.log_param("Tool", "PyCaret")

    mlflow.sklearn.log_model(
        sk_model=final_model,
        artifact_path="model"
    )

print("Experiment Logged in MLflow")