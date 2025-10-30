# filepath: /ml-pipeline-project/ml-pipeline-project/src/pipelineml.py
import pandas as pd
from data_loader import load_data
from data_preparation import prepare_data
from model_trainer import train_model
from model_registry import register_model

def main():
    # Load data - cambie la ruta de la data
    data = load_data("C:/Users/LEGION Y540/UTEC/laboratorio_01/introduction-to-mlflow-grupo3/data/in/application_data.csv")
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(data)
    
    # Train model
    model, accuracy = train_model(X_train, y_train, X_test, y_test)
    n_estimators = model.n_estimators
    model_name = "RandomForestClassifier"

    # Register model with MLflow
    register_model(model, model_name, n_estimators, accuracy)

if __name__ == "__main__":
    main()
