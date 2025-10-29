def register_model(model, model_name, n_estimators, accuracy):
    import mlflow

    # Configuración de MLflow
    mlflow.set_tracking_uri("http://localhost:5000")

    # Creación del experimento
    mlflow.set_experiment("Mi primer Modelo")

    # Inicio de la sesión
    with mlflow.start_run():
        # Log de parámetros
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("n_estimators", n_estimators)
        # Log de métricas
        mlflow.log_metric("accuracy", accuracy)
        # Log del modelo
        mlflow.sklearn.log_model(model, "model")