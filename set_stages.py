import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = mlflow.tracking.MlflowClient()

client.set_registered_model_alias("IMDB_Sentiment_Model", "Staging", "1")
client.set_registered_model_alias("IMDB_Sentiment_Model", "Production", "2")

print("Staging set to Version 1")
print("Production set to Version 2")