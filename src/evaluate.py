import mlflow

def get_best_run():
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("IMDB_Sentiment")
    runs = client.search_runs(experiment.experiment_id,
                               order_by=["metrics.accuracy DESC"])
    best = runs[0]
    print(f"Best Run: {best.info.run_id}")
    print(f"Accuracy: {best.data.metrics['accuracy']:.4f}")
    return best

if __name__ == "__main__":
    get_best_run()