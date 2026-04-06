import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from preprocess import load_and_preprocess

mlflow.set_experiment("IMDB_Sentiment")

DATA_PATH = "data/IMDB_Data.csv"
X_train, X_test, y_train, y_test = load_and_preprocess(DATA_PATH)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "NaiveBayes": MultinomialNB()
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("clf", model)
        ])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        mlflow.log_param("model", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(pipeline, "model")

        print(f"{name} -> Accuracy: {acc:.4f}, F1: {f1:.4f}")