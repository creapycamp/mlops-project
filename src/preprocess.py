import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    df = df.dropna()
    X = df['review']
    y = df['sentiment']
    return train_test_split(X, y, test_size=0.2, random_state=42)