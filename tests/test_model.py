import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocess import load_and_preprocess

def test_data_loads():
    X_train, X_test, y_train, y_test = load_and_preprocess("data/IMDB_Data.csv")
    assert len(X_train) > 0
    assert len(X_test) > 0

def test_labels_are_binary():
    X_train, X_test, y_train, y_test = load_and_preprocess("data/IMDB_Data.csv")
    assert set(y_train.unique()).issubset({0, 1})