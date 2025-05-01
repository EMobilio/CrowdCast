import pandas as pd
from model import CrowdCast


def test_pipeline_runs():
    """
        Test that the pipeline runs end-to-end without errors and produces predictions.
    """
    # load a small subset of the data
    df = pd.read_csv("data/MLB_games_2000-2024.csv").sample(n=200, random_state=42)

    # split into train/test
    target = df["attendance"]
    features = df.drop(columns=["attendance"])
    X_train = features.iloc[:160]
    y_train = target.iloc[:160]
    X_test = features.iloc[160:]
    y_test = target.iloc[160:]

    # train model
    model = CrowdCast()
    model.fit(X_train, y_train)

    # ensure predictions run and return correct shape
    preds = model.predict(X_test)
    assert len(preds) == len(y_test), "Prediction length mismatch"

    print("Test passed: model runs and makes predictions.")
    

if __name__ == "__main__":
    test_pipeline_runs()