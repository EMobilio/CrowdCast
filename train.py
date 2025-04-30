import pandas as pd
from sklearn.model_selection import train_test_split
from model import CrowdCast


def train():
    # load and prepare data
    data = pd.read_csv("data/MLB_games_2000-2024.csv")

    # split predictors and target
    X = data.drop(columns=["attendance"])
    y = data["attendance"]

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # instantiate and train the CrowdCast model
    model = CrowdCast()
    model.fit(X_train, y_train)

    # evaluate on test data
    model.evaluate(X_test, y_test, dataset_type="Test")

    # save the model
    model.save("./CrowdCast_model.pkl")


if __name__ == "__main__":
    train()