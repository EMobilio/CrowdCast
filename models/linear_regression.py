import pandas as pd
import sys
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess import preprocess
from utils.evaluation import print_metrics, eval_metrics


def train(game_data):
    """
        Takes a DataFrame containing game data and builds a linear regression model to predict attendance.
    """
    # split predictors and outcome
    target = game_data['attendance']
    features = game_data.drop(columns=['attendance'])

    # preprocess data
    features = preprocess(features, model="linear")

    lin_reg = LinearRegression()

    # perform cross validation
    cv_scores = cross_val_score(lin_reg, features, target, cv=5, scoring='r2', n_jobs=-1)

    print("Cross Validation Score: {:0.5} ± {:0.5}".format(cv_scores.mean().round(5), cv_scores.std().round(5)))

    # split into train and test sets and make predictions
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    lin_reg.fit(X_train, y_train)
    y_train_pred = lin_reg.predict(X_train)
    y_test_pred = lin_reg.predict(X_test)

    # calculate and print metrixs
    print_metrics(eval_metrics(y_train, y_train_pred), dataset_type='Train')
    print("\n")
    print_metrics(eval_metrics(y_test, y_test_pred), dataset_type='Test')


def main():
    """
        Performs preprocessing and training for a linear model.
    """
    game_data = pd.read_csv("data/MLB_games_2000-2024.csv")
    train(game_data)


if __name__ == "__main__":
    main()
