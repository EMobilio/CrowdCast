import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess import preprocess
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math


def train(game_data):
    """
        Takes a DataFrame containing game data and builds a linear regression model to predict attendance.
    """
    features = game_data.drop(columns=['attendance'])  
    target = game_data['attendance']

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # fit the model
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    # make predictions on train and test sets
    y_train_pred = lin_reg.predict(X_train)
    y_test_pred = lin_reg.predict(X_test)

    # callculate training set metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    # calculate test set metrics
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # training metrics
    print("Training Metrics:")
    print(f"Train MAE: {train_mae:.2f}")
    print(f"Train MSE: {train_mse:.2f}")
    print(f"Train RMSE: {math.sqrt(train_mse):.2f}")
    print(f"Train R² Score: {train_r2:.2f}")

    # test metrics
    print("\nTest Metrics:")
    print(f"Test MAE: {test_mae:.2f}")
    print(f"Test MSE: {test_mse:.2f}")
    print(f"Test RMSE: {math.sqrt(test_mse):.2f}")
    print(f"Test R² Score: {test_r2:.2f}")


def main():
    """
        Performs preprocessing and training for a linear model.
    """
    game_data = pd.read_csv("data/MLB_games_2000-2024.csv")
    game_data = preprocess(game_data, model="linear")
    train(game_data)


if __name__ == "__main__":
    main()
