import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess import preprocess
from utils.evaluation import print_metrics, eval_metrics, plot_residuals


def train(game_data):
    """
        Takes a DataFrame containing game data and builds a tuned XGBoost model to predict attendance.
    """
    # split predictors and outcome
    target = game_data["attendance"]
    features = game_data.drop(columns=["attendance", "stadium"])

    # preprocess features
    features = preprocess(features, model="XGBoost")

    # split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # create the model
    xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

    # param grid for Grid Search
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # grid search with 5-fold CV
    grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"\nBest parameters: {grid_search.best_params_}\n")

    # make predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # calculate and print evaluation metrics
    print_metrics(eval_metrics(y_train, y_train_pred), model_type="XGBoost", dataset_type="Train")
    print()
    print_metrics(eval_metrics(y_test, y_test_pred), model_type="XGBoost", dataset_type="Test")
    print()

    # plot residuals
    plot_residuals(y_test, y_test_pred, model_name="XGBoost")

    return best_model


def main():
    """
        Performs preprocessing and training for an XGBoost model.
    """
    game_data = pd.read_csv("data/MLB_games_2000-2024.csv")
    train(game_data)

    
if __name__ == "__main__":
    main()