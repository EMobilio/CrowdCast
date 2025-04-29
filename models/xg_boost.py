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

    # split the data into train, validation, and test
    X_full_train, X_test, y_full_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_full_train, y_full_train, test_size=0.2, random_state=42) 

    # create the model
    xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

    # param grid for Grid Search
    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [5, 7, 10],
        'learning_rate': [0.05, 0.1],
        'gamma': [0, 1],
        'min_child_weight': [10, 15],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 5, 10]
    }

    # grid search with 5-fold CV
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    print(f"\nBest parameters: {grid_search.best_params_}\n")

    # fit the model with the best params with early stopping
    best_params = grid_search.best_params_
    best_model = XGBRegressor(
        **best_params,
        objective='reg:squarederror',
        early_stopping_rounds=20,
        random_state=42
    )
    best_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    best_model.save_model('xgboost_model.json')

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