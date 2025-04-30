import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
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
    features = game_data.drop(columns=["attendance"])

    # split the data into train, validation, and test
    X_full_train, X_test, y_full_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_full_train, y_full_train, test_size=0.2, random_state=42)

    # preprocess each set independently
    X_train_processed = preprocess(X_train.copy(), model="XGBoost")
    X_val_processed = preprocess(X_val.copy(), model="XGBoost")
    X_test_processed = preprocess(X_test.copy(), model="XGBoost")

    # align val and test columns with training
    X_val_processed = X_val_processed.reindex(columns=X_train_processed.columns, fill_value=0)
    X_test_processed = X_test_processed.reindex(columns=X_train_processed.columns, fill_value=0)

    # create the model
    xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

    # grid search with 5-fold CV
    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [7, 10],
        'learning_rate': [0.01, 0.1],
        'gamma': [0, 1],
        'min_child_weight': [15, 20],
        'reg_alpha': [0.1, 1],
        'reg_lambda': [5, 10],
    }
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train_processed, y_train)
    print(f"\nBest parameters: {grid_search.best_params_}\n")

    # fit the model with the best params
    best_params = grid_search.best_params_
    best_model = XGBRegressor(**best_params, objective='reg:squarederror', random_state=42)
    best_model.fit(X_train_processed, y_train, eval_set=[(X_val_processed, y_val)], verbose=False)
    best_model.save_model('xgboost_model.json')

    # make predictions
    y_train_pred = best_model.predict(X_train_processed)
    y_test_pred = best_model.predict(X_test_processed)

    # calculate and print evaluation metrics
    print_metrics(eval_metrics(y_train, y_train_pred), model_type="XGBoost", dataset_type="Train")
    print()
    print_metrics(eval_metrics(y_test, y_test_pred), model_type="XGBoost", dataset_type="Test")
    print()

    # plot residuals
    plot_residuals(y_test, y_test_pred, model_name="XGBoost")

    # perform cross-validation on full preprocessed set
    X_all = pd.concat([X_train, X_val, X_test])
    y_all = pd.concat([y_train, y_val, y_test])
    X_all_processed = preprocess(X_all.copy(), model="XGBoost")
    scores = cross_val_score(best_model, X_all_processed, y_all, cv=5, scoring='r2', n_jobs=-1)
    print("Cross-Validation R^2: {:.2f} ± {:.2f}".format(scores.mean(), scores.std()))

    return best_model


def main():
    """
        Performs preprocessing and training for an XGBoost model.
    """
    game_data = pd.read_csv("data/MLB_games_2000-2024.csv")
    train(game_data)

    
if __name__ == "__main__":
    main()