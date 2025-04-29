import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess import preprocess
from utils.evaluation import print_metrics, eval_metrics, plot_residuals

def train(game_data):
    """
        Takes a DataFrame containing game data and builds a random forest model to predict attendance.
    """
    # split predictors and outcome
    target = game_data["attendance"]
    features = game_data.drop(columns=["attendance", "stadium"])

    # preprocess features
    features = preprocess(features, model="random_forest")

    # split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # create the model
    rf = RandomForestRegressor(random_state=42)

    # param grid for Grid Search
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # grid search with 5-fold CV
    grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error', verbose=1)
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    print(f"\nBest parameters: {grid_search.best_params_}\n")

    # make predictions
    y_train_pred = best_rf.predict(X_train)
    y_test_pred = best_rf.predict(X_test)

    # calculate and print evaluation metrics
    print_metrics(eval_metrics(y_train, y_train_pred), model_type="Random Forest", dataset_type="Train")
    print()
    print_metrics(eval_metrics(y_test, y_test_pred), model_type="Random Forest", dataset_type="Test")
    print()

    # plot residuals
    plot_residuals(y_test, y_test_pred, model_name="Random_Forest")

    return best_rf


def main():
    """
        Performs preprocessing and training for a random forest model.
    """
    game_data = pd.read_csv("data/MLB_games_2000-2024.csv")
    train(game_data)

    
if __name__ == "__main__":
    main()