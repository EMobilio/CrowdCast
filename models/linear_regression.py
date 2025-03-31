import sys
import os
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess import preprocess
from utils.evaluation import print_metrics, eval_metrics, plot_residuals



def train(game_data):
    """
        Takes a DataFrame containing game data and builds a linear regression model to predict attendance.
    """
    # split predictors and outcome
    target = game_data["attendance"]
    features = game_data.drop(columns=["attendance", "stadium"])

    # preprocess features
    features = preprocess(features, model="linear")

    # create the models
    lin_reg = LinearRegression()
    ridge_reg = Ridge(alpha=1) 
    lasso_reg = Lasso(alpha=10)

    # split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # perform GridSearchCV to tune hyperparameters
    ridge_params = {'alpha': [0.1, 1, 10, 100, 200]}
    lasso_params = {'alpha': [0.1, 1, 10, 100, 200]}
    ridge_search = GridSearchCV(ridge_reg, ridge_params, cv=5, scoring='neg_mean_squared_error')
    lasso_search = GridSearchCV(lasso_reg, lasso_params, cv=5, scoring='neg_mean_squared_error')

    # fit models to find best parameters
    ridge_search.fit(X_train, y_train)
    lasso_search.fit(X_train, y_train)
    ridge_best = ridge_search.best_estimator_
    lasso_best = lasso_search.best_estimator_

    # train all models
    for model, name in [(lin_reg, "Linear_Regression"), 
                        (ridge_best, "Ridge_Regression"), 
                        (lasso_best, "Lasso_Regression")]:
        # fit the model and make predictions
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # calculate and print metrics
        print_metrics(eval_metrics(y_train, y_train_pred), model_type=name, dataset_type='Train')
        print("\n")
        print_metrics(eval_metrics(y_test, y_test_pred), model_type=name, dataset_type='Test')
        print("\n")

        # plot residuals
        plot_residuals(y_test, y_test_pred, model_name=name)

        # plotting for the tuning parameters
        if model == ridge_best or model == lasso_best:
            params = ridge_params if model == ridge_best else lasso_params
            cv_results = ridge_search.cv_results_ if model == ridge_best else lasso_search.cv_results_

            plt.figure(figsize=(10, 6))
            plt.plot(params['alpha'], -cv_results['mean_test_score'], label=f'{name} - MSE')
            plt.title(f"Model Performance vs Alpha")
            plt.xlabel("Alpha")
            plt.ylabel("Negative MSE")
            plt.legend()
            plt.xscale('log')
            plt.savefig(f"plots/parameter_plots/{name}_parameters.png", dpi=300, bbox_inches="tight")


def main():
    """
        Performs preprocessing and training for a linear model.
    """
    game_data = pd.read_csv("data/MLB_games_2000-2024.csv")
    train(game_data)


if __name__ == "__main__":
    main()
