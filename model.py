import os
import joblib
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from utils.preprocess import preprocess
from utils.evaluation import print_metrics, eval_metrics, plot_residuals, plot_feature_importances, plot_actual_vs_predicted, plot_error_distribution, plot_shap_summary

class CrowdCast:
    """ Model class for predicting attendance using XGBoost and parameters found from GridSearchCV. """

    def __init__(self, params=None):
        """
            Initializes the CrowdCast model with default parameters for XGBoost found with GridSearchCV
            hyperparameter tuning.If params are provided, they will override the default parameters.
        """
        default_params = {
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_estimators': 300,
            'max_depth': 7,
            'learning_rate': 0.1,
            'gamma': 0,
            'min_child_weight': 15,
            'reg_alpha': 1,
            'reg_lambda': 5
        }
        self.params = params if params else default_params
        self.model = XGBRegressor(**self.params)


    def fit(self, X_train, y_train):
        """
            Fits the XGBoost model to the training data (X_train, y_train) after preprocessing.
        """
        self.X_train = X_train.copy()
        self.y_train = y_train

        # preprocess the training data
        self.X_train_processed = preprocess(X_train.copy(), model="XGBoost", should_scale=False)

        # fit self.model
        self.model.fit(self.X_train_processed, self.y_train, verbose=False)


    def evaluate(self, X, y, dataset_type="Test"):
        """
            Evaluates the model on a given dataset (X, y) and prints metrics and generates plots of 
            residuals, actual vs predicted, error distribution, feature importances, and SHAP summary.
        """
        # preprocess the the feature data and make predictions
        X_processed = preprocess(X.copy(), model="XGBoost", should_scale=False)
        X_processed = X_processed.reindex(columns=self.X_train_processed.columns, fill_value=0)
        y_pred = self.model.predict(X_processed)

        # get the full set of features and target based on training data used at fit time and the X and y to be evaluated
        full_features = pd.concat([X, self.X_train]) 
        full_features_processed = preprocess(full_features.copy(), model="XGBoost", should_scale=False)
        full_features_processed = full_features_processed.reindex(columns=self.X_train_processed.columns, fill_value=0)
        full_y = pd.concat([y, self.y_train])

        # calculate and print evaluation metrics including cross-validation
        print_metrics(eval_metrics(y, y_pred), model_type="CrowdCast", dataset_type=dataset_type)
        scores = cross_val_score(self.model, full_features_processed, full_y, cv=5, scoring='r2', n_jobs=-1)
        print("Cross-Validation R^2: {:.2f} ± {:.2f}".format(scores.mean(), scores.std()))

        # plot residuals, actual vs predicted, error distribution, feature importances, and SHAP summary
        plot_residuals(y, y_pred, model_name="CrowdCast", path="./plots/CrowdCast")
        plot_actual_vs_predicted(y, y_pred, model_name="CrowdCast", path="./plots/CrowdCast")
        plot_error_distribution(y, y_pred, model_name="CrowdCast", path="./plots/CrowdCast")
        plot_feature_importances(self.model, full_features_processed.columns, 
                                 top_n=len(full_features_processed.columns), 
                                 model_name="CrowdCast", path="./plots/CrowdCast")
        plot_shap_summary(self.model, self.X_train_processed, model_name="CrowdCast", path="./plots/CrowdCast")


    def predict(self, X):
        """
            Predicts attendance for a given set of features X.
        """
        # preprocess the feature data, reindex it, and make predictions 
        X_processed = preprocess(X.copy(), model="XGBoost", should_scale=False)
        X_processed = X_processed.reindex(columns=self.X_train_processed.columns, fill_value=0)
        return self.model.predict(X_processed)


    def save(self, path="./CrowdCast_model.pkl"):
        """
            Saves the trained model to a file at the given path.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)


    def load(self, path="./CrowdCast_model.pkl"):
        """
            Loads a trained model from a file at the given path.
        """
        self.model = joblib.load(path)
