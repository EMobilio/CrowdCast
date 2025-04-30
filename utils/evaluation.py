import math
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap

def eval_metrics(y_true, y_pred):
    """
        Calculate and return evaluation metrics for the model predictions.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2
    }

    return metrics


def print_metrics(metrics, model_type, dataset_type):
    """
        Print the evaluation metrics for a given dataset.
    """
    print(f"{model_type} {dataset_type} Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")


def plot_residuals(y_true, y_pred, model_name):
    """
        Generate a residual plot to visualize the difference between the true and predicted values.
    """
    residuals = y_true - y_pred

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, color='blue', alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'Residual Plot - {model_name}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.savefig(f"plots/residual_plots/{model_name}_residuals.png", dpi=300, bbox_inches="tight")


def plot_feature_importances(model, feature_names, model_name, top_n=50):
    """
        Plots the top N feature importances from a model.
    """
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.gca().invert_yaxis()
    plt.title("Feature Importances")
    plt.xlabel("Importance")
    plt.savefig(f"plots/features/{model_name}_feature_importances.png", dpi=300, bbox_inches="tight")


def plot_actual_vs_predicted(y_true, y_pred, model_name, title="Actual vs. Predicted Attendance"):
    """
        Plots the actual vs predicted values of a model to visualize performance.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual Attendance")
    plt.ylabel("Predicted Attendance")
    plt.title(title)
    plt.savefig(f"plots/{model_name}_actual_vs_predicted.png", dpi=300, bbox_inches="tight")


def plot_error_distribution(y_true, y_pred, model_name):
    """
        Plots the distribution of prediction errors of a model.
    """
    errors = y_pred - y_true
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=50, edgecolor='k')
    plt.title("Distribution of Prediction Errors")
    plt.xlabel("Error")
    plt.ylabel("Count")
    plt.grid(True)
    plt.savefig(f"plots/{model_name}_error_distribution.png", dpi=300, bbox_inches="tight")


def plot_shap_summary(model, X_train, model_name):
    """
        Generates a SHAP summary plot for a model showing the importances and effects of features.
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train, show=False)
    plt.xlabel("SHAP value (impact on model output)")
    plt.ylabel("Feature")
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig(f"plots/features/{model_name}_shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close('all')