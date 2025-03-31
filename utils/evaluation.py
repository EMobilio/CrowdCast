import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

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
