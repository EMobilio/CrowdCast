import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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


def print_metrics(metrics, dataset_type='Train'):
    """
        Print the evaluation metrics for a given dataset (Train or Test).
    """
    print(f"{dataset_type} Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
