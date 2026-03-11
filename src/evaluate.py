"""Evaluation metrics for BrandX Sales Forecasting."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> dict:
    """Print and return MAE, RMSE, MAPE for a given model."""
    metrics = {
        "Model"    : model_name,
        "MAE"      : round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE"     : round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "MAPE (%)" : round(mape(y_true, y_pred), 2),
    }
    print(f"\n📊 {model_name} Results:")
    print(f"   MAE      : ₹{metrics['MAE']:,.2f}")
    print(f"   RMSE     : ₹{metrics['RMSE']:,.2f}")
    print(f"   MAPE     : {metrics['MAPE (%)']:.2f}%")
    return metrics