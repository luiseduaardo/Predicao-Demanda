import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def to_units(y_pred_log: np.ndarray) -> np.ndarray:
    """Reescala de log1p pra unidades."""
    return np.clip(np.expm1(y_pred_log), 0, None)


def regression_metrics(y_true_raw, y_pred_log, y_true_log) -> dict:
    y_true_raw = np.asarray(y_true_raw, dtype=float)
    y_pred_raw = to_units(np.asarray(y_pred_log, dtype=float))

    return {
        "rmse_log": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "mae_log": float(mean_absolute_error(y_true_log, y_pred_log)),
        "r2_log": float(r2_score(y_true_log, y_pred_log)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))),
        "mae": float(mean_absolute_error(y_true_raw, y_pred_raw)),
        "r2": float(r2_score(y_true_raw, y_pred_raw))
    }
