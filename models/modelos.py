from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.linear_model import ElasticNet, LinearRegression


SEED = 42

# escolhe uma janela específica
def so_coluna(coluna, estimador):
    """Usa só uma coluna de X."""
    return make_pipeline(ColumnTransformer([("sel", "passthrough", [coluna])]), estimador)


# MODELOS ESCOLHIDOS
MODELOS = {
    # Modelo de baseline
    "baseline": {
        "criar": lambda p: so_coluna(
            p["coluna"], 
            LinearRegression()
        ),
        "params": {
            "coluna": "qtd_total_lag_1"
        },
    },

    # Baseline regularizado
    "elasticnet": {
        "criar": lambda p: ElasticNet(
            max_iter=20_000,
            **p
        ),
        "params": {
            "alpha": 0.1,
            "l1_ratio": 0.5
        },
        "espaco": lambda t: {
            "alpha": t.suggest_float("alpha", 1e-4, 10, log=True),
            "l1_ratio": t.suggest_float("l1_ratio", 0.05, 1.0),
        },
    },

    # Modelos baseados em árvores
    "xgb": {
        "criar": lambda p: XGBRegressor(
            tree_method = 'hist',
            device = 'cuda', # roda em gpu
            random_state = SEED,
            **p
        ),
        "params": {
            "max_depth": 3,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "min_child_weight": 20,
            "reg_lambda": 1.0,
        },
        "espaco": lambda t: {
            "max_depth": t.suggest_int("max_depth", 2, 4),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": t.suggest_int("n_estimators", 100, 800, step=50),
            "min_child_weight": t.suggest_int("min_child_weight", 5, 60),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True)
        },
    },
    "random_forest": {
        "criar": lambda p: XGBRFRegressor(
            tree_method = 'hist',
            device = 'cuda', # também roda em gpu
            random_state = SEED,
            **p
        ),
        "params": {
            "n_estimators": 500,
            "max_depth": 8,
            "colsample_bynode": 0.5,
        },
        "espaco": lambda t: {
            "n_estimators": t.suggest_int("n_estimators", 200, 1000, step=100),
            "max_depth": t.suggest_categorical("max_depth", [4, 6, 8, 12]),
            "colsample_bynode": t.suggest_float("colsample_bynode", 0.2, 1.0),
        },
    },
}
