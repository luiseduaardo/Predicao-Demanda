from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
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
    "hgb": {
        "criar": lambda p: HistGradientBoostingRegressor(
            random_state=SEED,
            **p,
        ),
        "params": {
            "max_depth": 3,
            "learning_rate": 0.05,
            "max_iter": 300,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
        },
        "espaco": lambda t: {
            "max_depth": t.suggest_int("max_depth", 2, 4),
            "learning_rate": t.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "max_iter": t.suggest_int("max_iter", 100, 800, step=50),
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 5, 60),
            "l2_regularization": t.suggest_float(
                "l2_regularization", 1e-3, 10, log=True
            ),
        },
    },
    "random_forest": {
        "criar": lambda p: RandomForestRegressor(
            random_state=SEED,
            n_jobs=-1,
            **p,
        ),
        "params": {
            "n_estimators": 500,
            "max_depth": 8,
            "max_features": 0.5,
        },
        "espaco": lambda t: {
            "n_estimators": t.suggest_int("n_estimators", 200, 1000, step=100),
            "max_depth": t.suggest_categorical("max_depth", [4, 6, 8, 12]),
            "max_features": t.suggest_float("max_features", 0.2, 1.0),
        },
    }
}
