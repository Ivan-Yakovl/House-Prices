"""
Обучение моделей
"""

import time
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None


def get_model(name: str, config):
    """Создание модели по имени"""
    
    if name == "logistic_regression":
        return LogisticRegression(
            C=config.lr_C,
            max_iter=config.lr_max_iter,
            solver=config.lr_solver,
            random_state=config.random_state
        )
    

    elif name == "ridge":
        return RidgeClassifier(
            alpha=config.ridge_alpha,
            random_state=config.random_state
        )
    elif name == "lasso":
        return LogisticRegression(
            penalty='l1',
            solver='saga',
            C=config.lasso_C,
            max_iter=config.lr_max_iter,
            random_state=config.random_state
        )
    elif name == "elasticnet":
        return LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            C=config.en_C,
            l1_ratio=config.en_l1_ratio,
            max_iter=config.lr_max_iter,
            random_state=config.random_state
        )
    elif name == "knn":
        return KNeighborsClassifier(
            n_neighbors=config.knn_n_neighbors,
            weights=config.knn_weights,
            metric=config.knn_metric
        )
    elif name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=config.dt_max_depth,
            min_samples_split=config.dt_min_samples_split,
            random_state=config.random_state
        )
    elif name == "random_forest":
        return RandomForestClassifier(
            n_estimators=config.rf_n_estimators,
            max_depth=config.rf_max_depth,
            min_samples_split=config.rf_min_samples_split,
            min_samples_leaf=config.rf_min_samples_leaf,
            random_state=config.random_state,
            n_jobs=-1
        )
    elif name == "catboost" and CatBoostClassifier:
        return CatBoostClassifier(
            iterations=config.cb_iterations,
            learning_rate=config.cb_learning_rate,
            depth=config.cb_depth,
            verbose=config.cb_verbose,
            early_stopping_rounds=config.cb_early_stopping_rounds,
            random_state=config.random_state
        )
    elif name == "xgboost" and XGBClassifier:
        return XGBClassifier(
            n_estimators=config.xgb_n_estimators,
            learning_rate=config.xgb_learning_rate,
            max_depth=config.xgb_max_depth,
            subsample=config.xgb_subsample,
            colsample_bytree=config.xgb_colsample_bytree,
            random_state=config.random_state,
            verbosity=0,
            n_jobs=-1
        )
    elif name == "lightgbm" and LGBMClassifier:
        return LGBMClassifier(
            n_estimators=config.lgb_n_estimators,
            learning_rate=config.lgb_learning_rate,
            num_leaves=config.lgb_num_leaves,
            max_depth=config.lgb_max_depth,
            random_state=config.random_state,
            n_jobs=-1,
            verbose=-1
        )
    return None


def train_models(X_train, y_train, config, model_names=None):
    """Обучает все модели из списка"""
    if model_names is None:
        model_names = [
            "logistic_regression", "ridge", "lasso", "elasticnet",
            "knn", "decision_tree", "random_forest",
            "catboost", "xgboost", "lightgbm"
        ]
    
    results = {}
    for name in model_names:
        model = get_model(name, config)
        if model is None:
            continue
        start = time.time()
        model.fit(X_train, y_train)
        results[name] = (model, time.time() - start)
    return results