"""
Оценка моделей с кросс-валидацией (регрессия)
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from src.train import get_model

def rmse_scorer(y_true, y_pred):
    """Собственный scorer для RMSE"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def cross_validate_models(X_train, y_train, config):
    """Кросс-валидация всех моделей для регрессии"""
    cv = KFold(
        n_splits=config.n_folds,
        shuffle=config.shuffle,
        random_state=config.random_state
    )
    
    model_names = [
        "ridge", "lasso", "elasticnet",
        "knn", "decision_tree", "random_forest",
        "catboost", "xgboost", "lightgbm"
    ]
    
    results = []
    trained_models = {}
    
    for name in model_names:
        model = get_model(name, config)
        if model is None:
            continue
        
        print(f"  {name}...", end=" ", flush=True)
        
        start = time.time()
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring=make_scorer(rmse_scorer, greater_is_better=False),
            n_jobs=config.n_jobs
        )
        elapsed = time.time() - start
        
        # scores отрицательные, т.к. make_scorer с greater_is_better=False
        mean_rmse = -np.mean(scores)
        std_rmse = np.std(scores)
        
        results.append({
            "name": name,
            "mean": mean_rmse,
            "std": std_rmse,
            "min": -np.max(scores),
            "max": -np.min(scores),
            "time": elapsed
        })
        
        print(f"RMSE: {mean_rmse:.4f} (+/- {std_rmse:.4f})")
        
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    df = pd.DataFrame(results)
    best_idx = df["mean"].idxmin()  # минимальная RMSE
    
    return {
        "results": df,
        "best_name": df.loc[best_idx, "name"],
        "best_score": df.loc[best_idx, "mean"],
        "best_std": df.loc[best_idx, "std"],
        "trained_models": trained_models
    }