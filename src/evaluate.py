"""
Оценка моделей с кросс-валидацией
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

from src.train import get_model


def grid_search_models(X_train, y_train, config):
    """Подбор гиперпараметров для Random Forest и CatBoost"""
    results = {}
    
    # Random Forest
    print("GridSearch для Random Forest...")
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }
    rf = RandomForestClassifier(random_state=config.random_state, n_jobs=-1)
    grid_rf = GridSearchCV(rf, rf_params, cv=3, scoring='accuracy', n_jobs=-1)
    grid_rf.fit(X_train, y_train)
    results['random_forest'] = {
        'best_params': grid_rf.best_params_,
        'best_score': grid_rf.best_score_
    }
    print(f"  RF best: {grid_rf.best_params_} -> {grid_rf.best_score_:.4f}")
    
    # CatBoost
    if CatBoostClassifier is not None:
        print("GridSearch для CatBoost...")
        cb_params = {
            'iterations': [100, 300, 500],
            'learning_rate': [0.03, 0.1, 0.3],
            'depth': [4, 6, 8]
        }
        cb = CatBoostClassifier(verbose=False, random_state=config.random_state)
        grid_cb = GridSearchCV(cb, cb_params, cv=3, scoring='accuracy', n_jobs=-1)
        grid_cb.fit(X_train, y_train)
        results['catboost'] = {
            'best_params': grid_cb.best_params_,
            'best_score': grid_cb.best_score_
        }
        print(f"  CB best: {grid_cb.best_params_} -> {grid_cb.best_score_:.4f}")
    else:
        print("  CatBoost не установлен, пропускаем GridSearch для CatBoost")
    
    return results


def compare_k_folds(X_train, y_train, config):
    """Сравнение кросс-валидации с 1 и 5 фолдами"""
    model_names = ["logistic_regression", "random_forest", "catboost"]
    results = []
    
    for k in [1, 5]:
        if k == 1:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_train, y_train, test_size=0.2, random_state=config.random_state, stratify=y_train
            )
            for name in model_names:
                model = get_model(name, config)
                if model is None:
                    continue
                model.fit(X_tr, y_tr)
                score = model.score(X_te, y_te)
                results.append({'k': 1, 'model': name, 'score': score})
        else:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.random_state)
            for name in model_names:
                model = get_model(name, config)
                if model is None:
                    continue
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=config.scoring)
                results.append({'k': 5, 'model': name, 'score': np.mean(scores)})
    
    df = pd.DataFrame(results)
    print("\nСравнение 1 vs 5 фолдов:")
    print(df.pivot(index='model', columns='k', values='score'))
    return df


def cross_validate_models(X_train, y_train, config):
    """Кросс-валидация всех моделей"""
    cv = StratifiedKFold(
        n_splits=config.n_folds,
        shuffle=config.shuffle,
        random_state=config.random_state
    )
    
    model_names = [
        "logistic_regression", "ridge", "lasso", "elasticnet",
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
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=config.scoring)
        elapsed = time.time() - start
        
        results.append({
            "name": name,
            "mean": np.mean(scores),
            "std": np.std(scores),
            "min": np.min(scores),
            "max": np.max(scores),
            "time": elapsed
        })
        
        print(f"CV: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
        
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    df = pd.DataFrame(results)
    best_idx = df["mean"].idxmax()
    
    return {
        "results": df,
        "best_name": df.loc[best_idx, "name"],
        "best_score": df.loc[best_idx, "mean"],
        "best_std": df.loc[best_idx, "std"],
        "trained_models": trained_models
    }