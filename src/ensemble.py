"""
Модуль для создания ансамблей
"""

import numpy as np

def create_ensembles(X_train, y_train, X_test, trained_models, cv_results, config, extra_preds=None):
    """Создаёт ансамбли для регрессии (simple average, weighted average)"""
    results = {}
    test_preds = []
    model_names = []
    
    for name, model in trained_models.items():
        try:
            pred = model.predict(X_test)
            test_preds.append(pred)
            model_names.append(name)
        except Exception as e:
            print(f"  Предупреждение: модель {name} не может предсказать: {e}")
    
    if extra_preds:
        for name, preds in extra_preds.items():
            test_preds.append(np.array(preds).flatten())
            model_names.append(name)
    
    if not test_preds:
        raise ValueError("Нет ни одной модели для ансамбля")
    
    test_preds = np.column_stack(test_preds)
    
    if "simple" in config.ensemble_methods:
        results["simple_average"] = {
            "test_pred": np.mean(test_preds, axis=1),
            "description": "Simple average"
        }
    
    if "weighted" in config.ensemble_methods:
        weights = []
        for name in model_names:
            row = cv_results["results"][cv_results["results"]["name"] == name]
            if not row.empty:
                weights.append(1.0 / row["mean"].values[0])  # обратная RMSE
            else:
                weights.append(1.0)
        weights = np.array(weights)
        weights = weights / weights.sum()
        weighted_avg = np.sum(test_preds * weights.reshape(1, -1), axis=1)
        results["weighted_average"] = {
            "test_pred": weighted_avg,
            "description": "Weighted average by RMSE"
        }
    
    best_key = "simple_average" if "simple_average" in results else list(results.keys())[0]
    results["best_name"] = best_key
    results["best_pred"] = results[best_key]["test_pred"]
    
    return results