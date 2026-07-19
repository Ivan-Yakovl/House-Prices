"""
Модуль для создания ансамблей и стекинга
"""

import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression


def create_stacking(X_train, y_train, X_test, base_models, config):
    """
    Создание стекинга через логистическую регрессию

    Args:
        X_train: признаки обучения
        y_train: целевая переменная
        X_test: признаки теста
        base_models: словарь {имя: модель} для базовых моделей
        config: конфигурация

    Returns:
        preds: предсказанные вероятности для теста
    """
    estimators = [(name, model) for name, model in base_models.items()]
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=1.0, random_state=config.random_state),
        cv=5,
        n_jobs=-1
    )
    stack.fit(X_train, y_train)
    preds = stack.predict_proba(X_test)[:, 1]
    return preds


def create_ensembles(X_train, y_train, X_test, trained_models, cv_results, config, extra_preds=None):
    results = {}
    test_preds = []
    model_names = []
    
    # 1. Собираем предсказания от обученных моделей
    for name, model in trained_models.items():
        try:
            if hasattr(model, "predict_proba"):
                pred = model.predict_proba(X_test)[:, 1]
            else:
                pred = model.predict(X_test).astype(float)
            test_preds.append(pred)
            model_names.append(name)
        except Exception as e:
            print(f"  Предупреждение: модель {name} не может предсказать вероятности: {e}")
            # пробуем просто predict
            try:
                pred = model.predict(X_test).astype(float)
                test_preds.append(pred)
                model_names.append(name)
            except:
                print(f"  Ошибка: модель {name} не может предсказать: {e}")
    
    # 2. Добавляем дополнительные предсказания (например, DNN)
    if extra_preds:
        for name, preds in extra_preds.items():
            test_preds.append(np.array(preds).flatten())
            model_names.append(name)
    
    if not test_preds:
        raise ValueError("Нет ни одной модели для ансамбля")
    
    test_preds = np.column_stack(test_preds)
    
    # 3. Soft voting (усреднение вероятностей)
    if "simple" in config.ensemble_methods:
        results["simple_average"] = {
            "test_pred": np.mean(test_preds, axis=1),
            "description": "Simple average (soft voting)"
        }
    
    # 4. Weighted average
    if "weighted" in config.ensemble_methods:
        weights = []
        for name in model_names:
            row = cv_results["results"][cv_results["results"]["name"] == name]
            if not row.empty:
                weights.append(row["mean"].values[0])
            else:
                weights.append(1.0)  # для DNN нет CV скора, даём вес 1
        weights = np.array(weights)
        weights = weights / weights.sum()
        weighted_avg = np.sum(test_preds * weights.reshape(1, -1), axis=1)
        results["weighted_average"] = {
            "test_pred": weighted_avg,
            "description": "Weighted average by CV (soft voting)"
        }
    
    # 5. Hard voting (голосование по классам)
    # Превращаем вероятности в бинарные классы (>=0.5)
    binary_preds = (test_preds >= 0.5).astype(int)
    # Голосование: выбираем класс с большинством голосов
    hard_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=binary_preds)
    results["hard_voting"] = {
        "test_pred": hard_vote.astype(float),  # для единообразия
        "description": "Hard voting (majority vote)"
    }
    
    best_key = "simple_average" if "simple_average" in results else list(results.keys())[0]
    results["best_name"] = best_key
    results["best_pred"] = results[best_key]["test_pred"]
    
    return results