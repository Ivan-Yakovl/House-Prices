"""
Главный скрипт проекта House Prices ML
Запуск: python main.py
"""

import os
import sys
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_config
from src.data_loader import load_train_data, load_test_data
from src.preprocessor import preprocess_data
from src.feature_engineering import build_features
from src.evaluate import cross_validate_models
from src.ensemble import create_ensembles
from src.dnn_model import train_dnn, predict_dnn
from src.train import get_model
from src.utils import setup_logger, save_results, save_submission_from_preds

warnings.filterwarnings("ignore")


def download_data():
    """Автоматически скачивает данные, если они отсутствуют"""
    os.makedirs("data", exist_ok=True)
    
    # Замените ссылки на актуальные, если нужно
    files = {
        "train.csv": "https://raw.githubusercontent.com/.../house-prices/train.csv",
        "test.csv": "https://raw.githubusercontent.com/.../house-prices/test.csv"
    }
    
    print("\n" + "="*60)
    print("Проверка наличия данных...")
    for filename, url in files.items():
        path = f"data/{filename}"
        if not os.path.exists(path):
            print(f"  Скачивание {filename}...")
            urllib.request.urlretrieve(url, path)
            print(f"    {filename} загружен")
        else:
            print(f"  {filename} уже существует")
    print("="*60 + "\n")


def main():
    download_data()
    config = get_config()
    logger = setup_logger()
    print("\n" + "="*60)
    logger.info("Запуск House Prices ML пайплайна")
    print("="*60 + "\n")

    # ---- Загрузка данных ----
    logger.info("Загрузка данных...")
    train_df = load_train_data(config.train_path)
    test_df = load_test_data(config.test_path)
    logger.info(f"Train: {train_df.shape}, Test: {test_df.shape}\n")

    # ---- Предобработка ----
    logger.info("Предобработка данных...")
    train_processed = preprocess_data(train_df, config, fit=True)
    test_processed = preprocess_data(test_df, config, fit=False)
    logger.info("Предобработка завершена\n")

    # ---- Создание признаков ----
    logger.info("Создание новых признаков...")
    X, y = build_features(train_processed, config, fit=True)
    X_test = build_features(test_processed, config, fit=False)[0]
    logger.info(f"Создано признаков: {X.shape[1]}\n")

    # ---- Разделение на train/val ----
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=config.random_state
    )
    logger.info(f"Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}\n")

    # ---- Обучение базовых моделей на X_train ----
    logger.info("Обучение базовых моделей на train...")
    model_names = ["ridge", "lasso", "elasticnet", "knn", "decision_tree", "random_forest", "catboost", "xgboost", "lightgbm"]
    trained_models_val = {}

    for name in model_names:
        model = get_model(name, config)
        if model is not None:
            model.fit(X_train, y_train)
            trained_models_val[name] = model
            logger.info(f"  {name} обучен")

    # ---- CatBoost с графиком RMSE ----
    logger.info("Обучение CatBoost на train (с графиком)...")
    cb_model = CatBoostRegressor(
        iterations=config.cb_iterations,
        learning_rate=config.cb_learning_rate,
        depth=config.cb_depth,
        verbose=False,
        early_stopping_rounds=config.cb_early_stopping_rounds,
        random_state=config.random_state,
        eval_metric='RMSE'
    )
    cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    trained_models_val['catboost'] = cb_model

    cb_history = cb_model.get_evals_result()
    train_rmse_cb = cb_history['learn']['RMSE']
    val_rmse_cb = cb_history['validation']['RMSE']

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_rmse_cb, label='Train RMSE')
    plt.plot(val_rmse_cb, label='Val RMSE')
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.title('CatBoost RMSE')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(val_rmse_cb, label='Val RMSE')
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.title('CatBoost Validation RMSE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    Path("outputs").mkdir(exist_ok=True)
    plt.savefig("outputs/catboost_learning_curves.png")
    plt.close()
    logger.info("  График CatBoost сохранён в outputs/catboost_learning_curves.png\n")

    # ---- Random Forest с графиком RMSE ----
    logger.info("Обучение Random Forest на train (с графиком)...")
    n_estimators_range = list(range(10, 210, 10))
    train_rmse_rf = []
    val_rmse_rf = []

    for n in n_estimators_range:
        rf = RandomForestRegressor(
            n_estimators=n,
            max_depth=config.rf_max_depth,
            min_samples_split=config.rf_min_samples_split,
            min_samples_leaf=config.rf_min_samples_leaf,
            random_state=config.random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        train_pred = rf.predict(X_train)
        val_pred = rf.predict(X_val)
        train_rmse_rf.append(np.sqrt(mean_squared_error(y_train, train_pred)))
        val_rmse_rf.append(np.sqrt(mean_squared_error(y_val, val_pred)))

    rf_final = RandomForestRegressor(
        n_estimators=config.rf_n_estimators,
        max_depth=config.rf_max_depth,
        min_samples_split=config.rf_min_samples_split,
        min_samples_leaf=config.rf_min_samples_leaf,
        random_state=config.random_state,
        n_jobs=-1
    )
    rf_final.fit(X_train, y_train)
    trained_models_val['random_forest'] = rf_final

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(n_estimators_range, train_rmse_rf, label='Train RMSE')
    plt.plot(n_estimators_range, val_rmse_rf, label='Val RMSE')
    plt.xlabel('Number of Trees')
    plt.ylabel('RMSE')
    plt.title('Random Forest RMSE')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(n_estimators_range, val_rmse_rf, label='Val RMSE')
    plt.xlabel('Number of Trees')
    plt.ylabel('RMSE')
    plt.title('RF Validation RMSE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/rf_learning_curves.png")
    plt.close()
    logger.info("  График Random Forest сохранён в outputs/rf_learning_curves.png\n")

    # ---- DNN на X_train ----
    logger.info("Обучение DNN на train...")
    dnn_model_val, dnn_history = train_dnn(
        X_train, y_train, X_val, y_val,
        config=config,
        return_history=True
    )
    dnn_preds_val = predict_dnn(dnn_model_val, X_val)
    dnn_rmse_val = np.sqrt(mean_squared_error(y_val, dnn_preds_val))
    logger.info(f"  DNN val RMSE: {dnn_rmse_val:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(dnn_history['train_loss'], label='Train Loss')
    if dnn_history.get('val_loss'):
        plt.plot(dnn_history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('DNN Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/dnn_loss_curves.png")
    plt.close()
    logger.info("  График DNN сохранён в outputs/dnn_loss_curves.png\n")

    # ---- Предсказания на X_val для ансамблей ----
    val_preds_dict = {}
    for name, model in trained_models_val.items():
        val_preds_dict[name] = model.predict(X_val)
    val_preds_dict['dnn'] = dnn_preds_val

    # ---- Оценка ансамблей на X_val ----
    logger.info("Оценка ансамблей на валидации...")
    ensemble_scores = {}

    # Simple average
    simple_avg = np.mean(list(val_preds_dict.values()), axis=0)
    ensemble_scores['simple_average'] = np.sqrt(mean_squared_error(y_val, simple_avg))

    # Weighted average (по обратной RMSE)
    weights = []
    for name in model_names:
        preds = val_preds_dict[name]
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        weights.append(1.0 / (rmse + 1e-8))
    weights.append(1.0 / (dnn_rmse_val + 1e-8))
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    all_preds = np.column_stack(list(val_preds_dict.values()))
    weighted_avg = np.sum(all_preds * weights.reshape(1, -1), axis=1)
    ensemble_scores['weighted_average'] = np.sqrt(mean_squared_error(y_val, weighted_avg))

    # Stacking (на базовых моделях, без DNN)
    logger.info("Оценка стекинга на валидации...")
    stack_models = {k: v for k, v in trained_models_val.items() if k != 'dnn'}
    estimators = [(name, model) for name, model in stack_models.items()]
    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1
    )
    stack.fit(X_train, y_train)
    stack_preds_val = stack.predict(X_val)
    ensemble_scores['stacking'] = np.sqrt(mean_squared_error(y_val, stack_preds_val))

    logger.info("Результаты на валидации (RMSE):")
    logger.info(f"  DNN (отдельно): {dnn_rmse_val:.4f}")
    for name, score in ensemble_scores.items():
        logger.info(f"  {name}: {score:.4f}")
    print("")

    # ---- Обучение финальных моделей на всех данных ----
    logger.info("Обучение финальных моделей на всех данных...")
    cv_results = cross_validate_models(X, y, config)
    trained_models = cv_results["trained_models"]

    # ---- DNN на всех данных ----
    logger.info("Обучение DNN на всех данных...")
    dnn_model_full = train_dnn(X, y, config=config)
    dnn_preds = predict_dnn(dnn_model_full, X_test)

    # ---- Ансамбли для сабмита ----
    logger.info("Создание ансамблей для сабмита...")
    ensemble_results = create_ensembles(
        X, y, X_test, trained_models, cv_results, config,
        extra_preds={'dnn': dnn_preds}
    )

    # ---- Стек для сабмита ----
    logger.info("Создание стекинга для сабмита...")
    top_models = {k: trained_models[k] for k in ['random_forest', 'catboost', 'xgboost', 'lightgbm'] if k in trained_models}
    estimators = [(name, model) for name, model in top_models.items()]
    stack_final = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1
    )
    stack_final.fit(X, y)
    stacking_preds = stack_final.predict(X_test)
    ensemble_results['stacking'] = {'test_pred': stacking_preds, 'description': 'Stacking (RF+XGB+LGB+CB)'}

    # ---- Сохранение сабмитов ----
    logger.info("Сохранение сабмитов...")

    # CatBoost
    if 'catboost' in trained_models:
        cb_preds = trained_models['catboost'].predict(X_test)
        save_submission_from_preds(cb_preds, test_df['Id'], suffix="catboost", log_transform=config.log_transform_target)
        logger.info("  Сохранён: submissions/submission_catboost.csv")

    # Random Forest
    if 'random_forest' in trained_models:
        rf_preds = trained_models['random_forest'].predict(X_test)
        save_submission_from_preds(rf_preds, test_df['Id'], suffix="random_forest", log_transform=config.log_transform_target)
        logger.info("  Сохранён: submissions/submission_random_forest.csv")

    # DNN
    save_submission_from_preds(dnn_preds, test_df['Id'], suffix="dnn", log_transform=config.log_transform_target)
    logger.info("  Сохранён: submissions/submission_dnn.csv")

    # Ансамбли
    for name, result in ensemble_results.items():
        if name in ['best_name', 'best_pred', 'best_model']:
            continue
        if 'test_pred' in result:
            preds = result['test_pred']
            save_submission_from_preds(preds, test_df['Id'], suffix=name, log_transform=config.log_transform_target)
            logger.info(f"  Сохранён: submissions/submission_{name}.csv")

    # Лучший ансамбль по валидации
    best_val_ensemble = min(ensemble_scores, key=ensemble_scores.get)
    best_preds = ensemble_results[best_val_ensemble]['test_pred']
    save_submission_from_preds(best_preds, test_df['Id'], suffix="best_ensemble", log_transform=config.log_transform_target)
    logger.info(f"  Сохранён: submissions/submission_best_ensemble.csv (лучший по валидации: {best_val_ensemble})")

    # ---- Сохранение результатов ----
    save_results(
        cv_results, ensemble_results, config,
        grid_results=None,
        fold_comparison=None,
        ensemble_scores=ensemble_scores
    )
    logger.info("Результаты сохранены в outputs/results.json")

    # ---- Финальный вывод ----
    print("\n" + "="*60)
    logger.info(f"Лучшая модель CV (RMSE): {cv_results['best_name']} ({cv_results['best_score']:.4f})")
    logger.info(f"Лучший ансамбль по валидации (RMSE): {best_val_ensemble} ({ensemble_scores[best_val_ensemble]:.4f})")
    logger.info("="*60 + "\n")


if __name__ == "__main__":
    main()