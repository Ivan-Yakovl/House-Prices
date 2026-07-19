"""
Главный скрипт проекта Titanic ML
Запуск: python main.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_config
from src.data_loader import load_train_data, load_test_data
from src.preprocessor import preprocess_data
from src.feature_engineering import build_features
from src.evaluate import cross_validate_models, grid_search_models, compare_k_folds
from src.ensemble import create_ensembles
from src.dnn_model import train_dnn, predict_dnn
from src.train import get_model
from src.utils import setup_logger, save_results, save_submission_from_preds

warnings.filterwarnings("ignore")


def ensure_numeric(df):
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].astype('category').cat.codes
    return df


def main():
    config = get_config()
    logger = setup_logger()
    print("\n" + "="*60)
    logger.info("Запуск Titanic ML пайплайна")
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
    X = ensure_numeric(X)
    X_test = ensure_numeric(X_test)
    logger.info(f"Создано признаков: {X.shape[1]}\n")

    # ---- Разделение на train/val ----
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=config.random_state, stratify=y
    )
    logger.info(f"Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}\n")

    # ---- Обучение базовых моделей на X_train с графиками ----
    logger.info("Обучение базовых моделей на train...")
    model_names = ["xgboost", "lightgbm", "logistic_regression"]
    trained_models_val = {}

    # CatBoost с графиком
    logger.info("Обучение CatBoost на train...")
    cb_model = CatBoostClassifier(
        iterations=config.cb_iterations,
        learning_rate=config.cb_learning_rate,
        depth=config.cb_depth,
        verbose=False,
        early_stopping_rounds=config.cb_early_stopping_rounds,
        random_state=config.random_state,
        custom_metric=['Accuracy']
    )
    cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    trained_models_val['catboost'] = cb_model

    cb_history = cb_model.get_evals_result()
    train_loss_cb = cb_history['learn']['Logloss']
    val_loss_cb = cb_history['validation']['Logloss']
    val_acc_cb = cb_history['validation']['Accuracy']

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_cb, label='Train Logloss')
    plt.plot(val_loss_cb, label='Val Logloss')
    plt.xlabel('Iteration')
    plt.ylabel('Logloss')
    plt.title('CatBoost Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(val_acc_cb, label='Val Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('CatBoost Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    Path("outputs").mkdir(exist_ok=True)
    plt.savefig("outputs/catboost_learning_curves.png")
    plt.close()
    logger.info("  График CatBoost сохранён в outputs/catboost_learning_curves.png\n")

    # Random Forest с графиком (loss + accuracy)
    logger.info("Обучение Random Forest на train...")
    n_estimators_range = list(range(10, 210, 10))
    train_scores_rf = []
    val_scores_rf = []
    train_loss_rf = []
    val_loss_rf = []

    for n in n_estimators_range:
        rf = RandomForestClassifier(
            n_estimators=n,
            max_depth=config.rf_max_depth,
            min_samples_split=config.rf_min_samples_split,
            min_samples_leaf=config.rf_min_samples_leaf,
            random_state=config.random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        train_scores_rf.append(rf.score(X_train, y_train))
        val_scores_rf.append(rf.score(X_val, y_val))
        train_probs = rf.predict_proba(X_train)
        val_probs = rf.predict_proba(X_val)
        train_loss_rf.append(log_loss(y_train, train_probs))
        val_loss_rf.append(log_loss(y_val, val_probs))

    rf_final = RandomForestClassifier(
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
    plt.plot(n_estimators_range, train_loss_rf, label='Train Logloss')
    plt.plot(n_estimators_range, val_loss_rf, label='Val Logloss')
    plt.xlabel('Number of Trees')
    plt.ylabel('Logloss')
    plt.title('Random Forest Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(n_estimators_range, train_scores_rf, label='Train Accuracy')
    plt.plot(n_estimators_range, val_scores_rf, label='Val Accuracy')
    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy')
    plt.title('Random Forest Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/rf_learning_curves.png")
    plt.close()
    logger.info("  График Random Forest сохранён в outputs/rf_learning_curves.png\n")

    # Остальные модели
    for name in model_names:
        model = get_model(name, config)
        if model is not None:
            model.fit(X_train, y_train)
            trained_models_val[name] = model

    # DNN на X_train
    logger.info("Обучение DNN на train...")
    dnn_model_val, dnn_history = train_dnn(
        X_train, y_train, X_val, y_val,
        config=config,
        return_history=True
    )
    dnn_preds_val = predict_dnn(dnn_model_val, X_val)
    dnn_acc_val = np.mean((dnn_preds_val >= 0.5) == y_val)
    logger.info(f"  DNN val accuracy: {dnn_acc_val:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(dnn_history['train_loss'], label='Train Loss')
    if dnn_history.get('val_loss'):
        plt.plot(dnn_history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DNN Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/dnn_loss_curves.png")
    plt.close()
    logger.info("  График DNN сохранён в outputs/dnn_loss_curves.png\n")

    # ---- Предсказания на X_val и оценка ансамблей ----
    val_preds_dict = {}
    for name, model in trained_models_val.items():
        if hasattr(model, "predict_proba"):
            val_preds_dict[name] = model.predict_proba(X_val)[:, 1]
        else:
            val_preds_dict[name] = model.predict(X_val).astype(float)
    val_preds_dict['dnn'] = dnn_preds_val

    logger.info("Оценка ансамблей на валидации...")
    ensemble_scores = {}

    simple_avg = np.mean(list(val_preds_dict.values()), axis=0)
    ensemble_scores['simple_average'] = np.mean((simple_avg >= 0.5) == y_val)

    weights = np.ones(len(val_preds_dict))
    weights = weights / weights.sum()
    weighted_avg = np.sum(np.column_stack(list(val_preds_dict.values())) * weights, axis=1)
    ensemble_scores['weighted_average'] = np.mean((weighted_avg >= 0.5) == y_val)

    binary_preds = np.column_stack([(p >= 0.5).astype(int) for p in val_preds_dict.values()])
    hard_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=binary_preds)
    ensemble_scores['hard_voting'] = np.mean(hard_vote == y_val)

    logger.info("Оценка стекинга на валидации...")
    stack_models = {k: v for k, v in trained_models_val.items() if k not in ['dnn', 'catboost']}
    estimators = [(name, model) for name, model in stack_models.items()]
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=1.0, random_state=config.random_state),
        cv=5,
        n_jobs=-1
    )
    stack.fit(X_train, y_train)
    stack_preds_val = stack.predict_proba(X_val)[:, 1]
    ensemble_scores['stacking'] = np.mean((stack_preds_val >= 0.5) == y_val)

    logger.info("Результаты на валидации:")
    logger.info(f"  DNN (отдельно): {dnn_acc_val:.4f}")
    for name, score in ensemble_scores.items():
        logger.info(f"  {name}: {score:.4f}")
    print("")

    # ---- Обучение финальных моделей на всех данных ----
    logger.info("Обучение финальных моделей на всех данных...")
    cv_results = cross_validate_models(X, y, config)
    trained_models = cv_results["trained_models"]

    # ---- GridSearch ----
    logger.info("GridSearch для Random Forest и CatBoost...")
    grid_results = grid_search_models(X, y, config)

    # ---- DNN на всех данных ----
    logger.info("Обучение DNN на всех данных...")
    dnn_model_full = train_dnn(X, y, config=config)
    dnn_preds = predict_dnn(dnn_model_full, X_test)

    # ---- Сравнение 1 vs 5 фолдов ----
    logger.info("Сравнение 1 vs 5 фолдов...")
    compare_df = compare_k_folds(X, y, config)
    print("")

    # ---- Ансамбли для сабмита ----
    logger.info("Создание ансамблей для сабмита...")
    ensemble_results = create_ensembles(
        X, y, X_test, trained_models, cv_results, config,
        extra_preds={'dnn': dnn_preds}
    )

    # ---- Стек для сабмита ----
    logger.info("Создание стекинга для сабмита...")
    top_models = {k: trained_models[k] for k in ['random_forest', 'xgboost', 'lightgbm'] if k in trained_models}
    estimators = [(name, model) for name, model in top_models.items()]
    stack_final = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=1.0, random_state=config.random_state),
        cv=5,
        n_jobs=-1
    )
    stack_final.fit(X, y)
    stacking_preds = stack_final.predict_proba(X_test)[:, 1]
    ensemble_results['stacking'] = {'test_pred': stacking_preds, 'description': 'Stacking (RF+XGB+LGB)'}

    # ---- Сохранение сабмитов ----
    logger.info("Сохранение сабмитов...")
    if 'catboost' in trained_models:
        cb_model_full = trained_models['catboost']
        if hasattr(cb_model_full, "predict_proba"):
            cat_preds = cb_model_full.predict_proba(X_test)[:, 1]
        else:
            cat_preds = cb_model_full.predict(X_test).astype(float)
        save_submission_from_preds(cat_preds, test_df['PassengerId'], suffix="catboost")
        logger.info("  Сохранён: submissions/submission_catboost.csv")

    save_submission_from_preds(dnn_preds, test_df['PassengerId'], suffix="dnn")
    logger.info("  Сохранён: submissions/submission_dnn.csv")

    for name, result in ensemble_results.items():
        if name in ['best_name', 'best_pred', 'best_model']:
            continue
        if 'test_pred' in result:
            preds = result['test_pred']
            save_submission_from_preds(preds, test_df['PassengerId'], suffix=name)
            logger.info(f"  Сохранён: submissions/submission_{name}.csv")

    best_val_ensemble = max(ensemble_scores, key=ensemble_scores.get)
    best_preds = ensemble_results[best_val_ensemble]['test_pred']
    save_submission_from_preds(best_preds, test_df['PassengerId'], suffix="best_ensemble")
    logger.info(f"  Сохранён: submissions/submission_best_ensemble.csv (лучший по валидации: {best_val_ensemble})")

    # ---- Сохранение результатов ----
    save_results(
        cv_results, ensemble_results, config,
        grid_results=grid_results,
        fold_comparison=compare_df,
        ensemble_scores=ensemble_scores
    )
    logger.info("Результаты сохранены в outputs/results.json")

    # ---- Финальный вывод ----
    print("\n" + "="*60)
    logger.info(f"Лучшая модель CV: {cv_results['best_name']} ({cv_results['best_score']:.4f})")
    logger.info(f"Лучший ансамбль по валидации: {best_val_ensemble} ({ensemble_scores[best_val_ensemble]:.4f})")
    logger.info("="*60 + "\n")


if __name__ == "__main__":
    main()