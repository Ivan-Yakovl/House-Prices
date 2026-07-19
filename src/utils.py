"""
Вспомогательные функции
"""

import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def save_results(cv_results, ensemble_results, config, grid_results=None, fold_comparison=None, ensemble_scores=None):
    """Сохранение результатов в JSON"""
    Path("outputs").mkdir(exist_ok=True)
    data = {
        "cv_results": cv_results["results"].to_dict(orient="records"),
        "best_model": {
            "name": cv_results["best_name"],
            "score": cv_results["best_score"],
            "std": cv_results["best_std"]
        },
        "ensemble_results": {
            name: {
                "description": res.get("description", ""),
                "val_accuracy": ensemble_scores.get(name, None) if ensemble_scores else None
            }
            for name, res in ensemble_results.items()
            if name not in ["best_name", "best_pred", "best_model"]
        },
        "grid_search": grid_results or {},
        "fold_comparison": fold_comparison.to_dict(orient="records") if fold_comparison is not None else {},
        "config": {
            "n_folds": config.n_folds,
            "random_state": config.random_state,
            "ensemble_methods": config.ensemble_methods
        }
    }
    with open("outputs/results.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_submission_from_preds(preds, passenger_ids, suffix='best'):
    """Сохранение сабмита из предсказаний (вероятности или бинарные)"""
    Path("submissions").mkdir(exist_ok=True)
    binary = (np.array(preds) >= 0.5).astype(int)
    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': binary
    })
    submission.to_csv(f"submissions/submission_{suffix}.csv", index=False)


# (удалена дублирующая функция save_submission, т.к. она не используется)