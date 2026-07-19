# Titanic ML Project

Предсказание выживаемости пассажиров Титаника (Kaggle competition).

## Структура проекта
- `src/` – исходный код (модули для загрузки, предобработки, обучения, оценки)
- `notebooks/eda.ipynb` – исследовательский анализ данных (EDA)
- `submissions/` – сгенерированные файлы для отправки на Kaggle
- `outputs/` – результаты обучения (JSON, графики обучения)

## Используемые модели
1. **Логистическая регрессия** (базовый классификатор)
2. **Ridge, Lasso, ElasticNet** (линейные модели с регуляризацией)
3. **KNN** (метод ближайших соседей)
4. **Decision Tree** (решающее дерево)
5. **Random Forest** (случайный лес) – CV Score: 0.8339
6. **CatBoost** – CV Score: 0.8406
7. **XGBoost** – CV Score: 0.8395
8. **LightGBM** – CV Score: 0.8462 (лучшая модель по CV)
9. **DNN (PyTorch)** – простая нейросеть с BatchNorm и Dropout

## Feature Engineering
- Извлечение обращения (Mr/Mrs/Miss/etc.) из имени для заполнения возраста
- Age_band (5 возрастных групп)
- Family_Size (Parch + SibSp)
- Alone (одинок ли пассажир)
- Fare_cat (категории стоимости билета)

## Ансамбли
- **Simple average** – усреднение вероятностей
- **Weighted average** – взвешенное усреднение по CV скорам
- **Hard voting** – голосование по классам
- **Stacking** – мета-модель (LR) на предсказаниях RF, XGB, LGB

## Графики обучения
- [CatBoost](outputs/catboost_learning_curves.png)
- [Random Forest](outputs/rf_learning_curves.png)
- [DNN](outputs/dnn_loss_curves.png)

## Результаты (на валидации)
| Модель / Ансамбль | Accuracy |
|-------------------|----------|
| LightGBM (CV) | 0.8462 |
| CatBoost (CV) | 0.8406 |
| XGBoost (CV) | 0.8395 |
| Hard Voting | 0.8436 |
| Simple Average | 0.8324 |
| Stacking | 0.8268 |

## Запуск проекта
1. Установка зависимостей:
   ```bash
   pip install -r requirements.txt