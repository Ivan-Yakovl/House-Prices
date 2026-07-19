import pandas as pd
import numpy as np

def build_features(df, config, fit=True):
    data = df.copy()

    # 1. Если есть SalePrice, логарифмируем (для регрессии)
    if 'SalePrice' in data.columns and config.log_transform_target:
        y = np.log1p(data['SalePrice'])
        X = data.drop(columns=['SalePrice'])
    elif 'SalePrice' in data.columns:
        y = data['SalePrice']
        X = data.drop(columns=['SalePrice'])
    else:
        y = None
        X = data

    # 2. Создание новых признаков (примеры для House Prices)
    # Общая площадь
    if 'TotalBsmtSF' in X.columns and '1stFlrSF' in X.columns and '2ndFlrSF' in X.columns:
        X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']

    # Общее качество
    if 'OverallQual' in X.columns and 'OverallCond' in X.columns:
        X['OverallQuality'] = X['OverallQual'] * X['OverallCond']

    # Возраст дома
    if 'YearBuilt' in X.columns and 'YrSold' in X.columns:
        X['HouseAge'] = X['YrSold'] - X['YearBuilt']

    # Ремонт
    if 'YearRemodAdd' in X.columns and 'YearBuilt' in X.columns:
        X['YearsSinceRemod'] = X['YearRemodAdd'] - X['YearBuilt']

    # Количество санузлов
    if 'FullBath' in X.columns and 'HalfBath' in X.columns and 'BsmtFullBath' in X.columns and 'BsmtHalfBath' in X.columns:
        X['TotalBath'] = X['FullBath'] + 0.5*X['HalfBath'] + X['BsmtFullBath'] + 0.5*X['BsmtHalfBath']

    # Количество крытых парковок
    if 'GarageCars' in X.columns and 'GarageArea' in X.columns:
        X['GarageCapacity'] = X['GarageCars'] * X['GarageArea']

    # 3. Удаление признаков с высоким процентом пропусков (если есть)
    # Например, если >50% пропусков — удаляем
    for col in X.columns:
        if X[col].isnull().mean() > 0.5:
            X = X.drop(columns=[col])

    return X, y