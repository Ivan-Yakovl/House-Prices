import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.label_encoders = {}
        self.fitted = False
        self.numerical_medians = {}

    def fit_transform(self, df):
        self.fitted = True
        return self._process(df)

    def transform(self, df):
        return self._process(df)

    def _process(self, df):
        data = df.copy()

        # 1. Заполнение пропусков в числовых признаках
        if self.config.fill_numerical:
            num_cols = data.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                if data[col].isnull().any():
                    if self.fitted:
                        median = data[col].median()
                        self.numerical_medians[col] = median
                    else:
                        median = self.numerical_medians.get(col, 0)
                    data[col] = data[col].fillna(median)

        # 2. Заполнение пропусков в категориальных признаках
        if self.config.fill_categorical:
            cat_cols = data.select_dtypes(include=['object']).columns
            for col in cat_cols:
                if data[col].isnull().any():
                    data[col] = data[col].fillna("None")

        # 3. Кодирование категориальных признаков
        cat_cols = data.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if self.fitted:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    data[col] = le.transform(data[col].astype(str))
                else:
                    data[col] = data[col].astype('category').cat.codes

        # 4. Удаление Id (не нужно для обучения)
        if 'Id' in data.columns:
            data = data.drop(columns=['Id'])

        return data

_preprocessor = None

def preprocess_data(df, config, fit=True):
    global _preprocessor
    if fit:
        _preprocessor = Preprocessor(config)
        return _preprocessor.fit_transform(df)
    else:
        if _preprocessor is None:
            raise ValueError("Preprocessor not fitted")
        return _preprocessor.transform(df)