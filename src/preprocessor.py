import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.label_encoders = {}
        self.fitted = False

    def fit_transform(self, df):
        self.fitted = True
        return self._process(df)

    def transform(self, df):
        return self._process(df)

    def _process(self, df):
        data = df.copy()

        # Извлечение Initial
        data["Initial"] = data["Name"].str.extract(r"([A-Za-z]+)\.")
        initial_map = {
            "Mlle": "Miss", "Mme": "Miss", "Ms": "Miss",
            "Dr": "Mr", "Major": "Mr", "Col": "Mr",
            "Rev": "Mr", "Capt": "Mr", "Sir": "Mr",
            "Don": "Mr", "Jonkheer": "Other",
            "Lady": "Mrs", "Countess": "Mrs"
        }
        data["Initial"] = data["Initial"].replace(initial_map)

        # Заполнение пропусков в Age
        if self.config.fill_age:
            age_means = data.groupby("Initial")["Age"].mean().round().astype(int)
            for initial, age in age_means.items():
                mask = data["Age"].isnull() & (data["Initial"] == initial)
                data.loc[mask, "Age"] = age

        # Заполнение пропусков в Embarked
        if self.config.fill_embarked:
            data["Embarked"] = data["Embarked"].fillna("S")

        if "Fare" in data.columns:
            data["Fare"] = data["Fare"].fillna(data["Fare"].median())

        # Кодирование категориальных признаков
        categorical_cols = ["Sex", "Embarked", "Initial"]
        for col in categorical_cols:
            if col in data.columns:
                if self.fitted:
                    le = LabelEncoder()
                    data[col] = le.fit_transform(data[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders.get(col)
                    if le:
                        data[col] = le.transform(data[col].astype(str))

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