import pandas as pd
import numpy as np

_FARE_BINS = None

def build_features(df: pd.DataFrame, config, fit: bool = True):
    global _FARE_BINS
    data = df.copy()
    
    # Age_band (фиксированные бины)
    bins = [0, 16, 32, 48, 64, 100]
    labels = [0, 1, 2, 3, 4]
    data["Age_band"] = pd.cut(data["Age"], bins=bins, labels=labels, right=False)
    data["Age_band"] = data["Age_band"].astype(float)
    
    data["Family_Size"] = data.get("SibSp", 0) + data.get("Parch", 0)
    data["Alone"] = (data["Family_Size"] == 0).astype(int)
    
    if "Fare" in data.columns:
        if fit:
            _, bins_fare = pd.qcut(data["Fare"].clip(lower=0.01), q=config.fare_bins, retbins=True, duplicates='drop')
            _FARE_BINS = bins_fare
            data["Fare_cat"] = pd.cut(data["Fare"].clip(lower=0.01), bins=_FARE_BINS, labels=False, include_lowest=True)
        else:
            if _FARE_BINS is not None:
                data["Fare_cat"] = pd.cut(data["Fare"].clip(lower=0.01), bins=_FARE_BINS, labels=False, include_lowest=True)
            else:
                raise ValueError("Fare bins not fitted. Call fit=True first.")
        data["Fare_cat"] = data["Fare_cat"].astype(float)
    
    drop_cols = ["Name", "Ticket", "Cabin", "PassengerId"]
    for col in drop_cols:
        if col in data.columns:
            data = data.drop(columns=[col])
    
    if "Survived" in data.columns:
        y = data["Survived"]
        X = data.drop(columns=["Survived"])
        return X, y
    else:
        return data, None