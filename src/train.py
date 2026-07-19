import time
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

def get_model(name, config):
    if name == "ridge":
        return Ridge(alpha=config.lr_alpha, random_state=config.random_state)
    elif name == "lasso":
        return Lasso(alpha=config.lr_alpha, max_iter=config.lr_max_iter, random_state=config.random_state)
    elif name == "elasticnet":
        return ElasticNet(alpha=config.lr_alpha, max_iter=config.lr_max_iter, random_state=config.random_state)
    elif name == "knn":
        return KNeighborsRegressor(n_neighbors=config.knn_n_neighbors, weights=config.knn_weights)
    elif name == "decision_tree":
        return DecisionTreeRegressor(max_depth=config.dt_max_depth, min_samples_split=config.dt_min_samples_split, random_state=config.random_state)
    elif name == "random_forest":
        return RandomForestRegressor(n_estimators=config.rf_n_estimators, max_depth=config.rf_max_depth, min_samples_split=config.rf_min_samples_split, min_samples_leaf=config.rf_min_samples_leaf, random_state=config.random_state, n_jobs=-1)
    elif name == "catboost" and CatBoostRegressor:
        return CatBoostRegressor(iterations=config.cb_iterations, learning_rate=config.cb_learning_rate, depth=config.cb_depth, verbose=config.cb_verbose, random_state=config.random_state)
    elif name == "xgboost" and XGBRegressor:
        return XGBRegressor(n_estimators=config.xgb_n_estimators, learning_rate=config.xgb_learning_rate, max_depth=config.xgb_max_depth, subsample=config.xgb_subsample, colsample_bytree=config.xgb_colsample_bytree, random_state=config.random_state, verbosity=0, n_jobs=-1)
    elif name == "lightgbm" and LGBMRegressor:
        return LGBMRegressor(n_estimators=config.lgb_n_estimators, learning_rate=config.lgb_learning_rate, num_leaves=config.lgb_num_leaves, max_depth=config.lgb_max_depth, random_state=config.random_state, n_jobs=-1, verbose=-1)
    return None

def train_models(X_train, y_train, config, model_names=None):
    if model_names is None:
        model_names = ["ridge", "lasso", "elasticnet", "knn", "decision_tree", "random_forest", "catboost", "xgboost", "lightgbm"]
    results = {}
    for name in model_names:
        model = get_model(name, config)
        if model is None:
            continue
        start = time.time()
        model.fit(X_train, y_train)
        results[name] = (model, time.time() - start)
    return results