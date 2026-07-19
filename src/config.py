# config.py (упрощённая версия для борьбы с переобучением)

class Config:
    def __init__(self):
        # Данные
        self.train_path = "data/train.csv"
        self.test_path = "data/test.csv"
        self.random_state = 42

        # Предобработка
        self.fill_age = True
        self.fill_embarked = True
        self.normalize = False
        self.age_bins = 5
        self.fare_bins = 4

        # Кросс-валидация
        self.n_folds = 5
        self.stratified = True
        self.shuffle = True
        self.scoring = "accuracy"
        self.n_jobs = -1

        # Logistic Regression
        self.lr_C = 0.5          # уменьшили регуляризацию
        self.lr_max_iter = 1000
        self.lr_solver = "liblinear"

        # Ridge
        self.ridge_alpha = 0.5

        # Lasso
        self.lasso_C = 0.5

        # ElasticNet
        self.en_C = 0.5
        self.en_l1_ratio = 0.5

        # KNN
        self.knn_n_neighbors = 7   # увеличили
        self.knn_weights = "distance"
        self.knn_metric = "minkowski"

        # Decision Tree
        self.dt_max_depth = 4      # было 5 → уменьшили
        self.dt_min_samples_split = 10  # было 5 → увеличили

        # Random Forest (упрощён)
        self.rf_n_estimators = 50       # было 100
        self.rf_max_depth = 4           # было 10 → сильно уменьшили
        self.rf_min_samples_split = 10  # было 5 → увеличили
        self.rf_min_samples_leaf = 5    # было 2 → увеличили

        # CatBoost (упрощён)
        self.cb_iterations = 200        # было 500
        self.cb_learning_rate = 0.05    # было 0.1
        self.cb_depth = 4               # было 6
        self.cb_verbose = False
        self.cb_early_stopping_rounds = 30

        # XGBoost
        self.xgb_n_estimators = 200     # было 500
        self.xgb_learning_rate = 0.05   # было 0.1
        self.xgb_max_depth = 4          # было 6
        self.xgb_subsample = 0.8
        self.xgb_colsample_bytree = 0.8

        # LightGBM
        self.lgb_n_estimators = 200     # было 500
        self.lgb_learning_rate = 0.05   # было 0.1
        self.lgb_num_leaves = 15        # было 31 → уменьшили
        self.lgb_max_depth = 4          # было -1 → ограничили

        # DNN (упрощён)
        self.dnn_hidden_sizes = [64, 32]   # было [128, 64, 32]
        self.dnn_activation = 'relu'
        self.dnn_dropout_rate = 0.5        # было 0.3 → увеличили
        self.dnn_batch_norm = True
        self.dnn_learning_rate = 0.001
        self.dnn_batch_size = 32           # было 64
        self.dnn_n_epochs = 50             # было 100
        self.dnn_optimizer = 'adam'

        # Ансамбли
        self.ensemble_methods = ["simple", "weighted"]


def get_config():
    return Config()