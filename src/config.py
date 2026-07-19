class Config:
    def __init__(self):
        # Данные
        self.train_path = "data/train.csv"
        self.test_path = "data/test.csv"
        self.random_state = 42

        # Предобработка
        self.fill_numerical = True
        self.fill_categorical = True
        self.log_transform_target = True   # <-- важно для регрессии

        # Кросс-валидация
        self.n_folds = 5
        self.shuffle = True
        self.scoring = "neg_mean_squared_error"
        self.n_jobs = -1

        # Модели
        self.lr_alpha = 1.0
        self.lr_max_iter = 1000

        self.knn_n_neighbors = 5
        self.knn_weights = "uniform"

        self.dt_max_depth = 5
        self.dt_min_samples_split = 5

        self.rf_n_estimators = 100
        self.rf_max_depth = 10
        self.rf_min_samples_split = 5
        self.rf_min_samples_leaf = 2

        self.cb_iterations = 500
        self.cb_learning_rate = 0.1
        self.cb_depth = 6
        self.cb_verbose = False
        self.cb_early_stopping_rounds = 50

        self.xgb_n_estimators = 500
        self.xgb_learning_rate = 0.1
        self.xgb_max_depth = 6
        self.xgb_subsample = 0.8
        self.xgb_colsample_bytree = 0.8

        self.lgb_n_estimators = 500
        self.lgb_learning_rate = 0.1
        self.lgb_num_leaves = 31
        self.lgb_max_depth = -1

        # DNN
        self.dnn_hidden_sizes = [128, 64, 32]
        self.dnn_activation = 'relu'
        self.dnn_dropout_rate = 0.3
        self.dnn_batch_norm = True
        self.dnn_learning_rate = 0.001
        self.dnn_batch_size = 64
        self.dnn_n_epochs = 100
        self.dnn_optimizer = 'adam'

        # Ансамбли
        self.ensemble_methods = ["simple", "weighted"]


def get_config():
    return Config()