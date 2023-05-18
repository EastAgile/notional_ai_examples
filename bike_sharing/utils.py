import copy
import optuna
import pandas as pd
import numpy as np
import pyarrow.parquet
from pmdarima.arima import auto_arima
import xgboost as xgb
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import warnings
warnings.filterwarnings(action='ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaObjective():
    def __init__(self, model_name, train_data, target_col, cvs, scoring, timestamp_col):
        self.model_name = model_name
        self.train_data = train_data
        self.target_col = target_col
        self.cvs = cvs
        self.scoring = scoring
        self.timestamp_col = timestamp_col
        self.model_dict = {
            'ridge': (self.create_model_ridge, Ridge()),
            'lasso': (self.create_model_lasso, Lasso()),
            'elastic_net': (self.create_model_enet, ElasticNet()),
            'xgboost': (self.create_model_xgb, xgb.XGBRegressor()),
            'decision_tree': (self.create_model_dtr, DecisionTreeRegressor()),
            'gradient_boosting_tree': (self.create_model_gbr, GradientBoostingRegressor()),
            'random_forest': (self.create_model_rfr, RandomForestRegressor()),
        }


    def __call__(self, trial):
        model = self.model_dict[self.model_name][0](trial)
        cv_result = run_cv(model, self.train_data, self.target_col,
                           self.cvs, self.scoring, self.timestamp_col)
        return np.mean(cv_result)

    def create_model_ridge(self, trial):
        hparams = {
            'alpha': trial.suggest_float("alpha", 0.0, 10.0),
            'max_iter': trial.suggest_int("max_iter", 1000, 10000)
        }
        model = Ridge(**hparams)
        return model

    def create_model_lasso(self, trial):
        alpha = trial.suggest_float("alpha", 0.0, 10.0)
        max_iter = trial.suggest_int("max_iter", 1000, 10000)
        model = Lasso(alpha=alpha, max_iter=max_iter)
        return model

    def create_model_enet(self, trial):
        alpha = trial.suggest_float("alpha", 0.0, 10.0)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
        max_iter = trial.suggest_int("max_iter", 1000, 10000)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter)

        return model

    def create_model_xgb(self, trial):
        eta = trial.suggest_float("eta", 0.01, 0.5)
        gamma = trial.suggest_float("gamma", 0.0, 1.0)
        max_depth = trial.suggest_int("max_depth", 1, 5)
        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        reg_alpha = trial.suggest_float("reg_alpha", 0.0, 10.0)
        reg_lambda = trial.suggest_float("reg_lambda", 0.0, 10.0)
        model = xgb.XGBRegressor(
            eta=eta,
            gamma=gamma,
            max_depth=max_depth,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            tree_method="gpu_hist",
            n_estimators=n_estimators,
            gpu_id=0,
            enable_categorical=True
        )

        return model

    def create_model_dtr(self, trial):
        # Define the hyperparameters to tune
        max_depth = trial.suggest_int("max_depth", 1, 10)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        model = DecisionTreeRegressor(
            max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)

        return model

    def create_model_gbr(self, trial):
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.5)
        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        max_depth = trial.suggest_int("max_depth", 1, 5)
        model = GradientBoostingRegressor(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

        return model

    def create_model_rfr(self, trial):
        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        max_depth = trial.suggest_int("max_depth", 1, 10)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )

        return model


class ARIMAModel:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.model = auto_arima(
            y,
            X=X
        )

    def predict(self, X):
        forecast_data = self.model.predict(
            n_periods=len(X),
            X=X
        )
        return forecast_data.values


def prepare_train_val_test_data(data, target_col, timestamp_col, test_size, val_ratio, cv_fold, prediction_length, add_lag_col=False):
        if add_lag_col:
            data[f'{target_col}_lag_{prediction_length}'] = data[target_col].shift(prediction_length)
            data = data[prediction_length:]
        
        if test_size is None:
            test_size = prediction_length
        if test_size < 0:
            raise Exception()
        if test_size < 1:
            test_size = int(np.floor(len(data)*(test_size)))

        test_data = data.tail(test_size)
        train_data = data.iloc[:-test_size]

        val_size = int(np.floor(val_ratio * len(data))/cv_fold)
        cvs = []
        for i in reversed(range(1, cv_fold + 1)):
            start = train_data[timestamp_col].iloc[-i*val_size]
            end = train_data[timestamp_col].iloc[(-i+1)*val_size - 1]
            cvs.append((start, end))

        return train_data, test_data, cvs
        
        
def get_feature_list(uri):
    schema = pyarrow.parquet.read_schema(uri, memory_map=True)
    return [x for x in schema.names if x != 'date']


def prepare_validation_data(X, end_train, end_val):
    split_1 = X[X['date'] <= end_train]
    split_2 = X[(X['date'] > end_train) & (X['date'] <= end_val)]
    return split_1, split_2


def create_X_y(data, target_col, timestamp_col):
    X = data.drop([target_col, timestamp_col], axis=1)
    y = data[target_col]
    return X, y


def run_cv(model, train_data, target_col, cvs, scoring, timestamp_col, baseline_loss=None):
    losses = [np.nan] * len(cvs)

    for cv, idx in zip(cvs[::-1], reversed(range(len(cvs)))):
        end_train = cv[0]
        end_val = cv[1]

        model = copy.deepcopy(model)

        train, val = prepare_validation_data(train_data, end_train, end_val)

        X_train, y_train = create_X_y(train, target_col, timestamp_col)
        X_val, y_val = create_X_y(val, target_col, timestamp_col)

        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        loss = scoring(y_val, preds)
        if baseline_loss is not None:
            if loss >= baseline_loss[idx]*1 and idx >= len(cvs)-2:
                break
        losses[idx] = loss
    return losses


def add_exo_features(input_df, timestamp_col, feature_list, parquet_file_path, prediction_length):
    min_date = input_df[timestamp_col].min()
    max_date = input_df[timestamp_col].max()
    feature_day = pd.read_parquet(parquet_file_path, columns=feature_list)
    security_features = [x for x in list(feature_day) if x[0] == 'F']
    feature_day.loc[:, security_features] = feature_day[security_features].shift(
        prediction_length).copy()
    feature_day = feature_day[prediction_length:]
    feature_day.index = feature_day.index.astype(str)

    feature_day = feature_day[(feature_day.index >= min_date) & (
        feature_day.index <= max_date)]
    # missing_pct = feature_day.isna().mean()
    # feature_day = feature_day.loc[:, missing_pct < 0.2]
    feature_day.dropna(axis=1, inplace=True)

    input_df = input_df.merge(
        feature_day, left_on=timestamp_col, right_index=True, how='left').copy()
    return input_df


def fine_tune_model(model_name, train_data, target_col, cvs, scoring, timestamp_col, optuna_n_trials):
    optuna_study = optuna.create_study(direction="minimize")
    optuna_objective = OptunaObjective(model_name, train_data,
                        target_col, cvs, scoring, timestamp_col)
    optuna_study.optimize(
        optuna_objective,
        n_trials=optuna_n_trials,
        show_progress_bar=False
    )
    model = optuna_objective.model_dict[model_name][1] 
    model.set_params(**optuna_study.best_params)
    return model


def evaluate_models(models, train_data, test_data, target_col, timestamp_col, scoring):
    X_train, y_train = create_X_y(train_data, target_col, timestamp_col)
    X_test, y_test = create_X_y(test_data, target_col, timestamp_col)
    best_loss = float('inf')
    best_model = 'none'
    for model in models:
        name = type(model).__name__
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        loss = scoring(preds, y_test)
        if loss < best_loss:
            best_loss = loss
            best_model = name
        print(f'Model name: {name}')
        print(f'Loss: {loss}')
    print('==============================')
    print(f'Best model: {best_model}')
    print(f'Best loss: {best_loss}')

