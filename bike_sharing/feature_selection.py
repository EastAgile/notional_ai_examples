import optuna
import os
import subprocess
import cloudpickle
import numpy as np
from utils import OptunaObjective, get_feature_list, run_cv, add_exo_features
import xgboost as xgb
import pandas as pd
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings(action='ignore')

optuna.logging.set_verbosity(optuna.logging.WARNING)


class FeatureSelector():

    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.cvs = None
        self.timestamp_col = None
        self.target_col = None
        self.prediction_length = None
        self.features_parquet_path = None
        self.output_dir = None
        self.scoring = None
        self.optuna_n_trials = None
        self.xgb_model = None
        self.baseline_loss = None
        self.selected_features = None
        self.gpu_id = None

    def __finetune_xgb(self):
        # Fine-tune xgb
        print('Fine tuning Xgboost model')
        study_xgb = optuna.create_study(direction="minimize")
        study_xgb.optimize(
            OptunaObjective('xgboost', self.train_data, self.target_col,
                            self.cvs, self.scoring, self.timestamp_col),
            n_trials=self.optuna_n_trials,
            show_progress_bar=True
        )

        xgb_params = study_xgb.best_params

        xgb_model = xgb.XGBRegressor(
            enable_categorical=True, tree_method="gpu_hist", gpu_id=self.gpu_id, **xgb_params, random_state=42,
        )
        self.xgb_model = xgb_model

    def __run_feature_selection_loop(self):
        feature_codes = get_feature_list(self.features_parquet_path)
        
        feature_splits = np.array_split(
            feature_codes, len(feature_codes)//5000)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for idx, feature_codes_subset in enumerate(tqdm(feature_splits)):
            train_data_exo = add_exo_features(
                self.train_data,
                self.timestamp_col,
                feature_codes_subset,
                self.features_parquet_path,
                self.prediction_length
            )

            exo_columns = [c for c in list(
                train_data_exo) if c not in self.org_columns]

            file_name = f'part_{idx}'
            input_dict = {
                'model': self.xgb_model,
                'train_data_exo': train_data_exo,
                'org_columns': self.org_columns,
                'exo_columns': exo_columns,
                'output_dir': self.output_dir,
                'file_name': file_name,
                'baseline_loss': self.baseline_loss,
                'scoring': self.scoring,
                'target_col': self.target_col,
                'timestamp_col': self.timestamp_col,
                'cvs': self.cvs
            }

            process = subprocess.Popen(
                ['python', 'calculate_feature_score.py'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                bufsize=0
            )
            cloudpickle.dump(input_dict, process.stdin)
            output = cloudpickle.load(process.stdout)
            print(output)

    def __get_top_10_features(self):
        fs_results = []
        for file_name in os.listdir(self.output_dir):
            if file_name[:5] != 'part_':
                continue
            file_path = os.path.join(self.output_dir, file_name)
            fs_result = pd.read_csv(file_path, index_col=0)
            fs_results.append(fs_result)

        fs_result_full = pd.concat(
            fs_results, axis=0).sort_values('mean').dropna()
        return [str(x) for x in list(fs_result_full.index)[:10]]

    def __get_best_feature_combination(self, top_features: list):
        baseline_mean = np.mean(self.baseline_loss)
        results = {}
        best_mean = baseline_mean

        train_data_exo = add_exo_features(
            self.train_data,
            self.timestamp_col,
            top_features,
            self.features_parquet_path,
            self.prediction_length
        )

        for i in tqdm(range(len(top_features) - 1)):
            selected_features = [i]
            for j in range(i, len(top_features)):
                if i == j:
                    # Calculate mean
                    cur_mean = baseline_mean
                    feature_subset = [top_features[i]]
                else:
                    feature_subset = [top_features[k]
                                      for k in selected_features + [j]]

                cv_result = run_cv(
                    self.xgb_model,
                    train_data_exo.loc[:, self.org_columns + feature_subset],
                    self.target_col,
                    self.cvs,
                    self.scoring,
                    self.timestamp_col
                )
                cv_mean = np.mean(cv_result)
                results[','.join(feature_subset)] = cv_result

                # Update mean if there is improvement
                if cv_mean < cur_mean and i != j:
                    cur_mean = cv_mean
                    selected_features.append(j)

                if cv_mean < best_mean:
                    best_mean = cv_mean

        result_df = pd.DataFrame(results).T
        result_df['mean'] = result_df.mean(axis=1)
        result_df = result_df.sort_values('mean', ascending=False)

        selected_features = [features.split(',') for features in result_df.index]
        self.selected_features = selected_features

    def fit(
        self, train_data, cvs, timestamp_col, target_col,
        prediction_length, features_parquet_path, output_dir, scoring, optuna_n_trials=200, gpu_id=0, fitted=False
    ):
        self.train_data = train_data
        self.org_columns = train_data.columns.tolist()
        self.cvs = cvs
        self.timestamp_col = timestamp_col
        self.target_col = target_col
        self.prediction_length = prediction_length
        self.features_parquet_path = features_parquet_path
        self.output_dir = output_dir
        self.scoring = scoring
        self.optuna_n_trials = optuna_n_trials
        self.gpu_id = gpu_id
        
        self.__finetune_xgb()
        
        self.baseline_loss = run_cv(self.xgb_model, self.train_data,
            self.target_col, self.cvs, self.scoring, self.timestamp_col)
        if not fitted:
            self.__run_feature_selection_loop()
        
        top_features = self.__get_top_10_features()
        if len(top_features) == 0:
            print('There is no feature that would boost the prediction on your data.')
            self.selected_features = []
        else:
            self.__get_best_feature_combination(top_features)

    def get_n_best_features(self, n=5):
        return self.selected_features[:n]

