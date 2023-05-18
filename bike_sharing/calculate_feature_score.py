import time
import sys
import cloudpickle as pickle
from tqdm import tqdm
import pandas as pd
from utils import run_cv
import warnings
warnings.filterwarnings(action='ignore')


def calculate_feature_score(
    model, train_data_exo, org_columns, exo_columns,
    output_dir, file_name, baseline_loss, scoring, target_col, timestamp_col, cvs
):
    start_time = time.time()
    losses_df = pd.DataFrame()
    for i, feature_code in enumerate(tqdm(exo_columns)):
        if isinstance(feature_code, str):
            feature_code = [feature_code]
        train_data_exo_small = train_data_exo.loc[:,
                                                  org_columns + feature_code]
        losses = run_cv(model, train_data_exo_small, target_col,
                        cvs, scoring, timestamp_col, baseline_loss)
        losses_df[','.join(feature_code) if len(
            feature_code) > 0 else 'nothing'] = losses

    results_df = pd.DataFrame(losses_df).T
    cv_cols = [f'cv{x}' for x in range(1, results_df.shape[1] + 1)]
    results_df.columns = cv_cols
    results_df['mean'] = results_df[cv_cols].mean(axis=1)
    results_df.to_csv(f'{output_dir}/{file_name}.csv')
    elapsed_time = time.time() - start_time
    msg = f'{len(exo_columns)} features finished in {elapsed_time:.2f} seconds'
    return msg


input_dict = pickle.load(sys.stdin.buffer)
output = calculate_feature_score(**input_dict)
pickle.dump(output, sys.stdout.buffer)
