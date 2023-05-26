import pytest
import os
from sklearn.linear_model import LinearRegression
from calculate_feature_score import calculate_feature_score
from utils import run_cv
import numpy as np
import pandas as pd

# Mocking run_cv to return predefined output
def mock_run_cv(model, train_data_exo_small, target_col,
                        cvs, scoring, timestamp_col, baseline_loss):
    if 'F1' in train_data_exo_small.columns or 'F2' in train_data_exo_small.columns:
        return [1, 2, 3, 2]
    else:
        return [np.nan, np.nan, np.nan, np.nan]

def test_calculate_feature_score(monkeypatch, tmpdir):
    # Mock the run_cv function
    monkeypatch.setattr("calculate_feature_score.run_cv", mock_run_cv)

    # Setup for test
    model = LinearRegression()
    train_data_exo = pd.DataFrame({
        'timestamp_col': pd.date_range(start='1/1/2021', periods=5),
        'target_col': [1, 2, 3, 4, 5],
        'org_column': [5, 4, 3, 2, 1],
        'F1': [10, 20, 30, 40, 50],
        'F2': [50, 40, 30, 20, 10],
        'W1': [60, 70, 80, 90, 100],
        'W2': [100, 90, 80, 70, 60],
    })
    org_columns = ['timestamp_col', 'target_col', 'org_column']
    exo_columns = ['F1', 'F2', 'W1', 'W2']
    output_dir = str(tmpdir)
    file_name = 'test_file'
    baseline_loss = [10, 20, 30]
    scoring = 'mse'
    target_col = 'target_col'
    timestamp_col = 'timestamp_col'
    cvs = [('1/1/2021', '1/1/2021'),('1/1/2021', '1/1/2021'),('1/1/2021', '1/1/2021')]
    
    # Call the calculate_feature_score function
    result = calculate_feature_score(
        model,
        train_data_exo,
        org_columns,
        exo_columns,
        output_dir,
        file_name,
        baseline_loss,
        scoring,
        target_col,
        timestamp_col,
        cvs
    )

    # Check if the CSV file was created
    output_file = os.path.join(output_dir, f"{file_name}.csv")
    assert os.path.isfile(output_file)

    # Load the CSV file and check its content
    df = pd.read_csv(output_file, index_col=0)
    assert all(df.columns == ['cv1', 'cv2', 'cv3', 'cv4', 'mean'])
    assert all(df.loc['F1'] == [1, 2, 3, 2, np.mean([1, 2, 3, 2])])
    assert all(df.loc['F2'] == [1, 2, 3, 2, np.mean([1, 2, 3, 2])])
    assert all(np.isnan(df.loc['W1']))
    assert all(np.isnan(df.loc['W2']))
