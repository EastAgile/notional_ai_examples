import pytest
from utils import prepare_train_val_test_data, run_cv, add_exo_features
import pandas as pd
from pandas import Timestamp
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


@pytest.mark.parametrize(
    "add_lag_col, prediction_length, expected_lag, expected_x_train_len, expected_x_test_len", 
    [
        (True, 3, True, 77, 20), 
        (False, 3, False, 80, 20),
    ]
)
def test_prepare_train_val_test_data(add_lag_col, prediction_length, expected_lag, expected_x_train_len, expected_x_test_len):
    data = pd.DataFrame({
        'target_col': np.arange(100),
        'timestamp_col': pd.date_range(start='1/1/2021', periods=100)
    })
    train_data, test_data, cvs = prepare_train_val_test_data(
        data=data,
        target_col='target_col',
        timestamp_col='timestamp_col',
        test_size=0.2,
        val_ratio=0.1,
        cv_fold=5,
        prediction_length=prediction_length,
        add_lag_col=add_lag_col
    )
    
    if expected_lag:
        assert f'target_col_lag_{prediction_length}' in train_data.columns
    else:
        assert f'target_col_lag_{prediction_length}' not in train_data.columns

    # Assert the lengths of train_data and test_data
    assert len(train_data) == expected_x_train_len
    assert len(test_data) == expected_x_test_len

    # Assert the content of cvs
    expected_cvs = [(Timestamp('2021-03-11 00:00:00'), Timestamp('2021-03-13 00:00:00')), (Timestamp('2021-03-13 00:00:00'), Timestamp('2021-03-15 00:00:00')), (Timestamp('2021-03-15 00:00:00'), Timestamp('2021-03-17 00:00:00')), (Timestamp('2021-03-17 00:00:00'), Timestamp('2021-03-19 00:00:00')), (Timestamp('2021-03-19 00:00:00'), Timestamp('2021-03-21 00:00:00'))]
    assert cvs == expected_cvs


@pytest.mark.parametrize(
    "feature_1, baseline_loss, expected", 
    [
        (np.arange(100), None, 0),  
        (np.random.normal(0, 1, 100), [1e-5, 1e-5], np.nan), 
    ]
)
def test_run_cv(feature_1, baseline_loss, expected):
    # Create test data
    np.random.seed(42)
    timestamp_col = 'date'
    input_feature1 = 'input_feature1'
    input_feature2 = 'input_feature2'
    target_col = 'target'
    timestamps = pd.date_range(start='1/1/2022', periods=100)
    noise2 = np.random.normal(0, 1, 100)
    y = np.arange(100)
    train_data = pd.DataFrame({
        timestamp_col: timestamps,
        input_feature1: y,
        input_feature2: noise2,
        target_col: y
    })
    train_data['input_feature1'] = feature_1

    # Run tests
    model = LinearRegression()
    cvs = [(pd.Timestamp('2022-01-31'), pd.Timestamp('2022-02-28')), 
           (pd.Timestamp('2022-02-28'), pd.Timestamp('2022-03-31'))]

    losses = run_cv(model, train_data, target_col, cvs, mean_squared_error, timestamp_col, baseline_loss)

    if np.isnan(expected):
        assert all(np.isnan(losses)), "Test failed."
    else:
        assert np.allclose(losses, expected, atol=1e-5), "Test failed."


def test_add_exo_features(monkeypatch):
    np.random.seed(42)
    timestamp_col = 'date'
    timestamps = pd.date_range(start='1/1/2022', periods=100)
    input_feature = np.arange(100)
    target = np.arange(100)
    input_df = pd.DataFrame({
        timestamp_col: timestamps,
        'input_feature_1': input_feature,
        'target': target
    })

    feature_list = ['F1', 'F2', 'W1', 'W2']
    feature_day = pd.DataFrame(
        np.arange(103)[:, None] * np.ones((1, 4)),
        columns=feature_list, 
        index=pd.date_range(start='29/12/2021', periods=103).astype(str))
    feature_day.iloc[0, 0] = np.nan
    feature_day.iloc[99, 2] = np.nan

    prediction_length = 3
    parquet_file_path = 'mock_directory'
    
    # Mock read_parquet to return our create_feature_day DataFrame
    monkeypatch.setattr(pd, 'read_parquet', lambda *args, **kwargs: feature_day.copy())
    
    # Make sure the dates in input_df are also strings
    input_df[timestamp_col] = input_df[timestamp_col].astype(str)

    returned_df = add_exo_features(input_df, timestamp_col, feature_list, parquet_file_path, prediction_length)

    expected_df = input_df.copy()
    expected_df.loc[:, ['F2', 'W2']] = np.array([np.arange(100), np.arange(3, 103)]).T.astype('float')
    pd.testing.assert_frame_equal(returned_df, expected_df)
