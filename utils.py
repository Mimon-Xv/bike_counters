import os
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import external_data.data_handling as dh 
import external_data.date_features as date
 
problem_title = "Bike count prediction"



def get_cv(X, y, random_state=0):
    cv = TimeSeriesSplit(n_splits=8)
    rng = np.random.RandomState(random_state)

    for train_idx, test_idx in cv.split(X):
        # Take a random sampling on test_idx so it's that samples are not consecutives.
        yield train_idx, rng.choice(test_idx, size=len(test_idx) // 3, replace=False)

    
def train_test_split_temporal(X, y, delta_threshold="30 days"):
    """
    Split the data into training and validation sets based on a temporal cutoff.
    Args:
        X (pd.DataFrame): Features with a `date` column.
        y (pd.Series): Target variable.
        delta_threshold (str): Time delta defining the validation cutoff.
    Returns:
        Tuple: X_train, y_train, X_valid, y_valid
    """
    cutoff_date = X["date"].max() - pd.Timedelta(delta_threshold)
    mask = (X["date"] <= cutoff_date)
    X_train, X_valid = X.loc[mask], X.loc[~mask]
    y_train, y_valid = y[mask], y[~mask]
    return X_train, X_valid, y_train, y_valid


def preparation(data):
    # Sort by date first, so that time based cross-validation would produce correct results
    #data = data.sort_values(["date", "counter_name"])
    
    # merging with the weather dataset
    data = dh._merge_external_data(data)
    
    # Addid is_sun_up column, encoding dates, holidays and arrondissements
    # data['is_school_holiday'] = date.get_school_holidays(data['date']) Doesn't work yet
    # data = date.calculate_sunrise_sunset_astral(data) Seems to be useless
    data = date.is_holidays(data)
    data = date._encode_dates(data)
    data = dh.add_arrondissement(data)
    
    # Dropping irrelevant columns
    data = data.drop(columns=["coordinates", "counter_name", "site_name",
                              "counter_installation_date","counter_technical_id",
                              "counter_id"])

    #data = dh.defining_columns(data) # Defining column types
    
    return data


_target_column_name = "log_bike_count"

def get_train_data(path="../data/train.parquet"):
    data = pd.read_parquet(path)
    
    y = data[_target_column_name].values
    X = data.drop([_target_column_name, "bike_count"], axis=1)
    X = preparation(X)
    
    return X, y


def submit_test(pipeline, model_name, path="../data/final_test.parquet"):
    file_name = "../submissions/" + model_name + "_submission.csv"
    
    X_test = pd.read_parquet(path)
    X_test = preparation(X_test)
    
    y_predict = pipeline.predict(X_test)
    
    results = pd.DataFrame(
    dict(
        Id=np.arange(y_predict.shape[0]),
        log_bike_count=y_predict,
        )
    )
    results.to_csv(file_name, index=False)


def create_preprocessor(X):
    """
    Create a reusable ColumnTransformer for preprocessing.

    Parameters:
    - X: DataFrame containing the dataset to derive feature types.

    Returns:
    - preprocessor: A ColumnTransformer for scaling numerical features
                    and one-hot encoding categorical features.
    """
    numerical_features = ['latitude', 'longitude', 't', 'u', 'tend24']

    
    categorical_features = ['weekday', 'season', 'year']

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor

def get_feature_lists(data, features):
    numerical_features = []
    categorical_features = []
    date_features = []
    
    for feature in features:
        if data[feature].dtype in [np.int64, np.float64]:
            numerical_features.append(feature)
        elif pd.api.types.is_datetime64_any_dtype(data[feature]):
            date_features.append(feature)
        else:
            categorical_features.append(feature)
            
    return numerical_features, categorical_features, date_features