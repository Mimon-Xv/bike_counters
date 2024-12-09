from pathlib import Path
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler



def weather_cleaning(weather):
    """
    Cleans the weather dataset by handling missing values and dropping unnecessary columns.

    Steps:
    - Impute missing values using mode, median, or mean as appropriate.
    - Fill precipitation and snow-related columns with 0.
    - Drop columns with more than 1000 missing values.

    Args:
        weather (pd.DataFrame): The raw weather dataset.

    Returns:
        pd.DataFrame: The cleaned weather dataset.
    """
    # important to use a copy
    weather = weather.copy()

    mode_columns = ['w1', 'w2', 'n', 'cl', 'cm', 'ch', 'etat_sol', 'nnuage1', 'ctype1']
    for col in mode_columns:
        weather[col].fillna(weather[col].mode()[0], inplace=True)

    # Impute using mean or median for numerical variables
    weather['hnuage1'].fillna(int(weather['hnuage1'].mean()), inplace=True)
    weather['hbas'].fillna(int(weather['hbas'].mean()), inplace=True)
    weather['nbas'].fillna(int(weather['nbas'].mean()), inplace=True)
    weather['tend24'].fillna(weather['tend24'].median(), inplace=True)
    weather['raf10'].fillna(weather['raf10'].median(), inplace=True)

    # Fill precipitation and snow-related columns with 0
    zero_fill_columns = ['ht_neige', 'ssfrai', 'perssfrai', 'rr1', 'rr3', 'rr6', 'rr12', 'rr24']
    weather[zero_fill_columns] = weather[zero_fill_columns].fillna(0)

    # Drop columns with more than 1000 missing values
    columns_to_drop = weather.columns[weather.isnull().sum() > 1000]
    weather.drop(columns=columns_to_drop, inplace=True)
    
    
    '''
    weather = weather.drop(columns=['td', 'dd', 'hnuage1', 'vv', 'tend', 'rafper', 'rr24', 'pres',
                              'pmer', 'ff', 'raf10', 'etat_sol', 'perssfrai', 'ww', 'numer_sta',
                              'cod_tend', 'nnuage1', 'rr12', 'nbas', 'n', 'hbas', 'cl','rr6',
                              'rr3', 'w1', 'rr1', 'ctype1', 'cm', 'w2', 'ch', 'ssfrai', 'per'])
    '''
    
    # Only extracting the vital columns
    
    output = weather
    output = output.drop(columns=['ssfrai', 'n', 'ff', 'perssfrai', 'hbas', 'nbas',
                                  'w2', 'rr12', 'vv', 'rr6', 'cm', 'ctype1',
                                  'cl', 'cod_tend', 'ht_neige', 'ch', 'per', 'numer_sta',
                                  'dd', 'pres', 'pmer', 'hnuage1', 'td', 'rafper',
                                  'u', 'raf10', 'rr24', 'nnuage1']) # proven to be useless
    
    return output



def _merge_external_data(X, external_data_path="../external_data/external_data.csv", merge_columns=None, additional_functions=None):
    """
    Merges the initial dataset X with an external dataset, aligning by the closest timestamp.
    
    Args:
        X (pd.DataFrame): The initial dataset, must have a 'date' column.
        external_data_path (str): Path to the external dataset CSV file.
        merge_columns (list or None): Columns from the external dataset to merge.
            If None, all columns except 'date' will be merged.
    
    Returns:
        pd.DataFrame: Enriched dataset with external data.
    """
    # Load the external dataset
    file_path = os.path.join(os.getcwd(), external_data_path)
    external_data = pd.read_csv(file_path, parse_dates=["date"])
    # Function to clean this dataset. But it may be a different one for different datasets
    external_data = weather_cleaning(external_data)
    
    # Ensure 'date' columns are in datetime format
    X = X.copy()
    external_data = external_data.copy()
    X['date'] = pd.to_datetime(X['date'])
    external_data['date'] = pd.to_datetime(external_data['date'])

    # Default to merging all columns except 'date' if not specified
    merge_columns = [col for col in external_data.columns if col != "date"]

    X['date'] = pd.to_datetime(X['date']).dt.tz_localize(None)
    external_data['date'] = pd.to_datetime(external_data['date']).dt.tz_localize(None) # Add it to avoid tz error
    
    # Add a temporary index to restore original order later
    X["orig_index"] = np.arange(X.shape[0])

    # Perform the as-of merge
    enriched_data = pd.merge_asof(
        X.sort_values("date"),
        external_data[["date"] + merge_columns].sort_values("date"),
        on="date",
        direction="nearest"
    )

    # Restore the original order and clean up temporary columns
    enriched_data = enriched_data.sort_values("orig_index").drop(columns=["orig_index"])

    return enriched_data


def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)
    date_cols = ["year", "month", "day", "weekday", "hour"]

    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["counter_name", "site_name"]

    preprocessor = ColumnTransformer(
        [
            ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
            ("cat", categorical_encoder, categorical_cols),
        ]
    )
    regressor = Ridge()

    pipe = make_pipeline(
        FunctionTransformer(_merge_external_data, validate=False),
        date_encoder,
        preprocessor,
        regressor,
    )

    return pipe

    
def add_arrondissement(df):
    """
    Adds district information to the DataFrame based on a predefined dictionary.
    """
    district_mapping = {
        '28 boulevard Diderot': 12,
        '39 quai François Mauriac': 13,
        "18 quai de l'Hôtel de Ville": 4,
        'Voie Georges Pompidou': 4,
        '67 boulevard Voltaire SE-NO': 11,
        'Face au 48 quai de la marne': 19,
        "Face 104 rue d'Aubervilliers": 19,
        'Face au 70 quai de Bercy': 12,
        '6 rue Julia Bartet': 16,
        "Face au 25 quai de l'Oise": 19,
        '152 boulevard du Montparnasse': 14,
        'Totem 64 Rue de Rivoli': 1,
        'Pont des Invalides S-N': 7,
        'Pont de la Concorde S-N': 7,
        'Pont des Invalides N-S': 7,
        'Face au 8 avenue de la porte de Charenton': 12,
        'Face au 4 avenue de la porte de Bagnolet': 20,
        'Pont Charles De Gaulle': 13,
        '36 quai de Grenelle': 15,
        "Face au 40 quai D'Issy": 15,
        'Pont de Bercy': 12,
        '38 rue Turbigo': 3,
        "Quai d'Orsay": 7,
        '27 quai de la Tournelle': 5,
        "Totem 85 quai d'Austerlitz": 13,
        'Totem Cours la Reine': 8,
        'Totem 73 boulevard de Sébastopol': 1,
        '90 Rue De Sèvres': 7,
        '20 Avenue de Clichy': 17,
        '254 rue de Vaugirard': 15
    }
    # Apply the district mapping
    df = df.copy()
    df['arrondissement'] = df['site_name'].map(district_mapping)
    
    return df

def defining_columns(X):
    X = X.astype({
    'latitude': 'float64',
    'longitude': 'float64',
    'site_id': 'category',
    'is_bank_holiday': 'int8',
    'year': 'int32',
    'month': 'int32', 
    'day': 'int32',
    'weekday': 'int32',
    'hour': 'int32',
    'date': 'datetime64[ns]',
    #'ff': 'float',
    #'t': 'float',
    #'u': 'float',
    #'rr1': 'float',
    #'raf10': 'float',
    #'n': 'category',
    #'ww': 'category',
    'is_weekend': 'int8',
    'season': 'category',
    'hour_sin': 'float64',
    'hour_cos': 'float64',
    'day_sin': 'float64',
    'day_cos': 'float64',
    'time_calm': 'int8',
    'time_morning': 'int8',
    'time_peak_hours': 'int8',
    'time_working_hours': 'int8',
    'arrondissement': 'category'})

    return X