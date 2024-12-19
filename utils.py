import os
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import LabelEncoder

import external_data.data_handling as dh 
import external_data.date_features as date
 
problem_title = "Bike count prediction"

night_zero_dict = {'28 boulevard Diderot E-O': 0.7433155080213903,
 '28 boulevard Diderot O-E': 0.10900178253119429,
 '39 quai François Mauriac NO-SE': 0.1778966131907308,
 '39 quai François Mauriac SE-NO': 0.3105169340463458,
 "18 quai de l'Hôtel de Ville NO-SE": 0.3129233511586453,
 "18 quai de l'Hôtel de Ville SE-NO": 0.27263814616755794,
 'Voie Georges Pompidou NE-SO': 0.35168539325842696,
 'Voie Georges Pompidou SO-NE': 0.2335205992509363,
 '67 boulevard Voltaire SE-NO': 0.04420677361853832,
 'Face au 48 quai de la marne NE-SO': 0.15347593582887697,
 'Face au 48 quai de la marne SO-NE': 0.09509803921568627,
 "Face 104 rue d'Aubervilliers N-S": 0.22174688057041,
 "Face 104 rue d'Aubervilliers S-N": 0.2640819964349376,
 'Face au 70 quai de Bercy N-S': 0.24973262032085558,
 'Face au 70 quai de Bercy S-N': 0.342602495543672,
 '6 rue Julia Bartet NE-SO': 0.20463458110516933,
 '6 rue Julia Bartet SO-NE': 0.40962566844919784,
 "Face au 25 quai de l'Oise NE-SO": 0.24358288770053474,
 "Face au 25 quai de l'Oise SO-NE": 0.13163992869875224,
 '152 boulevard du Montparnasse E-O': 0.22540106951871658,
 '152 boulevard du Montparnasse O-E': 0.2625668449197861,
 'Totem 64 Rue de Rivoli E-O': 0.12433155080213903,
 'Totem 64 Rue de Rivoli O-E': 0.08511586452762923,
 'Pont des Invalides S-N': 0.553030303030303,
 'Pont de la Concorde S-N': 0.19447415329768267,
 'Pont des Invalides N-S': 0.11363636363636362,
 'Face au 8 avenue de la porte de Charenton NO-SE': 0.35401069518716577,
 'Face au 8 avenue de la porte de Charenton SE-NO': 0.3424242424242424,
 'Face au 4 avenue de la porte de Bagnolet E-O': 0.28386809269162205,
 'Face au 4 avenue de la porte de Bagnolet O-E': 0.2760249554367201,
 'Pont Charles De Gaulle NE-SO': 0.24696969696969695,
 'Pont Charles De Gaulle SO-NE': 0.3343137254901961,
 '36 quai de Grenelle NE-SO': 0.1405525846702317,
 '36 quai de Grenelle SO-NE': 0.2767379679144385,
 "Face au 40 quai D'Issy NE-SO": 0.7599821746880571,
 "Face au 40 quai D'Issy SO-NE": 0.8393939393939392,
 'Pont de Bercy NE-SO': 0.23921568627450981,
 'Pont de Bercy SO-NE': 0.37192513368983954,
 '38 rue Turbigo NE-SO': 0.1360071301247772,
 '38 rue Turbigo SO-NE': 0.13814616755793224,
 "Quai d'Orsay E-O": 0.19670231729055257,
 "Quai d'Orsay O-E": 0.24500891265597147,
 '27 quai de la Tournelle NO-SE': 0.14812834224598928,
 '27 quai de la Tournelle SE-NO': 0.07905525846702317,
 "Totem 85 quai d'Austerlitz NO-SE": 0.1910873440285205,
 "Totem 85 quai d'Austerlitz SE-NO": 0.10089126559714795,
 'Totem Cours la Reine E-O': 0.2972370766488413,
 'Totem Cours la Reine O-E': 0.12584670231729053,
 'Totem 73 boulevard de Sébastopol N-S': 0.028074866310160422,
 'Totem 73 boulevard de Sébastopol S-N': 0.027183600713012478,
 '90 Rue De Sèvres NE-SO': 0.16550802139037432,
 '90 Rue De Sèvres SO-NE': 0.4191622103386809,
 '20 Avenue de Clichy NO-SE': 0.3163992869875223,
 '20 Avenue de Clichy SE-NO': 0.31996434937611407,
 '254 rue de Vaugirard NE-SO': 0.2624242424242424,
 '254 rue de Vaugirard SO-NE': 0.16666666666666666}

def append_night_zero_from_dict(df):
    """
    Appends a `night_zero` column to the dataframe using a precomputed dictionary mapping `counter_name`
    to its corresponding `night_zero` value.

    Parameters:
        df (pd.DataFrame): Input dataframe with a `counter_name` column.
        night_zero_dict (dict): Dictionary mapping `counter_name` to `night_zero` values.

    Returns:
        pd.DataFrame: Dataframe with the `night_zero` column appended.
    """
    df['night_zero'] = df['counter_name'].map(night_zero_dict)
    return df


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
    data['is_school_holiday'] = date.get_school_holidays(data['date'])
    # data = date.calculate_sunrise_sunset_astral(data)
    data = date.is_holidays(data)
    data = date._encode_dates(data)
    data = dh.add_arrondissement(data)
    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Fit and transform the counter_name column
    data['counter_name'] = label_encoder.fit_transform(data['counter_name'])
    
    # Dropping irrelevant columns
    data = data.drop(columns=["coordinates", "counter_id", "site_name",
                              "counter_installation_date","counter_technical_id",
                              "site_id"])

    #data = dh.defining_columns(data) # Defining column types
    
    return data


_target_column_name = "log_bike_count"

def get_train_data(path="../data/train.parquet"):
    data = pd.read_parquet(path)
    
    data = append_night_zero_from_dict(data)
    y = data[_target_column_name].values
    X = data.drop([_target_column_name, "bike_count"], axis=1)
    X = preparation(X)
    
    return X, y


def submit_test(pipeline, model_name, path="../data/final_test.parquet"):
    file_name = "../submissions/" + model_name + "_submission.csv"
    
    X_test = pd.read_parquet(path)
    X_test = preparation(X_test)
    X_test = X_test.drop(columns=["date"])
    
    y_predict = pipeline.predict(X_test)
    
    results = pd.DataFrame(
    dict(
        Id=np.arange(y_predict.shape[0]),
        log_bike_count=y_predict,
        )
    )
    results.to_csv(file_name, index=False)

numeric_columns = ['latitude', 'longitude', 'tend', 't', 'tend24', 'rr1', 'rr3', 'hour_sin',
                    'hour_cos', 'day_sin', 'day_cos', 'ww', ]
categorical_columns = ['counter_name', 'w1', 'etat_sol', 'arrondissement']

# Define the preprocessor
custom_preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_columns),  # Scale numeric columns
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)  # One-hot encode categorical columns
    ]
)

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