from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def encode_geography(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the 'Geography' column using frequency encoding.

    Args:
        df (pd.DataFrame): The input DataFrame containing the 'Geography' column.

    Returns:
        pd.DataFrame: The DataFrame with the encoded 'Geography' column.
    """
    frequency_encoding = df['Geography'].value_counts(normalize=True)
    df['Geography_encoded'] = df['Geography'].map(frequency_encoding)
    return df


def encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the 'Gender' column with binary values.

    Args:
        df (pd.DataFrame): The input DataFrame containing the 'Gender' column.

    Returns:
        pd.DataFrame: The DataFrame with the encoded 'Gender' column.
    """
    gender_codes = {'Male': 1, 'Female': 0}
    df['Gender_Type_Code'] = df['Gender'].map(gender_codes)
    return df


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop unnecessary columns from the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with unnecessary columns dropped.
    """
    df.drop('Surname', axis=1, inplace=True)
    df.drop(['Gender', 'Geography'], axis=1, inplace=True)
    df.drop(['CustomerId'], axis=1, inplace=True)
    return df


def split_dataset(df: pd.DataFrame, test_size: float, random_state: int, stratify_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame into training and validation sets.

    Args:
        df (pd.DataFrame): The input DataFrame.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The seed used by the random number generator.
        stratify_col (str): The column to use for stratified sampling.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The training and validation DataFrames.
    """
    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[stratify_col])


def create_train_val_sets(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Create training and validation input and target sets.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The target column name.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: The training and validation inputs and targets.
    """
    input_cols = list(df.columns)
    input_cols.remove(target_col)
    train_df, val_df = split_dataset(df, test_size=0.20, random_state=42, stratify_col=target_col)
    train_inputs, train_targets = train_df[input_cols].copy(), train_df[target_col].copy()
    val_inputs, val_targets = val_df[input_cols].copy(), val_df[target_col].copy()
    return train_inputs, train_targets, val_inputs, val_targets


def preprocess_data(raw_df: pd.DataFrame, scale_numeric: bool = False) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Optional[StandardScaler], Optional[SimpleImputer]]:
    """
    Process the raw DataFrame and return training and validation sets along with the scaler and imputer.

    Args:
        raw_df (pd.DataFrame): The input raw DataFrame.
        scale_numeric (bool): Whether to scale numeric features.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Optional[StandardScaler], Optional[SimpleImputer]]: The training and validation inputs and targets, along with the scaler and imputer.
    """
    # Encode categorical columns
    raw_df = encode_geography(raw_df)
    raw_df = encode_gender(raw_df)
    raw_df = drop_unnecessary_columns(raw_df)

    # Split into features and target
    target_col = 'Exited'
    input_cols = list(raw_df.columns)
    input_cols.remove(target_col)
    features_df = raw_df[input_cols]
    target_df = raw_df[target_col]

    # Handle missing values in features
    imputer = SimpleImputer(strategy='mean')
    features_df = pd.DataFrame(imputer.fit_transform(features_df), columns=features_df.columns)

    # Recombine features and target
    raw_df = pd.concat([features_df, target_df], axis=1)

    # Split into training and validation sets
    train_df, val_df = split_dataset(raw_df, test_size=0.20, random_state=42, stratify_col=target_col)

    # Create training and validation sets
    X_train, y_train = create_train_val_sets(train_df, target_col='Exited')[:2]
    X_val, y_val = create_train_val_sets(val_df, target_col='Exited')[:2]

    # Scale numeric features if requested
    scaler = None
    if scale_numeric:
        scaler = StandardScaler()
        numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])

    return X_train, y_train, X_val, y_val, scaler, imputer


def preprocess_new_data(new_data: pd.DataFrame, scaler: Optional[StandardScaler] = None, imputer: Optional[SimpleImputer] = None) -> pd.DataFrame:
    """
    Preprocess new data using the same transformations as the training data.

    Args:
        new_data (pd.DataFrame): The new data to preprocess.
        scaler (Optional[StandardScaler]): The scaler used to scale numeric data.
        imputer (Optional[SimpleImputer]): The imputer used to fill missing values.

    Returns:
        pd.DataFrame: The preprocessed new data.
    """
    new_data = encode_geography(new_data)
    new_data = encode_gender(new_data)
    new_data = drop_unnecessary_columns(new_data)

    # Handle missing values in features
    if imputer:
        input_cols = list(new_data.columns)
        new_data = pd.DataFrame(imputer.transform(new_data), columns=input_cols)

    # Scale numeric features if scaler is provided
    if scaler:
        numeric_cols = new_data.select_dtypes(include=['float64', 'int64']).columns
        new_data[numeric_cols] = scaler.transform(new_data[numeric_cols])

    return new_data

# raw_df = pd.read_csv('path_to_your_data.csv')  # Load your data
# X_train, y_train, X_val, y_val, scaler, imputer = preprocess_data(raw_df, scale_numeric=True)
# new_raw_df = pd.read_csv('path_to_your_new_data.csv')  # Load new data
# preprocessed_new_data = preprocess_new_data(new_raw_df, scaler=scaler, imputer=imputer)