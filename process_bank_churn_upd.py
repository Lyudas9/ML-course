import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from typing import Tuple, List, Dict, Any


def split_train_val(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataframe into training and validation sets.

    Args:
        df (pd.DataFrame): The raw dataframe.
        target_col (str): The target column for stratification.
        test_size (float): The proportion of the dataset to include in the validation split.
        random_state (int): Random state for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and validation dataframes.
    """
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target_col])
    return train_df, val_df


def separate_inputs_targets(df: pd.DataFrame, input_cols: List[str], target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate inputs and targets from the dataframe.

    Args:
        df (pd.DataFrame): The dataframe.
        input_cols (List[str]): List of input columns.
        target_col (str): Target column.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: DataFrame of inputs and Series of targets.
    """
    inputs = df[input_cols].copy()
    targets = df[target_col].copy()
    return inputs, targets


def fit_scaler(train_inputs: pd.DataFrame, numeric_cols: List[str]) -> MinMaxScaler:
    """
    Fit a MinMaxScaler to the training inputs.

    Args:
        train_inputs (pd.DataFrame): Training inputs.
        numeric_cols (List[str]): List of numerical columns.

    Returns:
        MinMaxScaler: Fitted scaler.
    """
    scaler = MinMaxScaler().fit(train_inputs[numeric_cols])
    return scaler


def scale_numeric_features(scaler: MinMaxScaler, inputs: pd.DataFrame, numeric_cols: List[str]) -> None:
    """
    Scale numeric features using the provided scaler.

    Args:
        scaler (MinMaxScaler): Fitted scaler.
        inputs (pd.DataFrame): Inputs to be scaled.
        numeric_cols (List[str]): List of numerical columns.
    """
    inputs[numeric_cols] = scaler.transform(inputs[numeric_cols])


def fit_encoder(train_inputs: pd.DataFrame, categorical_cols: List[str]) -> OneHotEncoder:
    """
    Fit a OneHotEncoder to the categorical columns in the training inputs.

    Args:
        train_inputs (pd.DataFrame): Training inputs.
        categorical_cols (List[str]): List of categorical columns.

    Returns:
        OneHotEncoder: Fitted encoder.
    """
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore').fit(train_inputs[categorical_cols])
    return encoder


def encode_categorical_features(encoder: OneHotEncoder, inputs: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
    """
    Encode categorical features using the provided encoder.

    Args:
        encoder (OneHotEncoder): Fitted encoder.
        inputs (pd.DataFrame): Inputs to be encoded.
        categorical_cols (List[str]): List of categorical columns.

    Returns:
        pd.DataFrame: DataFrame with encoded categorical features.
    """
    encoded = encoder.transform(inputs[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=inputs.index)
    inputs = inputs.drop(columns=categorical_cols)
    return pd.concat([inputs, encoded_df], axis=1)


def delete_columns(df: pd.DataFrame, cols_to_delete: List[str]) -> pd.DataFrame:
    """
    Delete specified columns from the dataframe.

    Args:
        df (pd.DataFrame): The dataframe.
        cols_to_delete (List[str]): List of columns to delete.

    Returns:
        pd.DataFrame: DataFrame with specified columns deleted.
    """
    return df.drop(columns=cols_to_delete)


def preprocess_data(raw_df: pd.DataFrame, target_col: str, cols_to_delete: List[str] = None, scale_numeric: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], MinMaxScaler, OneHotEncoder]:
    """
    Preprocess the raw dataframe.

    Args:
        raw_df (pd.DataFrame): The raw dataframe.
        target_col (str): The target column.
        cols_to_delete (List[str]): List of columns to delete.
        scale_numeric (bool): Option to scale numerical features.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], MinMaxScaler, OneHotEncoder]:
            Processed training inputs, training targets, validation inputs, validation targets,
            input columns, fitted scaler, and fitted encoder.
    """
    if cols_to_delete:
        raw_df = delete_columns(raw_df, cols_to_delete)
    
    # Split data into training and validation sets
    train_df, val_df = split_train_val(raw_df, target_col)

    # Define input columns
    input_cols = list(train_df.columns)
    input_cols.remove(target_col)

    # Separate inputs and targets
    train_inputs, train_targets = separate_inputs_targets(train_df, input_cols, target_col)
    val_inputs, val_targets = separate_inputs_targets(val_df, input_cols, target_col)

    # Identify numeric and categorical columns
    numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = train_inputs.select_dtypes(include='object').columns.tolist()

    # Scale numeric features if required
    scaler = None
    if scale_numeric:
        scaler = fit_scaler(train_inputs, numeric_cols)
        scale_numeric_features(scaler, train_inputs, numeric_cols)
        scale_numeric_features(scaler, val_inputs, numeric_cols)

    # Encode categorical features
    encoder = fit_encoder(train_inputs, categorical_cols)
    train_inputs = encode_categorical_features(encoder, train_inputs, categorical_cols)
    val_inputs = encode_categorical_features(encoder, val_inputs, categorical_cols)

    # Extract X_train, X_val
    X_train = train_inputs
    X_val = val_inputs

    return X_train, train_targets, X_val, val_targets, input_cols, scaler, encoder


def preprocess_new_data(new_df: pd.DataFrame, input_cols: List[str], scaler: MinMaxScaler, encoder: OneHotEncoder, scale_numeric: bool = True) -> pd.DataFrame:
    """
    Preprocess new data using the provided scaler and encoder.

    Args:
        new_df (pd.DataFrame): The new dataframe.
        input_cols (List[str]): List of input columns.
        scaler (MinMaxScaler): Fitted scaler.
        encoder (OneHotEncoder): Fitted encoder.
        scale_numeric (bool): Option to scale numerical features.

    Returns:
        pd.DataFrame: Processed inputs for the new data.
    """
    inputs = new_df[input_cols].copy()

    # Identify numeric and categorical columns
    numeric_cols = inputs.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = inputs.select_dtypes(include='object').columns.tolist()

    # Scale numeric features if required
    if scale_numeric:
        scale_numeric_features(scaler, inputs, numeric_cols)

    # Encode categorical features
    inputs = encode_categorical_features(encoder, inputs, categorical_cols)

    return inputs