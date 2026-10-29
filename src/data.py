import pandas as pd


def load_data(
    train_data_path: str, 
    test_data_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the train and test datasets from CSV files."""
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    return train_data, test_data


def prepare_model_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Select model features and the target from train and test datasets."""
    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    return X_train, y_train, X_test


def set_categorical_features(
    data: pd.DataFrame,
    categorical_features: list[str],
) -> pd.DataFrame:
    """Convert selected columns to pandas categorical dtype."""
    data = data.copy()

    for feature in categorical_features:
        data[feature] = data[feature].astype("category")

    return data
