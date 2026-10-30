import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(
    train_data_path: str, 
    test_data_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Загружает тренировочную и тестовую выборки"""
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    return train_data, test_data


def prepare_model_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    return X_train, y_train, X_test


def set_categorical_features(
    data: pd.DataFrame,
    categorical_features: list[str],
) -> pd.DataFrame:
    """Приводим к категориальному типу заданные фичи"""
    data = data.copy()

    for feature in categorical_features:
        data[feature] = data[feature].astype("category")

    return data


def build_preprocessor(
    num_features: list[str],
    cat_features: list[str],
) -> ColumnTransformer:
    """Препроцессинг для линейной модели и нейронной сети.
    
    Числовые фичи заполняются медианой и стандартизуются.
    Категориальные заполняются модой и кодируются one-hot.
    """
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_features),
        ("cat", cat_pipe, cat_features),
    ])
