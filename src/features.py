import numpy as np
import pandas as pd


TITLE_REPLACEMENTS = {
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Miss",
}
COMMON_TITLES = ["Mr", "Miss", "Mrs", "Master"]
CHILD_MAX_AGE = 12


def extract_title(names: pd.Series) -> pd.Series:
    """Извлекает титул из имени."""
    return names.str.extract(r',\s*([^.]*)\.')[0].str.strip()


def add_title(data: pd.DataFrame) -> pd.DataFrame:
    """Добавляет колонку Title в датасет. Редкие титулы заменяет на Rare"""
    data = data.copy()
    title = extract_title(data["Name"]).replace(TITLE_REPLACEMENTS)
    data["Title"] = title.where(title.isin(COMMON_TITLES), "Rare")
    return data


def add_family_size(data: pd.DataFrame) -> pd.DataFrame:
    """Добавляет колонку FamilySize - число членов семьи, включая пассажира."""
    data = data.copy()
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    return data


def add_family_group(data: pd.DataFrame) -> pd.DataFrame:
    """Добавляет колонку FamilyGroup - бакеты по размеру семьи."""
    data = data.copy()
    family_size = data["SibSp"] + data["Parch"] + 1
    data["FamilyGroup"] = pd.cut(
        family_size,
        bins=[0, 1, 4, float("inf")],
        labels=["Alone", "Small", "Large"],
    )
    return data


def add_has_cabin(data: pd.DataFrame) -> pd.DataFrame:
    """Добавляет флаг HasCabin: 1, если заполнен Cabin, иначе 0"""
    data = data.copy()
    data["HasCabin"] = data["Cabin"].notna().astype(int)
    return data


def add_is_child(data: pd.DataFrame) -> pd.DataFrame:
    """Добавляет флаг IsChild по заданному порогу CHILD_MAX_AGE."""
    data = data.copy()
    data["IsChild"] = np.where(
        data["Age"].isna(),
        np.nan,
        (data["Age"] <= CHILD_MAX_AGE).astype(int),
    )
    return data


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Добавляет все необходимые фичи для финальной модели."""
    data = add_title(data)
    data = add_family_group(data)
    data = add_has_cabin(data)
    return data
