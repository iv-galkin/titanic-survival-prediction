import pandas as pd


TITLE_REPLACEMENTS = {
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Miss",
}
COMMON_TITLES = ["Mr", "Miss", "Mrs", "Master"]


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create the final feature set used by the XGBoost pipeline."""
    data = data.copy()

    data["Title"] = (
        data["Name"]
        .str.extract(r',\s*([^.]*)\.')[0]
        .str.strip()
    )

    data["Title"] = data["Title"].replace(TITLE_REPLACEMENTS)

    data["Title"] = data["Title"].where(
        data["Title"].isin(COMMON_TITLES),
        "Rare",
    )

    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1

    data["FamilyGroup"] = pd.cut(
        data["FamilySize"],
        bins=[0, 1, 4, float("inf")],
        labels=["Alone", "Small", "Large"],
    )

    data = data.drop(columns=["FamilySize"])

    data["HasCabin"] = data["Cabin"].notna().astype(int)

    return data
