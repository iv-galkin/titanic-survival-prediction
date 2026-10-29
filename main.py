from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from src.data import load_data, prepare_model_data, set_categorical_features
from src.features import add_features

import yaml

def main() -> None:
    config_path = Path(__file__).parent / "config.yaml"

    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    logger.add(
        "output/pipeline.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )

    target = config["target"]
    num_features = config["features"]["numeric"]
    cat_features = config["features"]["categorical"]
    features = num_features + cat_features

    logger.info("Loading train and test datasets")
    train_data, test_data = load_data(
        config["data"]["train_path"],
        config["data"]["test_path"],
    )
    logger.info("Loaded {} train rows and {} test rows", len(train_data), len(test_data))

    logger.info("Generating features")
    train_data = add_features(train_data)
    test_data = add_features(test_data)

    logger.info("Preparing model inputs")
    X_train, y_train, X_test = prepare_model_data(
        train_data,
        test_data,
        features,
        target,
    )

    X_train = set_categorical_features(
        X_train,
        cat_features,
    )

    X_test = set_categorical_features(
        X_test,
        cat_features,
    )

    logger.info("Creating XGBoost model")
    model = XGBClassifier(**config["model"]["params"])

    if config["validation"]["run_cv"]:
        logger.info("Running cross-validation")
        cv = StratifiedKFold(
            n_splits=config["validation"]["n_splits"],
            random_state=config["validation"]["random_state"],
            shuffle=config["validation"]["shuffle"],
        )

        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=config["validation"]["scoring"],
            n_jobs=config["validation"]["n_jobs"],
            error_score="raise",
        )

        logger.info(
            "CV accuracy: {:.4f} ± {:.4f}",
            cv_scores.mean(),
            cv_scores.std(),
        )

    logger.info("Training XGBoost on all training data")
    model.fit(X_train, y_train)

    logger.info("Making predictions for the test dataset")
    predictions = model.predict(X_test)

    submission = pd.DataFrame(
        {
            "PassengerId": test_data["PassengerId"],
            "Survived": predictions,
        }
    )

    submission.to_csv(
        config["data"]["submission_path"],
        index=False,
    )
    logger.success("Submission saved to {}", config["data"]["submission_path"])


if __name__ == "__main__":
    main()
