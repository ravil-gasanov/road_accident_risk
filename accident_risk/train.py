from loguru import logger
import mlflow
import pandas as pd
import typer

from accident_risk.config import MLFLOW_TRACKING_URI, RAW_TEST_PATH, RAW_TRAIN_PATH
from accident_risk.data.load import load_X_y

app = typer.Typer()

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def load_model(model_id):
    model_uri = f"models:/{model_id}"
    try:
        model = mlflow.sklearn.load_model(model_uri=model_uri)
    except Exception as e:
        logger.error("Failed to load the model")

        raise e

    return model


def train(model_id):
    X, y = load_X_y(path=RAW_TRAIN_PATH)

    model = load_model(model_id)

    model.fit(X, y)

    return model


def make_submission(model):
    X_test = pd.read_csv(RAW_TEST_PATH)

    y_pred = model.predict(X_test)

    submission = pd.DataFrame()
    submission["id"] = X_test["id"]
    submission["accident_risk"] = y_pred

    submission.to_csv("submission.csv", index=False)


@app.command()
def train_predict(model_id: str):
    logger.info(f"Training model with ID: {model_id}")
    model = train(model_id=model_id)

    logger.info("Predicting on test and making a submission file")
    make_submission(model=model)


if __name__ == "__main__":
    app()
