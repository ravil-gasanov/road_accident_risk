import mlflow
from sklearn.model_selection import GridSearchCV

from accident_risk.config import CV_SPLITS, MLFLOW_TRACKING_URI, RAW_TRAIN_PATH
from accident_risk.data.load import load_X_y


def eval_with_cv(
    experiment_name,
    model,
    param_grid,
):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    X, y = load_X_y(path=RAW_TRAIN_PATH)

    gridcv = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=CV_SPLITS,
        scoring="neg_root_mean_squared_error",
    )

    with mlflow.start_run():
        gridcv.fit(X, y)

        mlflow.log_params(gridcv.best_params_)
        mlflow.log_metric("rmse_best_score", -gridcv.best_score_)

        mlflow.sklearn.log_model(gridcv.best_estimator_, "baseline_model")
