from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from accident_risk.experiments.evaluate import eval_with_cv


def build_baseline():
    numerical = ["curvature", "speed_limit", "num_reported_accidents"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical),
        ]
    )

    lr = LinearRegression()

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("lr", lr)])

    param_grid = {}

    return pipeline, param_grid


def run_baseline():
    model, param_grid = build_baseline()
    eval_with_cv(
        experiment_name="baseline",
        model=model,
        param_grid=param_grid,
    )


if __name__ == "__main__":
    run_baseline()
