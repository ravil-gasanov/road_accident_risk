from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from accident_risk.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, ORDINAL_FEATURES
from accident_risk.experiments.evaluate import eval_with_cv
from accident_risk.utils import make_experiment_name


def build_random_forest():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", ["curvature"]),
            ("ord", "passthrough", ["speed_limit", "num_reported_accidents"]),
            ("cat", OneHotEncoder(), ["lighting", "weather"]),
        ]
    )

    rf = RandomForestRegressor(random_state=42)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("rf", rf),
        ]
    )

    param_grid = {}

    return pipeline, param_grid


if __name__ == "__main__":
    model, param_grid = build_random_forest()
    eval_with_cv(
        experiment_name=make_experiment_name(__file__),
        model=model,
        param_grid=param_grid,
    )
