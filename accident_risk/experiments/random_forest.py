from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest

from accident_risk.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, ORDINAL_FEATURES
from accident_risk.experiments.evaluate import eval_with_cv
from accident_risk.utils import make_experiment_name


def build_random_forest():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERICAL_FEATURES),
            ("ord", "passthrough", ORDINAL_FEATURES),
            ("cat", OneHotEncoder(), CATEGORICAL_FEATURES),
        ]
    )

    SelectKBest(k=10)

    rf = RandomForestRegressor(random_state=42)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("feature_selection", SelectKBest(k=10)),
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
