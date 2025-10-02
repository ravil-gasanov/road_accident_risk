import pandas as pd

from accident_risk.config import TARGET


def load_X_y(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load data from a CSV file."""
    data = pd.read_csv(path)

    y = data[TARGET]
    X = data.drop(columns=[TARGET])

    return X, y
