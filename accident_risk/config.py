import os

from dotenv import load_dotenv

load_dotenv()


TARGET = "accident_risk"
CATEGORICAL_FEATURES = [
    "road_type",
    "lighting",
    "weather",
    "road_signs_present",  # binary
    "public_road",  # binary
    "time_of_day",
    "holiday",  # binary
    "school_season",
]
ORDINAL_FEATURES = ["num_lanes", "speed_limit"]
NUMERICAL_FEATURES = ["curvature", "speed_limit", "num_reported_accidents"]

RAW_TRAIN_PATH = "data/raw/train.csv"
RAW_TEST_PATH = "data/raw/test.csv"

CV_SPLITS = 5
RANDOM_STATE = 42


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
