import os

from dotenv import load_dotenv

load_dotenv()


TARGET = "accident_risk"

RAW_TRAIN_PATH = "data/raw/train.csv"
RAW_TEST_PATH = "data/raw/test.csv"

CV_SPLITS = 5
RANDOM_STATE = 42


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
