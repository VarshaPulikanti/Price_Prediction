from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "california_housing.csv"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
MODEL_REGISTRY_PATH = MODELS_DIR / "model_registry.json"

RANDOM_STATE = 42
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

TARGET_COLUMN = "MedHouseVal"

CATEGORICAL_COLUMNS = []
NUMERIC_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
    "income_per_room",
    "income_per_person",
    "geo_density",
    "lat_lon_interaction",
]

FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
