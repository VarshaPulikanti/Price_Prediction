from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from src.config import FEATURE_COLUMNS, RANDOM_STATE, RAW_DATA_PATH, TARGET_COLUMN


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["rooms_per_household"] = enriched["AveRooms"] / enriched["AveOccup"].clip(
        lower=0.1
    )
    enriched["bedrooms_per_room"] = enriched["AveBedrms"] / enriched["AveRooms"].clip(
        lower=0.1
    )
    enriched["population_per_household"] = enriched["Population"] / enriched[
        "AveOccup"
    ].clip(lower=0.1)
    enriched["income_per_room"] = enriched["MedInc"] / enriched["AveRooms"].clip(
        lower=0.1
    )
    enriched["income_per_person"] = enriched["MedInc"] / enriched["Population"].clip(
        lower=1.0
    )
    enriched["geo_density"] = enriched["Population"] / (
        enriched["Latitude"].abs() + enriched["Longitude"].abs()
    ).clip(lower=0.1)
    enriched["lat_lon_interaction"] = enriched["Latitude"] * enriched["Longitude"]
    return enriched


def download_real_estate_dataset(output_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    dataset = fetch_california_housing(as_frame=True)
    frame = dataset.frame.rename(columns={"MedHouseVal": TARGET_COLUMN})
    frame = _add_engineered_features(frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def load_dataset(data_path: Path = RAW_DATA_PATH, auto_generate: bool = True) -> pd.DataFrame:
    if data_path.exists():
        df = pd.read_csv(data_path)
        if not set(FEATURE_COLUMNS).issubset(df.columns):
            df = _add_engineered_features(df)
            df.to_csv(data_path, index=False)
        return df
    if not auto_generate:
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    return download_real_estate_dataset(output_path=data_path)


def split_dataset(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
