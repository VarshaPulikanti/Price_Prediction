import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.config import FEATURE_COLUMNS, MODEL_PATH


RAW_FEATURE_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


def prepare_features(raw_features: dict) -> dict:
    missing = [col for col in RAW_FEATURE_COLUMNS if col not in raw_features]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    features = {key: float(raw_features[key]) for key in RAW_FEATURE_COLUMNS}
    features["rooms_per_household"] = features["AveRooms"] / max(features["AveOccup"], 0.1)
    features["bedrooms_per_room"] = features["AveBedrms"] / max(features["AveRooms"], 0.1)
    features["population_per_household"] = features["Population"] / max(
        features["AveOccup"], 0.1
    )
    features["income_per_room"] = features["MedInc"] / max(features["AveRooms"], 0.1)
    features["income_per_person"] = features["MedInc"] / max(features["Population"], 1.0)
    features["geo_density"] = features["Population"] / max(
        abs(features["Latitude"]) + abs(features["Longitude"]), 0.1
    )
    features["lat_lon_interaction"] = features["Latitude"] * features["Longitude"]
    return features


def predict_price(features: dict, model_path: Path = MODEL_PATH) -> float:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Run training first: python -m src.train"
        )

    prepared_features = prepare_features(features)
    model = joblib.load(model_path)
    input_df = pd.DataFrame([prepared_features], columns=FEATURE_COLUMNS)
    prediction = model.predict(input_df)[0]
    return round(float(prediction), 4)


def _build_parser():
    parser = argparse.ArgumentParser(description="Predict California house value")
    parser.add_argument("--MedInc", type=float, required=True)
    parser.add_argument("--HouseAge", type=float, required=True)
    parser.add_argument("--AveRooms", type=float, required=True)
    parser.add_argument("--AveBedrms", type=float, required=True)
    parser.add_argument("--Population", type=float, required=True)
    parser.add_argument("--AveOccup", type=float, required=True)
    parser.add_argument("--Latitude", type=float, required=True)
    parser.add_argument("--Longitude", type=float, required=True)
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    output = predict_price(vars(args))
    print(f"Predicted house value (in $100k units): {output}")
