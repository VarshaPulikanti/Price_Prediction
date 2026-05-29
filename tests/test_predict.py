from src.predict import predict_price, prepare_features
from src.train import train_model


def test_predict_price_returns_float(tmp_path):
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    registry_path = tmp_path / "registry.json"
    train_model(
        model_path=model_path,
        metrics_path=metrics_path,
        registry_path=registry_path,
        fast_mode=True,
    )

    payload = {
        "MedInc": 4.2,
        "HouseAge": 30.0,
        "AveRooms": 5.6,
        "AveBedrms": 1.1,
        "Population": 1200.0,
        "AveOccup": 3.0,
        "Latitude": 34.2,
        "Longitude": -118.2,
    }
    prediction = predict_price(payload, model_path=model_path)
    assert isinstance(prediction, float)


def test_prepare_features_creates_engineered_columns():
    payload = {
        "MedInc": 4.2,
        "HouseAge": 30.0,
        "AveRooms": 5.6,
        "AveBedrms": 1.1,
        "Population": 1200.0,
        "AveOccup": 3.0,
        "Latitude": 34.2,
        "Longitude": -118.2,
    }
    prepared = prepare_features(payload)

    assert "rooms_per_household" in prepared
    assert "bedrooms_per_room" in prepared
    assert "population_per_household" in prepared
    assert "income_per_room" in prepared
    assert "income_per_person" in prepared
    assert "geo_density" in prepared
    assert "lat_lon_interaction" in prepared
