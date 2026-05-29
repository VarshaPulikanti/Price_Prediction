from src.train import train_model


def test_training_pipeline_runs(tmp_path):
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    registry_path = tmp_path / "model_registry.json"
    metrics = train_model(
        model_path=model_path,
        metrics_path=metrics_path,
        registry_path=registry_path,
        fast_mode=True,
    )

    assert model_path.exists()
    assert metrics_path.exists()
    assert registry_path.exists()
    assert "selected_model" in metrics
    assert "cross_validation" in metrics
    assert "test" in metrics
    assert metrics["test"]["r2"] > 0
