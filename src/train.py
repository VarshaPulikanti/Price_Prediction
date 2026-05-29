import json
from pathlib import Path

import joblib
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODEL_PATH,
    MODEL_REGISTRY_PATH,
    MODELS_DIR,
    NUMERIC_COLUMNS,
    RANDOM_STATE,
)
from src.data import load_dataset, split_dataset


def _build_pipeline(model, use_scaler: bool) -> Pipeline:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if use_scaler:
        numeric_steps.append(("scaler", StandardScaler()))

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=numeric_steps),
                NUMERIC_COLUMNS,
            ),
        ]
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def _evaluate(model: Pipeline, X, y):
    predictions = model.predict(X)
    return {
        "r2": round(r2_score(y, predictions), 4),
        "rmse": round(mean_squared_error(y, predictions) ** 0.5, 4),
        "mae": round(mean_absolute_error(y, predictions), 4),
    }


def _candidate_models(fast_mode: bool = False) -> dict:
    if fast_mode:
        return {
            "ridge": Ridge(alpha=1.0),
            "random_forest": RandomForestRegressor(
                n_estimators=80,
                max_depth=12,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "hist_gradient_boosting": HistGradientBoostingRegressor(
                learning_rate=0.06,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                random_state=RANDOM_STATE,
            ),
        }
    return {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=260,
            max_depth=18,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            learning_rate=0.045,
            n_estimators=320,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_leaf_nodes=63,
            min_samples_leaf=15,
            l2_regularization=0.02,
            random_state=RANDOM_STATE,
        ),
    }


def _model_search_space(model_name: str) -> dict:
    spaces = {
        "random_forest": {
            "model__n_estimators": [220, 280, 340, 420],
            "model__max_depth": [14, 18, None],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", 0.8],
        },
        "extra_trees": {
            "model__n_estimators": [260, 340, 420, 520],
            "model__max_depth": [None, 18, 26],
            "model__min_samples_leaf": [1, 2, 3],
            "model__max_features": ["sqrt", "log2", 0.8],
        },
        "gradient_boosting": {
            "model__learning_rate": [0.03, 0.04, 0.05, 0.07],
            "model__n_estimators": [300, 420, 560],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "hist_gradient_boosting": {
            "model__learning_rate": [0.03, 0.04, 0.05, 0.07],
            "model__max_leaf_nodes": [31, 63, 127],
            "model__min_samples_leaf": [10, 15, 25, 35],
            "model__l2_regularization": [0.0, 0.02, 0.05, 0.1],
            "model__max_bins": [127, 255],
        },
        "ridge": {
            "model__alpha": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            "model__solver": ["auto", "svd", "cholesky", "lsqr"],
        },
    }
    return spaces.get(model_name, {})


def _use_scaler(model) -> bool:
    return isinstance(model, Ridge)


def train_model(
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    registry_path: Path = MODEL_REGISTRY_PATH,
    fast_mode: bool = False,
):
    df = load_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)

    cv_folds = 3 if fast_mode else 5
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    registry = {}
    best_name = ""
    best_rmse = float("inf")
    best_pipeline = None
    best_is_fitted = False
    candidate_pipelines = {}

    for model_name, model in _candidate_models(fast_mode=fast_mode).items():
        pipeline = _build_pipeline(model, use_scaler=_use_scaler(model))
        candidate_pipelines[model_name] = pipeline

        cv_scores = cross_val_score(
            pipeline,
            X_train[FEATURE_COLUMNS],
            y_train,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=1,
        )
        cv_rmse = float((-cv_scores).mean())
        registry[model_name] = {
            "baseline_cv_rmse": round(cv_rmse, 4),
            "tuned_cv_rmse": None,
            "best_params": {},
        }

        if cv_rmse < best_rmse:
            best_rmse = cv_rmse
            best_name = model_name
            best_pipeline = clone(pipeline)
            best_is_fitted = False

    if not fast_mode:
        ranked = sorted(registry.items(), key=lambda item: item[1]["baseline_cv_rmse"])
        top_for_tuning = [name for name, _ in ranked[:2]]

        for model_name in top_for_tuning:
            search_space = _model_search_space(model_name)
            if not search_space:
                continue

            search = RandomizedSearchCV(
                estimator=candidate_pipelines[model_name],
                param_distributions=search_space,
                n_iter=8,
                scoring="neg_root_mean_squared_error",
                n_jobs=1,
                cv=cv,
                random_state=RANDOM_STATE,
                refit=True,
            )
            search.fit(X_train[FEATURE_COLUMNS], y_train)

            tuned_rmse = float(-search.best_score_)
            registry[model_name]["tuned_cv_rmse"] = round(tuned_rmse, 4)
            registry[model_name]["best_params"] = search.best_params_

            if tuned_rmse < best_rmse:
                best_rmse = tuned_rmse
                best_name = f"{model_name}_tuned"
                best_pipeline = search.best_estimator_
                best_is_fitted = True

    if not best_is_fitted:
        best_pipeline.fit(X_train[FEATURE_COLUMNS], y_train)

    metrics = {
        "selected_model": best_name,
        "selection_cv_rmse": round(best_rmse, 4),
        "cross_validation": registry,
        "train": _evaluate(best_pipeline, X_train[FEATURE_COLUMNS], y_train),
        "validation": _evaluate(best_pipeline, X_val[FEATURE_COLUMNS], y_val),
        "test": _evaluate(best_pipeline, X_test[FEATURE_COLUMNS], y_test),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    results = train_model()
    print(json.dumps(results, indent=2))
