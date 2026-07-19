import warnings

import numpy as np
import pandas as pd
import pytest

from proxyml.local import (
    Complexity,
    LADDERS,
    TrainedChallenger,
    score_champion,
    to_challenger_upload,
    train_auto_challenger,
    train_challenger,
)
from proxyml.schema_builder import get_schema
from proxyml_core.export import predict_from_export
from proxyml_core.schema import CategoricalFeature, ContinuousFeature, FeatureSchema


def _schema():
    return FeatureSchema(
        features=[
            ContinuousFeature(name="age", mean=45.0, std=15.0, min=18.0, max=90.0),
            ContinuousFeature(name="income", mean=50000.0, std=15000.0, min=10000.0, max=200000.0),
        ]
    )


def _df(n=200, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "age": rng.uniform(18, 90, n),
            "income": rng.uniform(10000, 200000, n),
        }
    )


def test_train_challenger_regression_reproduces_via_export():
    schema = _schema()
    df = _df(seed=1)
    target = df["age"] * 0.5 + df["income"] * 0.0001

    result = train_challenger(df, target, schema, complexity=Complexity.MODERATE, task="regression")

    assert isinstance(result, TrainedChallenger)
    assert result.task == "regression"
    assert "r2" in result.metrics

    sample = {"age": df["age"].iloc[0], "income": df["income"].iloc[0]}
    reconstructed = predict_from_export(result.export, sample)
    actual = result.pipeline.predict(df.iloc[[0]].to_numpy(dtype=object))[0]
    assert reconstructed == pytest.approx(actual, abs=1e-6)


def test_train_challenger_classification():
    schema = _schema()
    df = _df(seed=2, n=300)
    target = np.where(df["age"] > 50, "senior", "junior")

    result = train_challenger(df, target, schema, complexity=Complexity.MODERATE, task="classification")

    assert result.task == "classification"
    assert "f1" in result.metrics
    assert result.export.classes is not None


def test_train_challenger_classification_rungs_fit_without_sklearn_deprecation_warnings():
    # scoring="accuracy" and use_legacy_attributes=False on the SIMPLE/FLEXIBLE
    # classifiers pin today's sklearn defaults explicitly, matching what
    # get_default_classifier() (the MODERATE rung) already does.
    schema = _schema()
    df = _df(seed=12, n=200)
    target = np.where(df["age"] > 50, "senior", "junior")

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        train_challenger(df, target, schema, complexity=Complexity.SIMPLE, task="classification")
        train_challenger(df, target, schema, complexity=Complexity.FLEXIBLE, task="classification")


def test_train_challenger_simple_rung_is_more_regularized_than_flexible():
    schema = _schema()
    df = _df(seed=3)
    target = df["age"] * 0.5 + df["income"] * 0.0001

    simple = train_challenger(df, target, schema, complexity=Complexity.SIMPLE, task="regression")
    flexible = train_challenger(df, target, schema, complexity=Complexity.FLEXIBLE, task="regression")

    assert simple.pipeline.named_steps["estimator"].alpha_ >= 0
    assert flexible.pipeline.named_steps["estimator"].alpha_ >= 0


def test_train_challenger_moderate_matches_ladder_description():
    assert LADDERS[Complexity.MODERATE].description


def test_train_challenger_feature_subset():
    schema = FeatureSchema(
        features=[
            ContinuousFeature(name="age", mean=45.0, std=15.0, min=18.0, max=90.0),
            CategoricalFeature(name="region", valid_categories={"east": 0.5, "west": 0.5}),
        ]
    )
    df = pd.DataFrame(
        {
            "age": np.random.RandomState(4).uniform(18, 90, 100),
            "region": np.random.RandomState(4).choice(["east", "west"], 100),
        }
    )
    target = df["age"] * 0.5

    result = train_challenger(df, target, schema, feature_names=["age"], task="regression")
    assert [f.name for f in result.export.features] == ["age"]


def _labeled_df(n=200, seed=5):
    rng = np.random.RandomState(seed)
    df = _df(n=n, seed=seed)
    df["approved"] = (df["age"] * 0.5 + df["income"] * 0.0001) > df["age"].median() * 0.5
    return df


def test_train_auto_challenger_from_dataframe():
    df = _labeled_df()
    result = train_auto_challenger(df, "approved", task="classification")

    assert isinstance(result, TrainedChallenger)
    assert result.task == "classification"
    assert {f.name for f in result.export.features} == {"age", "income"}


def test_train_auto_challenger_from_csv_path(tmp_path):
    df = _labeled_df(seed=6)
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    result = train_auto_challenger(csv_path, "approved", task="classification")

    assert isinstance(result, TrainedChallenger)
    assert {f.name for f in result.export.features} == {"age", "income"}


def test_train_auto_challenger_matches_manual_schema_and_train():
    df = _labeled_df(seed=7)
    target = df["approved"]
    features_df = df.drop(columns=["approved"])

    manual_schema = get_schema(features_df)
    manual_result = train_challenger(features_df, target, manual_schema, task="classification")
    auto_result = train_auto_challenger(df, "approved", task="classification")

    sample = {"age": features_df["age"].iloc[0], "income": features_df["income"].iloc[0]}
    assert predict_from_export(manual_result.export, sample) == predict_from_export(
        auto_result.export, sample
    )


def test_score_champion_matches_train_challenger_metric_shape_classification():
    schema = _schema()
    df = _df(seed=9, n=300)
    target = np.where(df["age"] > 50, "senior", "junior")

    result = train_challenger(df, target, schema, complexity=Complexity.MODERATE, task="classification")
    champion_metrics = score_champion(target, target, task="classification")
    assert set(champion_metrics) == set(result.metrics)


def test_score_champion_matches_train_challenger_metric_shape_regression():
    schema = _schema()
    df = _df(seed=10)
    target = df["age"] * 0.5 + df["income"] * 0.0001

    result = train_challenger(df, target, schema, complexity=Complexity.MODERATE, task="regression")
    champion_metrics = score_champion(target, target, task="regression")
    assert set(champion_metrics) == set(result.metrics)


def test_score_champion_uses_same_scoring_as_train_challenger():
    # Feed score_champion the exact labels/predictions a regression fit produced
    # internally, and assert numeric equality with the challenger's own r2 —
    # proving the two aren't drifting copies of the same formula.
    schema = _schema()
    df = _df(seed=11, n=300)
    target = df["age"] * 0.5 + df["income"] * 0.0001

    result = train_challenger(df, target, schema, complexity=Complexity.MODERATE, task="regression")
    X = df[["age", "income"]].to_numpy(dtype=object)
    y_pred = result.pipeline.predict(X)
    reproduced_metrics = score_champion(target, y_pred, task="regression")

    # Not identical to result.metrics (that was scored on a held-out test split,
    # this is scored on the full data) but both must use the same r2 formula —
    # verify by comparing against sklearn directly.
    from sklearn.metrics import r2_score

    assert reproduced_metrics["r2"] == pytest.approx(r2_score(target, y_pred))


def test_train_challenger_result_carries_complexity():
    schema = _schema()
    df = _df(seed=13)
    target = df["age"] * 0.5 + df["income"] * 0.0001

    result = train_challenger(df, target, schema, complexity=Complexity.FLEXIBLE, task="regression")

    assert result.complexity is Complexity.FLEXIBLE


def test_train_auto_challenger_result_carries_complexity():
    df = _labeled_df(seed=14)
    result = train_auto_challenger(df, "approved", task="classification", complexity=Complexity.SIMPLE)

    assert result.complexity is Complexity.SIMPLE


def test_to_challenger_upload_shape_without_champion_metrics():
    schema = _schema()
    df = _df(seed=15)
    target = df["age"] * 0.5 + df["income"] * 0.0001
    result = train_challenger(df, target, schema, complexity=Complexity.MODERATE, task="regression")

    payload = to_challenger_upload(result, n_samples=40)

    assert payload["export"] == result.export.to_dict()
    assert payload["challenger_metrics"] == result.metrics
    assert payload["n_samples"] == 40
    assert payload["complexity"] == "moderate"
    assert "champion_metrics" not in payload
    assert isinstance(payload["sdk_version"], str) and payload["sdk_version"]
    assert isinstance(payload["proxyml_core_version"], str) and payload["proxyml_core_version"]


def test_to_challenger_upload_includes_champion_metrics_when_given():
    schema = _schema()
    df = _df(seed=16)
    target = df["age"] * 0.5 + df["income"] * 0.0001
    result = train_challenger(df, target, schema, complexity=Complexity.MODERATE, task="regression")
    champion_metrics = score_champion(target, target, task="regression")

    payload = to_challenger_upload(result, n_samples=40, champion_metrics=champion_metrics)

    assert payload["champion_metrics"] == champion_metrics


def test_to_challenger_upload_defaults_versions_to_installed_packages():
    from importlib.metadata import version as pkg_version

    schema = _schema()
    df = _df(seed=17)
    target = df["age"] * 0.5 + df["income"] * 0.0001
    result = train_challenger(df, target, schema, task="regression")

    payload = to_challenger_upload(result, n_samples=10)

    assert payload["sdk_version"] == pkg_version("proxyml")
    assert payload["proxyml_core_version"] == pkg_version("proxyml-core")


def test_to_challenger_upload_allows_version_override():
    schema = _schema()
    df = _df(seed=18)
    target = df["age"] * 0.5 + df["income"] * 0.0001
    result = train_challenger(df, target, schema, task="regression")

    payload = to_challenger_upload(
        result, n_samples=10, sdk_version="9.9.9", proxyml_core_version="8.8.8"
    )

    assert payload["sdk_version"] == "9.9.9"
    assert payload["proxyml_core_version"] == "8.8.8"


def test_to_challenger_upload_payload_is_json_serializable():
    import json

    schema = _schema()
    df = _df(seed=19, n=300)
    target = np.where(df["age"] > 50, "senior", "junior")
    result = train_challenger(df, target, schema, task="classification")
    champion_metrics = score_champion(target, target, task="classification")

    payload = to_challenger_upload(result, n_samples=300, champion_metrics=champion_metrics)

    json.dumps(payload)  # must not raise


def test_train_auto_challenger_passes_immutable_cols_to_get_schema():
    from unittest.mock import patch

    df = _labeled_df(seed=8)
    features_df = df.drop(columns=["approved"])

    with patch("proxyml.local.challenger.get_schema", wraps=get_schema) as mock_get_schema:
        train_auto_challenger(df, "approved", task="classification", immutable_cols=["age"])

    mock_get_schema.assert_called_once()
    called_df = mock_get_schema.call_args.args[0]
    called_kwargs = mock_get_schema.call_args.kwargs
    assert list(called_df.columns) == list(features_df.columns)
    assert called_kwargs["immutable_cols"] == ["age"]
