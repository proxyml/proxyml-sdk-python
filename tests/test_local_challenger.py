import numpy as np
import pandas as pd
import pytest

from proxyml.local import Complexity, LADDERS, TrainedChallenger, train_challenger
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
