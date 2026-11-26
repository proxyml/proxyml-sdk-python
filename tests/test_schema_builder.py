import numpy as np
import pandas as pd

from proxyml.schema_builder import get_schema, get_schema_like
from proxyml_core.drift import compare_schemas
from proxyml_core.schema import (
    CategoricalFeature,
    CategoricalOrdinalFeature,
    ContinuousFeature,
    CountFeature,
    FeatureSchema,
    NumericOrdinalFeature,
)


def test_get_schema_infers_continuous_for_float_columns():
    df = pd.DataFrame({"age": [20.0, 30.0, 40.0, 50.0]})
    schema = get_schema(df)
    assert isinstance(schema, FeatureSchema)
    assert len(schema.features) == 1
    feature = schema.features[0]
    assert isinstance(feature, ContinuousFeature)
    assert feature.name == "age"
    assert feature.min == 20.0
    assert feature.max == 50.0


def test_get_schema_infers_count_for_integer_columns():
    df = pd.DataFrame({"visits": [1, 2, 3, 4, 5]})
    schema = get_schema(df)
    feature = schema.features[0]
    assert isinstance(feature, CountFeature)
    assert feature.name == "visits"
    assert feature.max == 5


def test_get_schema_infers_categorical_for_bool_columns():
    df = pd.DataFrame({"flag": [True, False, True, True]})
    schema = get_schema(df)
    feature = schema.features[0]
    assert isinstance(feature, CategoricalFeature)
    assert set(feature.valid_categories) == {True, False}


def test_get_schema_infers_categorical_for_object_columns():
    df = pd.DataFrame({"colour": ["red", "blue", "red", "green"]})
    schema = get_schema(df)
    feature = schema.features[0]
    assert isinstance(feature, CategoricalFeature)
    assert np.isclose(sum(feature.valid_categories.values()), 1.0)


def test_get_schema_marks_immutable_columns():
    df = pd.DataFrame({"age": [20.0, 30.0], "id": [1, 2]})
    schema = get_schema(df, immutable_cols=["id"])
    by_name = {f.name: f for f in schema.features}
    assert by_name["id"].immutable is True
    assert by_name["age"].immutable is False


def test_get_schema_warns_on_unknown_immutable_col(caplog):
    df = pd.DataFrame({"age": [20.0, 30.0]})
    with caplog.at_level("WARNING"):
        get_schema(df, immutable_cols=["nonexistent"])
    assert "nonexistent" in caplog.text


# ── get_schema_like ──────────────────────────────────────────────────────────

BASELINE = FeatureSchema(features=[
    NumericOrdinalFeature(name="grade", categories=[1, 2, 3], mean=2.0, std=0.8),
    CategoricalOrdinalFeature(name="size", categories=["S", "M", "L"], probabilities=[0.3, 0.4, 0.3], immutable=True),
    CountFeature(name="visits", lambda_=3.0, max=9),
    ContinuousFeature(name="score", mean=0.0, std=1.0, min=-3.0, max=3.0),
    CategoricalFeature(name="code", valid_categories={1: 0.5, 2: 0.5}),
])


def _like(df):
    return {f.name: f for f in get_schema_like(df, BASELINE).features}


def test_get_schema_like_follows_baseline_types_when_values_fit():
    got = _like(pd.DataFrame({
        "grade": [1, 2, 3, 2], "size": ["S", "M", "M", "L"], "visits": [1, 2, 3, 4],
        "score": [1, 0, 2, -1], "code": [1, 2, 2, 1],
    }))
    assert {n: f.type for n, f in got.items()} == {
        "grade": "numeric_ordinal", "size": "categorical_ordinal", "visits": "count",
        "score": "continuous", "code": "categorical",
    }
    assert got["grade"].categories == [1, 2, 3]
    assert got["size"].immutable is True  # carried over from the baseline
    assert got["code"].valid_categories == {1: 0.5, 2: 0.5}


def test_get_schema_like_float_column_holding_integers():
    # an integer column with a missing value arrives as float64 (1.0, 2.0, NaN)
    got = _like(pd.DataFrame({"grade": [1.0, 2.0, None, 3.0], "visits": [1.0, 2.0, None, 4.0], "code": [1.0, 2.0, None, 2.0]}))
    assert got["grade"].type == "numeric_ordinal" and got["grade"].categories == [1, 2, 3]
    assert got["visits"].type == "count" and got["visits"].max == 4
    assert set(got["code"].valid_categories) == {1, 2}  # 1.0 and 2.0, not "1.0"


def test_get_schema_like_keeps_baseline_order_and_appends_new_categories():
    got = _like(pd.DataFrame({"size": ["M", "M", "XL", "S"]}))["size"]
    assert got.categories == ["S", "M", "L", "XL"]
    assert got.probabilities == [0.25, 0.5, 0.0, 0.25]


def test_get_schema_like_does_not_mask_values_that_do_not_fit():
    got = _like(pd.DataFrame({
        "score": ["high", "low", "high"],   # text where the model has a continuous feature
        "visits": [1, -2, 3],               # a negative "count"
        "grade": [1.5, 2.5, 3.5],           # non-integer "ordinal"
    }))
    assert got["score"].type == "categorical"
    assert got["visits"].type == "continuous"
    assert got["grade"].type == "continuous"


def test_get_schema_like_extra_and_empty_columns():
    got = _like(pd.DataFrame({"brand_new": [1.5, 2.5], "grade": [None, None]}))
    assert got["brand_new"].type == "continuous"
    assert "grade" not in got  # nothing to summarize: left out so the server reports it removed


def test_get_schema_like_removes_the_false_type_change_from_a_real_comparison():
    df = pd.DataFrame({"grade": [1, 2, 3, 2, 2, 1] * 40, "size": ["S", "M", "M", "L", "M", "L"] * 40})
    base = FeatureSchema(features=[f for f in BASELINE.features if f.name in ("grade", "size")])
    inferred = compare_schemas(base, get_schema(df), len(df)).to_dict()
    aligned = compare_schemas(base, get_schema_like(df, base), len(df)).to_dict()
    assert inferred["summary"]["type_changed"] == 2
    assert aligned["summary"]["type_changed"] == 0
