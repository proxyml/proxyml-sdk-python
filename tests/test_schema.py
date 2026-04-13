import numpy as np
import pandas as pd
import pytest

from proxyml.schema import (
    gen_continuous_schema,
    gen_categorical_schema,
    gen_discrete_schema,
    get_schema,
)


def test_gen_continuous_schema():
    s = pd.Series([1.0, 2.0, 3.0, 4.0], name="weight")
    result = gen_continuous_schema(s, name=None)
    assert result["type"] == "continuous"
    assert result["name"] == "weight"
    assert result["mean"] == pytest.approx(2.5)
    assert result["min"] == pytest.approx(1.0)
    assert result["max"] == pytest.approx(4.0)


def test_gen_continuous_schema_name_override():
    s = pd.Series([1.0, 2.0], name="original")
    result = gen_continuous_schema(s, name="overridden")
    assert result["name"] == "overridden"


def test_gen_categorical_schema():
    s = pd.Series(["a", "a", "b", "c"], name="color")
    result = gen_categorical_schema(s, name=None)
    assert result["type"] == "categorical"
    assert result["name"] == "color"
    assert set(result["valid_categories"].keys()) == {"a", "b", "c"}
    assert result["valid_categories"]["a"] == pytest.approx(0.5)


def test_gen_categorical_schema_bool():
    s = pd.Series([True, False, True, True], name="flag")
    result = gen_categorical_schema(s, name=None)
    assert result["type"] == "categorical"
    assert True in result["valid_categories"]


def test_gen_discrete_schema():
    s = pd.Series([0, 1, 2, 3, 4], name="count")
    result = gen_discrete_schema(s, name=None)
    assert result["type"] == "count"
    assert result["name"] == "count"
    assert result["lambda"] == pytest.approx(2.0)
    assert result["max"] == 4


def test_get_schema_infers_types():
    df = pd.DataFrame({
        "f_float": pd.array([1.0, 2.0, 3.0], dtype=float),
        "f_int": pd.array([1, 2, 3], dtype=int),
        "f_str": ["x", "y", "z"],
        "f_bool": [True, False, True],
    })
    schema = get_schema(df, immutable_cols=None)
    types = {f["name"]: f["type"] for f in schema["features"]}
    assert types["f_float"] == "continuous"
    assert types["f_int"] == "count"
    assert types["f_str"] == "categorical"
    assert types["f_bool"] == "categorical"


def test_get_schema_immutable_cols():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
    schema = get_schema(df, immutable_cols=["b"])
    immutable = {f["name"]: f["immutable"] for f in schema["features"]}
    assert immutable["a"] is False
    assert immutable["b"] is True
    assert immutable["c"] is False


def test_get_schema_no_immutable_cols():
    df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    schema = get_schema(df, immutable_cols=None)
    assert all(not f["immutable"] for f in schema["features"])
