import numpy as np
import pandas as pd

from proxyml.schema_builder import get_schema
from proxyml_core.schema import CategoricalFeature, ContinuousFeature, CountFeature, FeatureSchema


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
