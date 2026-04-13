import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from proxyml.client import (
    _cast_column,
    _headers,
    interpret_counterfactual,
    predict,
    put_schema,
    synthesize_data,
)


# ---------------------------------------------------------------------------
# _headers
# ---------------------------------------------------------------------------

def test_headers_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("PROXYML_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="PROXYML_API_KEY"):
        _headers()


def test_headers_returns_correct_dict(monkeypatch):
    monkeypatch.setenv("PROXYML_API_KEY", "test-key")
    h = _headers()
    assert h["X-API-KEY"] == "test-key"
    assert h["Content-Type"] == "application/json"


# ---------------------------------------------------------------------------
# _cast_column
# ---------------------------------------------------------------------------

def test_cast_column_continuous():
    s = pd.Series(["1.5", "2.5", "3.5"])
    result = _cast_column(s, "continuous")
    assert result.dtype == float


def test_cast_column_count():
    s = pd.Series([1.0, 2.0, 3.0])
    result = _cast_column(s, "count")
    assert result.dtype == int


def test_cast_column_numeric_ordinal():
    s = pd.Series([1.0, 2.0, 3.0])
    result = _cast_column(s, "numeric_ordinal")
    assert result.dtype == int


def test_cast_column_categorical_bool_strings():
    s = pd.Series(["true", "false", "true"])
    result = _cast_column(s, "categorical")
    assert result.tolist() == [True, False, True]


def test_cast_column_categorical_passthrough():
    s = pd.Series(["a", "b", "c"])
    result = _cast_column(s, "categorical")
    assert result.tolist() == ["a", "b", "c"]


def test_cast_column_unknown_type_passthrough():
    s = pd.Series([1, 2, 3])
    result = _cast_column(s, "unknown_type")
    assert result.tolist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# interpret_counterfactual
# ---------------------------------------------------------------------------

def test_interpret_counterfactual_prediction_changed():
    result = interpret_counterfactual(
        sample={"age": 30, "income": 50000},
        counterfactual={"age": 30, "income": 75000},
        prediction_changed=True,
        exclude_from_diff=None,
    )
    assert "income from 50000 to 75000" in result
    assert "different prediction" in result
    assert "surrogate model" in result


def test_interpret_counterfactual_prediction_unchanged():
    result = interpret_counterfactual(
        sample={"age": 30, "income": 50000},
        counterfactual={"age": 30, "income": 75000},
        prediction_changed=False,
        exclude_from_diff=None,
    )
    assert "income from 50000 to 75000" in result
    assert "did not change" in result


def test_interpret_counterfactual_no_diffs():
    result = interpret_counterfactual(
        sample={"age": 30},
        counterfactual={"age": 30},
        prediction_changed=True,
        exclude_from_diff=None,
    )
    assert result == "No meaningful differences found between sample and counterfactual."


def test_interpret_counterfactual_exclude_from_diff():
    result = interpret_counterfactual(
        sample={"age": 30, "id": 999},
        counterfactual={"age": 35, "id": 888},
        prediction_changed=True,
        exclude_from_diff=["id"],
    )
    assert "id" not in result
    assert "age from 30 to 35" in result


# ---------------------------------------------------------------------------
# API functions (mocked HTTP)
# ---------------------------------------------------------------------------

def _mock_response(status_code, json_body):
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = json_body
    r.text = str(json_body)
    return r


@patch("proxyml.client.put")
def test_put_schema_success(mock_put):
    mock_put.return_value = _mock_response(200, {"status": "ok"})
    result = put_schema({"features": []})
    assert result == {"status": "ok"}
    mock_put.assert_called_once_with(endpoint="/schema", payload={"features": []})


@patch("proxyml.client.put")
def test_put_schema_failure(mock_put):
    mock_put.return_value = _mock_response(422, {"detail": "invalid"})
    result = put_schema({"features": []})
    assert result is None


@patch("proxyml.client.post")
def test_predict_success(mock_post):
    mock_post.return_value = _mock_response(200, {"prediction": 1, "probability": 0.9})
    result = predict(samples=[1.0, 2.0, 3.0], version=None)
    assert result == {"prediction": 1, "probability": 0.9}
    payload = mock_post.call_args.kwargs["payload"]
    assert payload["inputs"] == [1.0, 2.0, 3.0]
    assert "version" not in payload


@patch("proxyml.client.post")
def test_predict_with_version(mock_post):
    mock_post.return_value = _mock_response(200, {"prediction": 0})
    predict(samples=[1.0, 2.0], version=3)
    payload = mock_post.call_args.kwargs["payload"]
    assert payload["version"] == 3


@patch("proxyml.client.post")
def test_predict_failure_returns_none(mock_post):
    mock_post.return_value = _mock_response(422, {"detail": "bad input"})
    result = predict(samples=[1.0, 2.0], version=None)
    assert result is None


@patch("proxyml.client.post")
def test_synthesize_data_no_sample(mock_post):
    mock_post.return_value = _mock_response(200, {
        "samples": [[1.0, "true"], [2.5, "false"]],
        "feature_names": ["f_cont", "f_cat"],
        "feature_types": ["continuous", "categorical"],
    })
    df = synthesize_data(num_points=2, sample=None)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["f_cont", "f_cat"]
    mock_post.assert_called_once_with(
        endpoint="/synthesize/neighbors", payload={"n": 2}
    )


@patch("proxyml.client.post")
def test_synthesize_data_failure_returns_none(mock_post):
    mock_post.return_value = _mock_response(500, {})
    result = synthesize_data(num_points=10, sample=None)
    assert result is None
