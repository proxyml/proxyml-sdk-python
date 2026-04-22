import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from proxyml.client import (
    _cast_column,
    _headers,
    delete_model,
    delete_schema,
    diff_models,
    fetch_schema,
    find_counterfactuals,
    get_model_summary,
    get_usage,
    interpret_counterfactual,
    list_models,
    list_schemas,
    predict,
    predict_batch,
    put_schema,
    rotate_key,
    synthesize_data,
    train_surrogate,
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
def test_put_schema_default(mock_put):
    mock_put.return_value = _mock_response(200, {"features": []})
    result = put_schema({"features": []})
    assert result == {"features": []}
    mock_put.assert_called_once_with(endpoint="/schema/default", payload={"features": []})


@patch("proxyml.client.put")
def test_put_schema_named(mock_put):
    mock_put.return_value = _mock_response(200, {"features": []})
    put_schema({"features": []}, name="credit")
    mock_put.assert_called_once_with(endpoint="/schema/credit", payload={"features": []})


@patch("proxyml.client.put")
def test_put_schema_failure(mock_put):
    mock_put.return_value = _mock_response(422, {"detail": "invalid"})
    result = put_schema({"features": []})
    assert result is None


# ---------------------------------------------------------------------------
# fetch_schema / list_schemas / delete_schema
# ---------------------------------------------------------------------------

@patch("proxyml.client.get")
def test_fetch_schema_default(mock_get):
    mock_get.return_value = _mock_response(200, {"features": [{"type": "continuous", "name": "age"}]})
    result = fetch_schema()
    assert result["features"][0]["name"] == "age"
    mock_get.assert_called_once_with(endpoint="/schema/default", params={})


@patch("proxyml.client.get")
def test_fetch_schema_named(mock_get):
    mock_get.return_value = _mock_response(200, {"features": []})
    fetch_schema(name="credit")
    mock_get.assert_called_once_with(endpoint="/schema/credit", params={})


@patch("proxyml.client.get")
def test_fetch_schema_not_found_returns_none(mock_get):
    mock_get.return_value = _mock_response(404, {"detail": "not found"})
    assert fetch_schema("missing") is None


@patch("proxyml.client.get")
def test_list_schemas_success(mock_get):
    schemas = [{"name": "default", "updated_at": "2026-04-22T10:00:00"},
               {"name": "credit", "updated_at": "2026-04-22T11:00:00"}]
    mock_get.return_value = _mock_response(200, {"schemas": schemas})
    result = list_schemas()
    assert result == schemas
    mock_get.assert_called_once_with(endpoint="/schemas", params={})


@patch("proxyml.client.get")
def test_list_schemas_failure_returns_none(mock_get):
    mock_get.return_value = _mock_response(401, {"detail": "unauthorized"})
    assert list_schemas() is None


@patch("proxyml.client.delete")
def test_delete_schema_success(mock_del):
    mock_del.return_value = _mock_response(204, None)
    assert delete_schema("credit") is True
    mock_del.assert_called_once_with(endpoint="/schema/credit")


@patch("proxyml.client.delete")
def test_delete_schema_not_found(mock_del):
    mock_del.return_value = _mock_response(404, {"detail": "not found"})
    assert delete_schema("no-such-schema") is False


@patch("proxyml.client.post")
def test_predict_success(mock_post):
    mock_post.return_value = _mock_response(200, {"prediction": 1, "probability": 0.9})
    result = predict(sample=[1.0, 2.0, 3.0], version=None)
    assert result == {"prediction": 1, "probability": 0.9}
    payload = mock_post.call_args.kwargs["payload"]
    assert payload["inputs"] == [1.0, 2.0, 3.0]
    assert "version" not in payload


@patch("proxyml.client.post")
def test_predict_with_version(mock_post):
    mock_post.return_value = _mock_response(200, {"prediction": 0})
    uid = "550e8400-e29b-41d4-a716-446655440000"
    predict(sample=[1.0, 2.0], version=uid)
    payload = mock_post.call_args.kwargs["payload"]
    assert payload["version"] == uid


@patch("proxyml.client.post")
def test_predict_failure_returns_none(mock_post):
    mock_post.return_value = _mock_response(422, {"detail": "bad input"})
    result = predict(sample=[1.0, 2.0], version=None)
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
    payload = mock_post.call_args.kwargs["payload"]
    assert payload["n"] == 2
    assert payload["schema_name"] == "default"


@patch("proxyml.client.post")
def test_synthesize_data_named_schema(mock_post):
    mock_post.return_value = _mock_response(200, {
        "samples": [[1.0]], "feature_names": ["x"], "feature_types": ["continuous"],
    })
    synthesize_data(num_points=1, schema_name="credit")
    payload = mock_post.call_args.kwargs["payload"]
    assert payload["schema_name"] == "credit"


@patch("proxyml.client.post")
def test_synthesize_data_failure_returns_none(mock_post):
    mock_post.return_value = _mock_response(500, {})
    result = synthesize_data(num_points=10, sample=None)
    assert result is None


@patch("proxyml.client.post")
def test_train_surrogate_with_metadata(mock_post):
    mock_post.return_value = _mock_response(200, {
        "version": "abc-123", "trained_at": "2026-04-19T12:00:00",
        "task": "regression", "name": "v1", "comments": "test run",
        "feature_names": None, "metrics": {"r2": 0.95}, "warning": None,
    })
    result = train_surrogate(
        samples=[[1.0, 2.0]], predictions=[3.0],
        feature_names=None, name="v1", comments="test run",
    )
    payload = mock_post.call_args.kwargs["payload"]
    assert payload["name"] == "v1"
    assert payload["comments"] == "test run"
    assert payload["schema_name"] == "default"
    assert result["version"] == "abc-123"
    assert result["trained_at"] == "2026-04-19T12:00:00"


@patch("proxyml.client.post")
def test_train_surrogate_named_schema(mock_post):
    mock_post.return_value = _mock_response(200, {"version": "abc-123"})
    train_surrogate(samples=[[1.0]], predictions=[1.0], feature_names=None, schema_name="credit")
    payload = mock_post.call_args.kwargs["payload"]
    assert payload["schema_name"] == "credit"


@patch("proxyml.client.post")
def test_train_surrogate_omits_none_metadata(mock_post):
    mock_post.return_value = _mock_response(200, {"version": "abc-123"})
    train_surrogate(samples=[[1.0]], predictions=[1.0], feature_names=None)
    payload = mock_post.call_args.kwargs["payload"]
    assert "name" not in payload
    assert "comments" not in payload


@patch("proxyml.client.get")
def test_list_models_success(mock_get):
    models = [{"version": "abc-123", "task": "regression", "name": "v1",
               "comments": None, "feature_names": None, "metrics": {"r2": 0.9},
               "trained_at": "2026-04-19T12:00:00"}]
    mock_get.return_value = _mock_response(200, {"models": models})
    result = list_models()
    assert result == models
    mock_get.assert_called_once_with(endpoint="/surrogate/models", params={})


@patch("proxyml.client.get")
def test_list_models_failure_returns_none(mock_get):
    mock_get.return_value = _mock_response(401, {"detail": "unauthorized"})
    assert list_models() is None


@patch("proxyml.client.delete")
def test_delete_model_success(mock_delete):
    mock_delete.return_value = _mock_response(204, None)
    assert delete_model("abc-123") is True
    mock_delete.assert_called_once_with(endpoint="/surrogate/models/abc-123")


@patch("proxyml.client.delete")
def test_delete_model_not_found(mock_delete):
    mock_delete.return_value = _mock_response(404, {"detail": "not found"})
    assert delete_model("no-such-id") is False


# ---------------------------------------------------------------------------
# predict_batch
# ---------------------------------------------------------------------------

@patch("proxyml.client.post")
def test_predict_batch_success(mock_post):
    mock_post.return_value = _mock_response(200, {
        "predictions": [0.74, 0.31],
        "model_version": "surrogate-abc-regression",
    })
    result = predict_batch(samples=[[1.0, 2.0], [3.0, 4.0]])
    assert result["predictions"] == [0.74, 0.31]
    payload = mock_post.call_args.kwargs["payload"]
    assert payload["inputs"] == [[1.0, 2.0], [3.0, 4.0]]
    assert "version" not in payload


@patch("proxyml.client.post")
def test_predict_batch_with_version(mock_post):
    mock_post.return_value = _mock_response(200, {"predictions": [1], "model_version": "surrogate-abc-classification"})
    uid = "550e8400-e29b-41d4-a716-446655440000"
    predict_batch(samples=[[1.0]], version=uid)
    payload = mock_post.call_args.kwargs["payload"]
    assert payload["version"] == uid


@patch("proxyml.client.post")
def test_predict_batch_failure_returns_none(mock_post):
    mock_post.return_value = _mock_response(422, {"detail": "bad input"})
    assert predict_batch(samples=[[1.0, 2.0]]) is None


# ---------------------------------------------------------------------------
# find_counterfactuals
# ---------------------------------------------------------------------------

_BATCH_CF_RESPONSE = {
    "results": [
        {"counterfactual": [1.5, "yes"], "outlier_score": 0.1, "warning": None},
        {"counterfactual": None, "outlier_score": 0.9, "warning": "no CF found"},
    ],
    "feature_names": ["f_cont", "f_cat"],
    "feature_types": ["continuous", "categorical"],
    "task": "classification",
    "target_label": "high",
    "model_version": "surrogate-abc-classification",
}


@patch("proxyml.client.post")
def test_find_counterfactuals_as_df(mock_post):
    mock_post.return_value = _mock_response(200, _BATCH_CF_RESPONSE)
    results = find_counterfactuals(samples=[[1.0, "no"], [2.0, "no"]], target="high")
    assert len(results) == 2
    assert isinstance(results[0], pd.DataFrame)
    assert results[0]["f_cont"].iloc[0] == 1.5
    assert results[1] is None  # no counterfactual for second instance


@patch("proxyml.client.post")
def test_find_counterfactuals_raw(mock_post):
    mock_post.return_value = _mock_response(200, _BATCH_CF_RESPONSE)
    result = find_counterfactuals(samples=[[1.0, "no"]], target="high", as_dfs=False)
    assert result == _BATCH_CF_RESPONSE


@patch("proxyml.client.post")
def test_find_counterfactuals_payload(mock_post):
    mock_post.return_value = _mock_response(200, _BATCH_CF_RESPONSE)
    uid = "550e8400-e29b-41d4-a716-446655440000"
    find_counterfactuals(
        samples=[[1.0, "no"]], target="high",
        n_neighbors=500, perturbation_scale=0.2, version=uid, as_dfs=False,
    )
    payload = mock_post.call_args.kwargs["payload"]
    assert payload["instances"] == [[1.0, "no"]]
    assert payload["target_label"] == "high"
    assert payload["n_neighbors"] == 500
    assert payload["perturbation_scale"] == 0.2
    assert payload["version"] == uid


@patch("proxyml.client.post")
def test_find_counterfactuals_failure_returns_none(mock_post):
    mock_post.return_value = _mock_response(404, {"detail": "no surrogate"})
    assert find_counterfactuals(samples=[[1.0]], target="high") is None


# ---------------------------------------------------------------------------
# get_model_summary
# ---------------------------------------------------------------------------

_SUMMARY_RESPONSE = {
    "model_version": "abc-123",
    "task": "regression",
    "trained_at": "2026-04-20T10:00:00",
    "name": "v2",
    "comments": None,
    "feature_names": ["MedInc", "Latitude"],
    "metrics": {"r2": 0.92},
    "feature_importances": [
        {"feature": "MedInc", "coefficient": 0.82, "abs_coefficient": 0.82},
        {"feature": "Latitude", "coefficient": -0.61, "abs_coefficient": 0.61},
    ],
    "per_class_importances": None,
    "note": "Coefficients are in the scaled feature space.",
}


@patch("proxyml.client.get")
def test_get_model_summary_success(mock_get):
    mock_get.return_value = _mock_response(200, _SUMMARY_RESPONSE)
    result = get_model_summary()
    assert result == _SUMMARY_RESPONSE
    mock_get.assert_called_once_with(endpoint="/explain/summary", params={})


@patch("proxyml.client.get")
def test_get_model_summary_with_version(mock_get):
    mock_get.return_value = _mock_response(200, _SUMMARY_RESPONSE)
    get_model_summary(version="abc-123")
    mock_get.assert_called_once_with(endpoint="/explain/summary", params={"version": "abc-123"})


@patch("proxyml.client.get")
def test_get_model_summary_failure_returns_none(mock_get):
    mock_get.return_value = _mock_response(404, {"detail": "not found"})
    assert get_model_summary() is None


# ---------------------------------------------------------------------------
# diff_models
# ---------------------------------------------------------------------------

_DIFF_RESPONSE = {
    "version_a": "aaa-111",
    "version_b": "bbb-222",
    "task": "regression",
    "metric_diff": {"r2": {"a": 0.87, "b": 0.92, "delta": 0.05}},
    "coefficient_diff": [
        {"feature": "MedInc", "a": 0.82, "b": 0.76, "delta": -0.06},
    ],
    "features_added": [],
    "features_removed": ["Population"],
}


@patch("proxyml.client.get")
def test_diff_models_success(mock_get):
    mock_get.return_value = _mock_response(200, _DIFF_RESPONSE)
    result = diff_models(version_a="aaa-111", version_b="bbb-222")
    assert result == _DIFF_RESPONSE
    mock_get.assert_called_once_with(
        endpoint="/explain/diff",
        params={"version_a": "aaa-111", "version_b": "bbb-222"},
    )


@patch("proxyml.client.get")
def test_diff_models_failure_returns_none(mock_get):
    mock_get.return_value = _mock_response(422, {"detail": "different tasks"})
    assert diff_models(version_a="aaa-111", version_b="bbb-222") is None


# ---------------------------------------------------------------------------
# get_usage
# ---------------------------------------------------------------------------

_USAGE_RESPONSE = {
    "tier": "hobbyist",
    "period": "2026-04",
    "calls_this_period": 42,
    "calls_limit": 1000,
    "calls_remaining": 958,
    "surrogates_trained": 2,
    "surrogate_limit": 3,
}


@patch("proxyml.client.get")
def test_get_usage_success(mock_get):
    mock_get.return_value = _mock_response(200, _USAGE_RESPONSE)
    result = get_usage()
    assert result == _USAGE_RESPONSE
    mock_get.assert_called_once_with(endpoint="/account/usage", params={})


@patch("proxyml.client.get")
def test_get_usage_failure_returns_none(mock_get):
    mock_get.return_value = _mock_response(401, {"detail": "unauthorized"})
    assert get_usage() is None


# ---------------------------------------------------------------------------
# rotate_key
# ---------------------------------------------------------------------------

@patch("proxyml.client.post")
def test_rotate_key_success(mock_post):
    mock_post.return_value = _mock_response(201, {"api_key": "proxyml_new_secret_key"})
    result = rotate_key()
    assert result == "proxyml_new_secret_key"
    mock_post.assert_called_once_with(endpoint="/account/keys/rotate", payload={})


@patch("proxyml.client.post")
def test_rotate_key_failure_returns_none(mock_post):
    mock_post.return_value = _mock_response(403, {"detail": "Key rotation is not available for test accounts"})
    assert rotate_key() is None
