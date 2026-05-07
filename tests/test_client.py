import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from proxyml.client import (
    _cast_column,
    _base_url,
    _headers,
    delete_model,
    delete_schema,
    diff_models,
    explain_local,
    explain_local_batch,
    export_surrogate,
    fetch_schema,
    find_counterfactual,
    find_counterfactuals,
    get_feature_importances,
    get_model_schema,
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
    update_model,
)


# ---------------------------------------------------------------------------
# _headers
# ---------------------------------------------------------------------------

def test_base_url_default(monkeypatch):
    monkeypatch.delenv("PROXYML_BASE_URL", raising=False)
    assert _base_url() == "https://api.proxyml.ai/api/v1"


def test_base_url_reads_env_at_call_time(monkeypatch):
    monkeypatch.setenv("PROXYML_BASE_URL", "https://custom.example.com/api/v1")
    assert _base_url() == "https://custom.example.com/api/v1"


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


def test_cast_column_categorical_bool_strings_capitalized():
    s = pd.Series(["True", "False", "True"])
    result = _cast_column(s, "categorical")
    assert result.tolist() == [True, False, True]


def test_cast_column_categorical_python_bools():
    s = pd.Series([True, False, True])
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
               "trained_at": "2026-04-19T12:00:00", "mlflow_run_id": None}]
    mock_get.return_value = _mock_response(200, {"models": models, "total": 1})
    result = list_models()
    assert result == {"models": models, "total": 1}
    mock_get.assert_called_once_with(endpoint="/surrogate/models", params={"limit": 50, "offset": 0})


@patch("proxyml.client.get")
def test_list_models_pagination(mock_get):
    models = [{"version": f"v{i}", "task": "regression", "name": None,
               "comments": None, "feature_names": None, "metrics": None,
               "trained_at": "2026-05-01T00:00:00", "mlflow_run_id": None}
              for i in range(10)]
    mock_get.return_value = _mock_response(200, {"models": models, "total": 42})
    result = list_models(limit=10, offset=20)
    assert result["total"] == 42
    assert len(result["models"]) == 10
    mock_get.assert_called_once_with(endpoint="/surrogate/models", params={"limit": 10, "offset": 20})


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


# ---------------------------------------------------------------------------
# export_surrogate
# ---------------------------------------------------------------------------

_EXPORT_RESPONSE = {
    "version": "abc-123",
    "classes": [0, 1],
    "intercept": [0.5],
    "per_class_intercepts": None,
    "features": [{"name": "age", "type": "continuous"}],
    "scalers": {"age": {"mean": 35.0, "scale": 10.0}},
}


@patch("proxyml.client.get")
def test_export_surrogate_success(mock_get):
    mock_get.return_value = _mock_response(200, _EXPORT_RESPONSE)
    result = export_surrogate(version="abc-123")
    assert result == _EXPORT_RESPONSE


@patch("proxyml.client.get")
def test_export_surrogate_calls_correct_endpoint(mock_get):
    mock_get.return_value = _mock_response(200, _EXPORT_RESPONSE)
    export_surrogate(version="abc-123")
    mock_get.assert_called_once_with(endpoint="/surrogate/models/abc-123/export", params={})


@patch("proxyml.client.get")
def test_export_surrogate_failure_returns_none(mock_get):
    mock_get.return_value = _mock_response(404, {"detail": "model not found"})
    assert export_surrogate(version="no-such-version") is None


# ---------------------------------------------------------------------------
# find_counterfactual
# ---------------------------------------------------------------------------

_CF_RESPONSE = {
    "counterfactual": [1.5, "yes"],
    "feature_names": ["f_cont", "f_cat"],
    "feature_types": ["continuous", "categorical"],
    "outlier_score": 0.1,
    "warning": None,
    "task": "classification",
    "target_label": "high",
    "model_version": "surrogate-abc-classification",
}

_CF_RESPONSE_NONE = {
    "counterfactual": None,
    "feature_names": ["f_cont", "f_cat"],
    "feature_types": ["continuous", "categorical"],
    "outlier_score": 0.9,
    "warning": "no counterfactual found",
    "task": "classification",
    "target_label": "high",
    "model_version": "surrogate-abc-classification",
}


@patch("proxyml.client.post")
def test_find_counterfactual_as_df(mock_post):
    mock_post.return_value = _mock_response(200, _CF_RESPONSE)
    result = find_counterfactual(sample=[1.0, "no"], target="high")
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["f_cont", "f_cat"]
    assert result["f_cont"].iloc[0] == 1.5


@patch("proxyml.client.post")
def test_find_counterfactual_raw(mock_post):
    mock_post.return_value = _mock_response(200, _CF_RESPONSE)
    result = find_counterfactual(sample=[1.0, "no"], target="high", as_df=False)
    assert result == _CF_RESPONSE


@patch("proxyml.client.post")
def test_find_counterfactual_none_returns_none(mock_post):
    mock_post.return_value = _mock_response(200, _CF_RESPONSE_NONE)
    result = find_counterfactual(sample=[1.0, "no"], target="high")
    assert result is None


@patch("proxyml.client.post")
def test_find_counterfactual_payload(mock_post):
    mock_post.return_value = _mock_response(200, _CF_RESPONSE)
    uid = "550e8400-e29b-41d4-a716-446655440000"
    find_counterfactual(
        sample=[1.0, "no"], target="high",
        n_neighbors=500, perturbation_scale=0.2, version=uid,
    )
    payload = mock_post.call_args.kwargs["payload"]
    assert payload["instance"] == [1.0, "no"]
    assert payload["target_label"] == "high"
    assert payload["n_neighbors"] == 500
    assert payload["perturbation_scale"] == 0.2
    assert payload["version"] == uid


@patch("proxyml.client.post")
def test_find_counterfactual_no_version_in_payload(mock_post):
    mock_post.return_value = _mock_response(200, _CF_RESPONSE)
    find_counterfactual(sample=[1.0, "no"], target="high")
    payload = mock_post.call_args.kwargs["payload"]
    assert "version" not in payload


@patch("proxyml.client.post")
def test_find_counterfactual_failure_returns_none(mock_post):
    mock_post.return_value = _mock_response(404, {"detail": "no surrogate"})
    assert find_counterfactual(sample=[1.0, "no"], target="high") is None


# ---------------------------------------------------------------------------
# get_feature_importances
# ---------------------------------------------------------------------------

_IMPORTANCES_RESPONSE = {
    "feature_importances": [
        {"feature": "MedInc", "coefficient": 0.82, "abs_coefficient": 0.82},
        {"feature": "Latitude", "coefficient": -0.61, "abs_coefficient": 0.61},
    ],
    "per_class_importances": None,
    "model_version": "abc-123",
    "task": "regression",
    "note": "Coefficients are in the scaled feature space.",
}


@patch("proxyml.client.get")
def test_get_feature_importances_success(mock_get):
    mock_get.return_value = _mock_response(200, _IMPORTANCES_RESPONSE)
    result = get_feature_importances()
    assert result == _IMPORTANCES_RESPONSE
    mock_get.assert_called_once_with(endpoint="/explain/importance", params={})


@patch("proxyml.client.get")
def test_get_feature_importances_with_version(mock_get):
    mock_get.return_value = _mock_response(200, _IMPORTANCES_RESPONSE)
    get_feature_importances(version="abc-123")
    mock_get.assert_called_once_with(endpoint="/explain/importance", params={"version": "abc-123"})


@patch("proxyml.client.get")
def test_get_feature_importances_failure_returns_none(mock_get):
    mock_get.return_value = _mock_response(404, {"detail": "not found"})
    assert get_feature_importances() is None


# ---------------------------------------------------------------------------
# get_model_schema
# ---------------------------------------------------------------------------

_MODEL_SCHEMA_RESPONSE = {
    "features": [
        {"type": "continuous", "name": "age", "mean": 35.0, "std": 10.0, "min": 18.0, "max": 90.0},
        {"type": "categorical", "name": "gender", "valid_categories": {"M": 0.5, "F": 0.5}},
    ]
}


@patch("proxyml.client.get")
def test_get_model_schema_success(mock_get):
    mock_get.return_value = _mock_response(200, _MODEL_SCHEMA_RESPONSE)
    result = get_model_schema(version="abc-123")
    assert result == _MODEL_SCHEMA_RESPONSE
    mock_get.assert_called_once_with(endpoint="/surrogate/models/abc-123/schema", params={})


@patch("proxyml.client.get")
def test_get_model_schema_failure_returns_none(mock_get):
    mock_get.return_value = _mock_response(404, {"detail": "not found"})
    assert get_model_schema(version="no-such-version") is None


# ---------------------------------------------------------------------------
# explain_local
# ---------------------------------------------------------------------------

_EXPLAIN_LOCAL_RESPONSE = {
    "prediction": 1,
    "feature_contributions": [
        {"feature": "MedInc", "contribution": 0.5, "abs_contribution": 0.5},
        {"feature": "Latitude", "contribution": -0.2, "abs_contribution": 0.2},
    ],
    "intercept": 0.1,
    "probabilities": [0.2, 0.8],
    "per_class_contributions": None,
}


@patch("proxyml.client.post")
def test_explain_local_success(mock_post):
    mock_post.return_value = _mock_response(200, _EXPLAIN_LOCAL_RESPONSE)
    result = explain_local(instance=[1.0, 2.0])
    assert result == _EXPLAIN_LOCAL_RESPONSE
    payload = mock_post.call_args.kwargs["payload"]
    assert payload["instance"] == [1.0, 2.0]
    assert "version" not in payload


@patch("proxyml.client.post")
def test_explain_local_with_version(mock_post):
    mock_post.return_value = _mock_response(200, _EXPLAIN_LOCAL_RESPONSE)
    uid = "550e8400-e29b-41d4-a716-446655440000"
    explain_local(instance=[1.0, 2.0], version=uid)
    payload = mock_post.call_args.kwargs["payload"]
    assert payload["version"] == uid


@patch("proxyml.client.post")
def test_explain_local_failure_returns_none(mock_post):
    mock_post.return_value = _mock_response(422, {"detail": "bad input"})
    assert explain_local(instance=[1.0, 2.0]) is None


# ---------------------------------------------------------------------------
# explain_local_batch
# ---------------------------------------------------------------------------

_EXPLAIN_LOCAL_BATCH_RESPONSE = {
    "results": [_EXPLAIN_LOCAL_RESPONSE, _EXPLAIN_LOCAL_RESPONSE],
    "model_version": "surrogate-abc-regression",
    "task": "regression",
    "schema_warning": None,
}


@patch("proxyml.client.post")
def test_explain_local_batch_success(mock_post):
    mock_post.return_value = _mock_response(200, _EXPLAIN_LOCAL_BATCH_RESPONSE)
    instances = [[1.0, 2.0], [3.0, 4.0]]
    result = explain_local_batch(instances=instances)
    assert result == _EXPLAIN_LOCAL_BATCH_RESPONSE
    payload = mock_post.call_args.kwargs["payload"]
    assert payload["instances"] == instances
    assert "version" not in payload


@patch("proxyml.client.post")
def test_explain_local_batch_with_version(mock_post):
    mock_post.return_value = _mock_response(200, _EXPLAIN_LOCAL_BATCH_RESPONSE)
    uid = "550e8400-e29b-41d4-a716-446655440000"
    explain_local_batch(instances=[[1.0, 2.0]], version=uid)
    payload = mock_post.call_args.kwargs["payload"]
    assert payload["version"] == uid


@patch("proxyml.client.post")
def test_explain_local_batch_failure_returns_none(mock_post):
    mock_post.return_value = _mock_response(422, {"detail": "bad input"})
    assert explain_local_batch(instances=[[1.0, 2.0]]) is None


# ---------------------------------------------------------------------------
# update_model
# ---------------------------------------------------------------------------

@patch("proxyml.client.patch")
def test_update_model_name(mock_patch):
    meta = {"version": "abc-123", "task": "regression", "name": "new name",
            "comments": None, "feature_names": None, "metrics": None,
            "trained_at": "2026-05-07T00:00:00", "mlflow_run_id": None}
    mock_patch.return_value = _mock_response(200, meta)
    result = update_model("abc-123", name="new name")
    assert result == meta
    payload = mock_patch.call_args.kwargs["payload"]
    assert payload == {"name": "new name"}


@patch("proxyml.client.patch")
def test_update_model_comments(mock_patch):
    mock_patch.return_value = _mock_response(200, {})
    update_model("abc-123", comments="some notes")
    payload = mock_patch.call_args.kwargs["payload"]
    assert payload == {"comments": "some notes"}


@patch("proxyml.client.patch")
def test_update_model_both_fields(mock_patch):
    mock_patch.return_value = _mock_response(200, {})
    update_model("abc-123", name="prod", comments="v2 data")
    payload = mock_patch.call_args.kwargs["payload"]
    assert payload == {"name": "prod", "comments": "v2 data"}


@patch("proxyml.client.patch")
def test_update_model_clear_field(mock_patch):
    mock_patch.return_value = _mock_response(200, {})
    update_model("abc-123", comments=None)
    payload = mock_patch.call_args.kwargs["payload"]
    assert payload == {"comments": None}


def test_update_model_no_fields_raises():
    with pytest.raises(ValueError):
        update_model("abc-123")


@patch("proxyml.client.patch")
def test_update_model_failure_returns_none(mock_patch):
    mock_patch.return_value = _mock_response(404, {"detail": "not found"})
    assert update_model("abc-123", name="x") is None
