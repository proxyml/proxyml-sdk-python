"""Array features (v1): summarized in the schema, sent as null, restored locally.
Array values must never appear in any request payload."""
import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from proxyml.client import (
    explain_local,
    explain_local_batch,
    find_counterfactual,
    find_counterfactuals,
    predict,
    predict_batch,
    synthesize_data,
    train_auto_surrogate,
    train_surrogate,
)
from proxyml.schema_builder import MIN_ARRAY_ROWS, get_schema, get_schema_like
from proxyml_core.drift import compare_schemas
from proxyml_core.schema import ArrayFeature, ContinuousFeature, FeatureSchema

rng = np.random.default_rng(0)
N = 200
VEC = rng.normal(size=(N, 4))


def _df(vecs=VEC, kind="list"):
    cells = [v.tolist() if kind == "list" else tuple(v) if kind == "tuple" else v for v in vecs]
    col = pd.Series(cells, dtype=object)
    return pd.DataFrame({"age": rng.normal(40, 5, len(vecs)), "emb": col})


def _resp(body, status=200):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = body
    r.text = str(body)
    return r


# ── schema inference ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", ["list", "tuple", "ndarray"])
def test_get_schema_summarizes_vector_column(kind):
    schema = get_schema(_df(kind=kind))
    age, emb = schema.features
    assert isinstance(age, ContinuousFeature)
    assert isinstance(emb, ArrayFeature) and emb.immutable and emb.width == 4
    np.testing.assert_allclose(emb.mean, VEC.mean(axis=0))
    expected = np.sqrt(((VEC - VEC.mean(axis=0)) ** 2).sum(axis=1).mean())
    assert emb.rms_radius == pytest.approx(expected)
    # the summary carries no rows
    d = emb.to_dict()
    assert set(d) == {"type", "name", "immutable", "width", "mean", "rms_radius"}


def test_array_column_is_immutable_even_if_not_listed():
    assert get_schema(_df(), immutable_cols=[]).features[1].immutable is True


def test_too_few_rows_refused():
    with pytest.raises(ValueError, match=str(MIN_ARRAY_ROWS)):
        get_schema(_df(VEC[: MIN_ARRAY_ROWS - 1]))
    get_schema(_df(VEC[:MIN_ARRAY_ROWS]))  # exactly the floor is fine


@pytest.mark.parametrize("bad", ["ragged", "nan", "missing", "text", "width1"])
def test_bad_array_columns_raise_clear_errors(bad):
    cells = [v.tolist() for v in VEC]
    if bad == "ragged":
        cells[3] = [1.0, 2.0]
    elif bad == "nan":
        cells[3][1] = float("nan")
    elif bad == "missing":
        cells[3] = None
    elif bad == "text":
        cells[3] = ["a", "b", "c", "d"]
    elif bad == "width1":
        cells = [[v[0]] for v in VEC]
    df = pd.DataFrame({"emb": pd.Series(cells, dtype=object)})
    with pytest.raises(ValueError, match="emb"):
        get_schema(df)


def test_get_schema_like_follows_array_baseline_and_flags_changes():
    base = get_schema(_df())
    cur = get_schema_like(_df(rng.normal(size=(N, 4))), base)
    assert isinstance(cur.features[1], ArrayFeature)
    assert compare_schemas(base, cur, N).passed
    wider = get_schema_like(_df(rng.normal(size=(N, 6))), base)
    assert compare_schemas(base, wider, N).features[1].status == "type_changed"
    no_longer_vectors = get_schema_like(pd.DataFrame({"age": [1.0] * N, "emb": [1.5] * N}), base)
    assert compare_schemas(base, no_longer_vectors, N).features[1].status == "type_changed"


def test_drift_end_to_end_on_summaries():
    base = get_schema(_df(rng.normal(size=(1000, 8))))
    same = get_schema(_df(rng.normal(size=(500, 8))))
    shifted = get_schema(_df(rng.normal(size=(500, 8)) + 1.0))
    assert compare_schemas(base, same, 500, baseline_n_samples=1000).passed
    r = compare_schemas(base, shifted, 500, baseline_n_samples=1000)
    assert not r.passed and r.features[1].status == "drift"


# ── nothing array-valued is ever sent ────────────────────────────────────────

SAMPLE = [40.0, [0.11, 0.22, 0.33], 5]
CLEAN = [40.0, None, 5]


def _assert_no_vectors(payload):
    import json
    text = json.dumps(payload, default=list)
    assert "0.11" not in text and "0.22" not in text and "0.33" not in text


@patch("proxyml.client.post")
def test_train_surrogate_nulls_arrays(mock_post):
    mock_post.return_value = _resp({"ok": 1})
    train_surrogate(samples=[SAMPLE, [41.0, np.array([0.11, 0.22, 0.33]), 6]], predictions=[1, 2], feature_names=None, schema_name="s")
    payload = mock_post.call_args.kwargs["payload"]
    assert payload["samples"] == [CLEAN, [41.0, None, 6]]
    _assert_no_vectors(payload)


@patch("proxyml.client.post")
def test_train_auto_surrogate_uploads_summary_only(mock_post):
    df = _df()
    df["y"] = rng.normal(size=len(df))
    with patch("proxyml.client.put_schema", return_value=object()) as mock_put:
        mock_post.return_value = _resp({"ok": 1})
        train_auto_surrogate(df, target_col="y", schema_name="s")
    uploaded = mock_put.call_args.args[0]
    assert isinstance(uploaded.features[1], ArrayFeature)
    payload = mock_post.call_args.kwargs["payload"]
    assert all(row[1] is None for row in payload["samples"])
    assert len(payload["samples"]) == N


@patch("proxyml.client.post")
def test_predict_and_explain_null_arrays(mock_post):
    mock_post.return_value = _resp({"ok": 1})
    predict(SAMPLE)
    assert mock_post.call_args.kwargs["payload"]["inputs"] == CLEAN
    predict_batch([SAMPLE, SAMPLE])
    assert mock_post.call_args.kwargs["payload"]["inputs"] == [CLEAN, CLEAN]
    explain_local(SAMPLE)
    assert mock_post.call_args.kwargs["payload"]["instance"] == CLEAN
    explain_local_batch([SAMPLE, np.array([1, 2], dtype=object).tolist()])
    assert mock_post.call_args.kwargs["payload"]["instances"][0] == CLEAN


# ── results get the local array back ─────────────────────────────────────────

CF_BODY = {
    "counterfactual": [55.0, None, 7], "feature_names": ["age", "emb", "visits"],
    "feature_types": ["continuous", "array", "count"], "outlier_score": 0.0,
    "task": "classification", "target_label": 1, "warning": None,
}


@patch("proxyml.client.post")
def test_find_counterfactual_restores_array(mock_post):
    mock_post.return_value = _resp(dict(CF_BODY))
    df = find_counterfactual(SAMPLE, target=1)
    assert mock_post.call_args.kwargs["payload"]["instance"] == CLEAN
    assert df.loc[0, "age"] == 55.0 and df.loc[0, "visits"] == 7
    np.testing.assert_allclose(df.loc[0, "emb"], [0.11, 0.22, 0.33])
    raw = find_counterfactual(SAMPLE, target=1, as_df=False)
    assert raw["counterfactual"] == [55.0, [0.11, 0.22, 0.33], 7]


@patch("proxyml.client.post")
def test_find_counterfactuals_restores_per_item(mock_post):
    other = [60.0, [9.0, 8.0, 7.0], 1]
    body = {
        "results": [{"counterfactual": [55.0, None, 7], "warning": None}, {"counterfactual": None, "warning": "none"}],
        "feature_names": CF_BODY["feature_names"], "feature_types": CF_BODY["feature_types"],
    }
    mock_post.return_value = _resp(body)
    dfs = find_counterfactuals([SAMPLE, other], target=1)
    assert mock_post.call_args.kwargs["payload"]["instances"] == [CLEAN, [60.0, None, 1]]
    np.testing.assert_allclose(dfs[0].loc[0, "emb"], [0.11, 0.22, 0.33])
    assert dfs[1] is None


SYN = {
    "samples": [[30.0, None, 1], [35.0, None, 2], [38.0, None, 3]],
    "feature_names": ["age", "emb", "visits"], "feature_types": ["continuous", "array", "count"],
}


def _syn(**kw):
    import copy
    return _resp(copy.deepcopy(SYN))


@patch("proxyml.client.post")
def test_synthesize_with_sample_carries_the_samples_array(mock_post):
    mock_post.return_value = _syn()
    df = synthesize_data(3, sample=SAMPLE, schema_name="s")
    assert mock_post.call_args.kwargs["payload"]["instance"] == CLEAN
    for v in df["emb"]:
        np.testing.assert_allclose(v, [0.11, 0.22, 0.33])
    raw = synthesize_data(3, sample=SAMPLE, as_df=False, schema_name="s")
    assert all(row[1] == [0.11, 0.22, 0.33] for row in raw["samples"])


@patch("proxyml.client.post")
def test_synthesize_global_fills_arrays_from_local_source_only(mock_post):
    mock_post.return_value = _syn()
    src = _df(VEC[:60])
    df = synthesize_data(3, schema_name="s", array_source=src)
    pool = {tuple(np.round(v, 9)) for v in VEC[:60]}
    assert all(tuple(np.round(v, 9)) in pool for v in df["emb"])
    # the request never carried the source
    assert "emb" not in str(mock_post.call_args.kwargs["payload"])


@patch("proxyml.client.post")
def test_synthesize_global_without_source_leaves_none_and_warns(mock_post, caplog):
    mock_post.return_value = _syn()
    with caplog.at_level(logging.WARNING, logger="proxyml.client"):
        df = synthesize_data(3, schema_name="s")
    assert df["emb"].isna().all()
    assert "array_source" in caplog.text
