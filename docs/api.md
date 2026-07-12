# API Reference

All functions are importable from the top-level `proxyml` package.

```python
import proxyml
# or
from proxyml import get_schema, put_schema, synthesize_data, ...
```

`proxyml.local` (local, offline model training — see [below](#local-training-proxymllocal)) requires an extra: `pip install 'proxyml[local]'`.

---

## Schema

### `get_schema(df, immutable_cols=None)`

Infer a ProxyML schema from a pandas DataFrame.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `df` | `pd.DataFrame` | Source data. Column dtypes determine feature types. |
| `immutable_cols` | `list[str] \| None` | Columns excluded from counterfactual search. |

**Returns** [`proxyml_core.schema.FeatureSchema`](https://github.com/proxyml/proxyml-core) — a typed schema. Its `.features` list holds `Feature` subclasses (`ContinuousFeature`, `CategoricalFeature`, `CategoricalOrdinalFeature`, `NumericOrdinalFeature`, `CountFeature`), suitable for passing to `put_schema`.

**Notes**
- Float columns → `ContinuousFeature` (mean, std, min, max)
- Bool columns → `CategoricalFeature` (true/false frequencies)
- Integer columns → `CountFeature` (Poisson lambda, max). Construct a `CategoricalOrdinalFeature`/`NumericOrdinalFeature` manually for ordered categories.
- Object/string columns → `CategoricalFeature` (category frequencies)

Review and adjust the returned schema before uploading if needed — e.g. `schema.features[0].immutable = True`.

---

### `put_schema(schema, name)`

Upload a schema to the ProxyML API under a name. Schemas are named resources — `synthesize_data` and `train_surrogate` reference them by `schema_name`.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `schema` | `FeatureSchema` | Typically produced by `get_schema`. |
| `name` | `str` | Name to store the schema under. Overwrites any existing schema with the same name. |

**Returns** `FeatureSchema | None` — the stored schema on success, `None` on failure.

---

### `fetch_schema(name)`

Retrieve a previously uploaded schema by name.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `name` | `str` | Name the schema was uploaded under. |

**Returns** `FeatureSchema | None` — the schema, or `None` if not found / on failure.

---

### `list_schemas()`

List all named schemas for the authenticated user.

**Returns** `list[dict] | None` — list of schema *metadata* dicts (`name`, `updated_at`, `feature_names`, `surrogate_versions` — not full feature definitions; use `fetch_schema` for that), or `None` on failure.

---

### `delete_schema(name)`

Delete a named schema.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `name` | `str` | Name of the schema to delete. |

**Returns** `bool` — `True` on success, `False` if the schema was not found or the request failed.

---

## Data Synthesis

### `synthesize_data(num_points=100, sample=None, as_df=True, *, schema_name)`

Generate synthetic data points using a previously uploaded schema.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `num_points` | `int` | Number of data points to generate. Default `100`. |
| `sample` | `list \| None` | If provided, generates neighbors blended around this instance. If `None`, samples from the global distribution. |
| `as_df` | `bool` | If `True` (default), returns a typed `pd.DataFrame`. If `False`, returns the raw API response dict. |
| `schema_name` | `str` | Name of the schema to use (as passed to `put_schema`). Keyword-only, required. |

**Returns** `pd.DataFrame | dict | None`

---

## Surrogate Model

### `train_surrogate(samples, predictions, feature_names, task="auto", test_size=0.2, *, schema_name, name=None, comments=None)`

Train a surrogate model on scored synthetic data.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `samples` | `list` | List of input feature vectors (list of lists). |
| `predictions` | `list` | Corresponding predictions from your black-box model. |
| `feature_names` | `list[str] \| None` | Column names matching the order of features in `samples`. Pass `None` to use all schema features in schema order. |
| `task` | `str` | `"auto"`, `"classification"`, or `"regression"`. `"auto"` infers from `predictions`. |
| `test_size` | `float` | Fraction of data held out for evaluation. Default `0.2`. |
| `schema_name` | `str` | Name of the schema to use (as passed to `put_schema`). Keyword-only, required. |
| `name` | `str \| None` | Optional human-readable label for this version. |
| `comments` | `str \| None` | Optional free-text notes stored with the model. |

**Returns** `dict | None` — API response including `version` (UUID), `trained_at`, `task`, `metrics`, and any training `warning`, or `None` on failure.

---

### `train_auto_surrogate(data, target_col, feature_names=None, task="auto", test_size=0.2, *, schema_name, immutable_cols=None, name=None, comments=None)`

Convenience wrapper for the common case where you already have samples and a target/prediction column in one place. Loads `data`, infers and uploads a schema, and trains a surrogate against `target_col` — all in one call. Skips `synthesize_data()` entirely, since `data` already contains the samples you want to train on (if you need the server to generate synthetic samples for you to score first, use `synthesize_data()` + `train_surrogate()` directly — see [`examples/basic_usage.py`](../examples/basic_usage.py)).

**Parameters**

| Name | Type | Description |
|---|---|---|
| `data` | `str \| Path \| pd.DataFrame` | A CSV path, or an already-loaded DataFrame, containing both the feature columns and `target_col`. |
| `target_col` | `str` | Name of the column to train against — either real ground-truth labels or a black box's predictions. |
| `feature_names` | `list[str] \| None` | Subset of feature columns to train on; omit for all. The full sample width is always uploaded — this only tells the surrogate which columns to fit on. |
| `task` | `str` | `"auto"`, `"classification"`, or `"regression"`. |
| `test_size` | `float` | Fraction of data held out for evaluation. Default `0.2`. |
| `schema_name` | `str` | Name to store the inferred schema under. Keyword-only, required. |
| `immutable_cols` | `list[str] \| None` | Passed through to `get_schema`. |
| `name` | `str \| None` | Optional human-readable label for this version. |
| `comments` | `str \| None` | Optional free-text notes stored with the model. |

**Returns** `dict | None` — same shape as `train_surrogate`, or `None` if schema upload or training failed.

```python
result = proxyml.train_auto_surrogate("data.csv", "approved", schema_name="my_schema")
version = result["version"]
```

---

### `export_surrogate(version)`

Export a trained surrogate — coefficients, intercept, scalers, and per-feature metadata needed to reconstruct its predictions without the SDK or API.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `version` | `str` | Version UUID of the surrogate to export. |

**Returns** [`proxyml_core.export.SurrogateExport`](https://github.com/proxyml/proxyml-core)` | None` — a typed, versioned export (stamped with `export_schema_version`), or `None` on failure.

Score it locally, with zero sklearn, via `proxyml_core.export.predict_from_export(export, sample)` — it handles every feature type the export can carry (continuous, count, categorical, both ordinal types) and multiclass, unlike a hand-rolled reconstruction. See [`examples/surrogate_export_example.py`](../examples/surrogate_export_example.py) for a worked example.

```python
export = proxyml.export_surrogate(version="<uuid>")

from proxyml_core.export import predict_from_export
prediction = predict_from_export(export, sample={"age": 42, "income": 55000})
```

---

### `predict(sample, version=None)`

Score a single instance using a trained surrogate model.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `sample` | `list` | A single input feature vector. |
| `version` | `str \| None` | Surrogate version UUID. `None` uses the latest version. |

**Returns** `dict | None` — API response with `prediction` and `model_version`, or `None` on failure.

---

### `predict_batch(samples, version=None)`

Score multiple instances in a single API call.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `samples` | `list[list]` | List of input feature vectors (one per row). |
| `version` | `str \| None` | Surrogate version UUID. `None` uses the latest version. |

**Returns** `dict | None` — API response with a `predictions` list (one value per input row) and `model_version`, or `None` on failure.

```python
result = predict_batch(samples=[[1.2, 0.5], [3.1, 0.8]])
# {"predictions": [0.74, 0.31], "model_version": "surrogate-<uuid>-regression"}
```

---

### `list_models(limit=50, offset=0)`

Return a page of trained surrogate models, newest first.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `limit` | `int` | Maximum models to return. Range 1–200, default `50`. |
| `offset` | `int` | Number of models to skip (for pagination). Default `0`. |

**Returns** `dict | None` — Dict with keys:
- `models`: list of metadata dicts, each with `version`, `task`, `name`, `comments`, `feature_names`, `metrics`, `trained_at`, and `mlflow_run_id`.
- `total`: total number of surrogates for this account (independent of `limit`/`offset`).

Returns `None` on failure.

```python
page = list_models(limit=10, offset=0)
print(f"{len(page['models'])} of {page['total']} models")
while len(page['models']) < page['total']:
    offset += 10
    next_page = list_models(limit=10, offset=offset)
    page['models'].extend(next_page['models'])
```

---

### `update_model(version, name=..., comments=...)`

Update the name and/or comments of a surrogate without retraining. Omit a parameter to leave that field unchanged; pass `None` to clear it.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `version` | `str` | UUID of the surrogate to update. |
| `name` | `str \| None` | New name, `None` to clear, or omit to leave unchanged. |
| `comments` | `str \| None` | New comments, `None` to clear, or omit to leave unchanged. |

**Returns** `dict | None` — Updated model metadata dict on success, `None` on failure. Raises `ValueError` if neither `name` nor `comments` is provided.

```python
update_model(version="<uuid>", name="prod-v3", comments="retrained on 2026-05 data")
update_model(version="<uuid>", comments=None)  # clears comments, leaves name unchanged
```

---

### `delete_model(model_id)`

Delete a surrogate model by its UUID.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `model_id` | `str` | UUID of the surrogate model to delete. |

**Returns** `bool` — `True` on success, `False` if the model was not found or the request failed.

---

### `get_model_schema(version)`

Retrieve the data schema a particular surrogate was trained against.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `version` | `str` | Version UUID of the surrogate model. |

**Returns** `FeatureSchema | None` — the schema, or `None` on failure.

---

## Local Training (`proxyml.local`)

Requires the `local` extra: `pip install 'proxyml[local]'` (adds scikit-learn and scipy). Everything below runs in-process — no API calls, no data leaves your machine.

### `train_challenger(df, target, schema, *, complexity=Complexity.MODERATE, feature_names=None, task="auto", test_size=0.2)`

Train a linear challenger model on `df` against `target`, locally. `target` can be real ground-truth labels (training a genuine challenger to compare against a champion model on real outcomes) or a black box's predictions (training a surrogate/explainer of that model) — the fit itself doesn't care which.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `df` | `pd.DataFrame` | Samples to train on, one column per schema feature. |
| `target` | `np.ndarray \| list` | The value to predict for each row of `df`. |
| `schema` | `FeatureSchema` | Describes `df`'s columns (e.g. from `get_schema`). |
| `complexity` | `Complexity` | Which rung of `LADDERS` to train at. Default `Complexity.MODERATE`. |
| `feature_names` | `list[str] \| None` | Subset of `schema.features` to train on; omit for all. |
| `task` | `str` | `"classification"`, `"regression"`, or `"auto"` (inferred from `target`). |
| `test_size` | `float` | Fraction of data held out to compute fidelity metrics. |

**Returns** `TrainedChallenger` — a dataclass with `pipeline` (the fitted scikit-learn `Pipeline`), `task`, `metrics`, `hyperparameters`, and `export` (a `SurrogateExport`, structurally identical to what `export_surrogate()` returns for a server-trained surrogate — score either one with the same `proxyml_core.export.predict_from_export`).

```python
from proxyml.local import train_challenger

result = train_challenger(df, df.pop("approved"), schema, task="classification")
print(result.metrics)  # {"f1": 0.91, "accuracy": 0.90}
```

---

### `train_auto_challenger(data, target_col, *, immutable_cols=None, complexity=Complexity.MODERATE, feature_names=None, task="auto", test_size=0.2)`

Convenience wrapper around `get_schema()` + `train_challenger()` — loads `data`, infers a schema, and trains in one call. This only automates schema inference and the feature/target column split; it does not search across `LADDERS` for the best-fitting rung — `complexity` still defaults to `Complexity.MODERATE` and remains overridable.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `data` | `str \| Path \| pd.DataFrame` | A CSV path, or an already-loaded DataFrame, containing both the feature columns and `target_col`. |
| `target_col` | `str` | Name of the column to train against. |
| `immutable_cols` | `list[str] \| None` | Passed through to `get_schema()`. |
| `complexity` | `Complexity` | Which rung of `LADDERS` to train at. |
| `feature_names` | `list[str] \| None` | Subset of feature columns to train on; omit for all. |
| `task` | `str` | `"classification"`, `"regression"`, or `"auto"`. |
| `test_size` | `float` | Fraction of data held out to compute fidelity metrics. |

**Returns** `TrainedChallenger` — same as `train_challenger`.

```python
from proxyml.local import train_auto_challenger

result = train_auto_challenger("data.csv", "approved", task="classification")
```

---

### `Complexity`, `Rung`, `LADDERS`

`Complexity` is a 3-member enum (`SIMPLE`, `MODERATE`, `FLEXIBLE`) selecting how strongly-regularized the trained model is. `LADDERS` maps each `Complexity` to a `Rung` (a small dataclass describing how the classifier/regressor is built at that rung).

`Complexity.MODERATE` matches the server's own default surrogate exactly — it's the baseline rung, not a separate model family. `SIMPLE` biases toward fewer effectively-nonzero coefficients (stronger regularization); `FLEXIBLE` searches a wider regularization grid for a closer per-sample fit. All three rungs stay within `LogisticRegressionCV`/`RidgeCV` — there are no tree/ensemble models to choose between, so results stay explainable by the same closed-form coefficient math the server uses.

```python
from proxyml.local import Complexity, LADDERS

print(LADDERS[Complexity.SIMPLE].description)
```

---

## Counterfactual Explanation

### `find_counterfactual(sample, target, n_neighbors=10000, perturbation_scale=0.1, version=None, as_df=True)`

Search for a minimal-change counterfactual that achieves a target prediction.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `sample` | `list` | The input instance to explain. |
| `target` | | The desired prediction label or value. For regression, pass a `float` (point target) or `[min, max]` range. |
| `n_neighbors` | `int` | Number of perturbations to evaluate during search. Higher values increase coverage. Default `10000`. |
| `perturbation_scale` | `float` | Controls how far perturbations stray from the original instance. Default `0.1`. |
| `version` | `str \| None` | Surrogate version UUID. `None` uses the latest version. |
| `as_df` | `bool` | If `True` (default), returns a `pd.DataFrame`. If `False`, returns the raw API response dict. |

**Returns** `pd.DataFrame | dict | None`
- `pd.DataFrame` (or `dict`) with the counterfactual instance on success.
- `None` if no counterfactual was found (logs a warning) or on API error.

Immutable columns (set in the schema) are excluded from the counterfactual search automatically.

---

### `find_counterfactuals(samples, target, n_neighbors=10000, perturbation_scale=0.1, version=None, as_dfs=True)`

Search for counterfactuals for multiple instances in a single API call.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `samples` | `list[list]` | List of input feature vectors (one per row). |
| `target` | | The desired prediction label or value, applied to all instances. For regression, pass a `float` or `[min, max]` range. |
| `n_neighbors` | `int` | Number of perturbations per instance. Default `10000`. |
| `perturbation_scale` | `float` | Controls perturbation range. Default `0.1`. |
| `version` | `str \| None` | Surrogate version UUID. `None` uses the latest version. |
| `as_dfs` | `bool` | If `True` (default), returns a list of results. If `False`, returns the raw API response dict. |

**Returns** `list[pd.DataFrame | None] | dict | None`
- With `as_dfs=True`: a list with one entry per input instance — a typed `pd.DataFrame` if a counterfactual was found, or `None` if not (a warning is logged for each missing result).
- With `as_dfs=False`: the raw API response dict.
- `None` on API error.

```python
results = find_counterfactuals(samples=[[1.2, 0.5], [3.1, 0.8]], target=1)
for i, cf in enumerate(results):
    if cf is not None:
        print(f"Instance {i}: {cf.iloc[0].to_dict()}")
    else:
        print(f"Instance {i}: no counterfactual found")
```

---

### `interpret_counterfactual(sample, counterfactual, prediction_changed, exclude_from_diff=None)`

Generate a human-readable explanation of the differences between a sample and its counterfactual.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `sample` | `dict` | Original instance as a feature-name → value dict. |
| `counterfactual` | `dict` | Counterfactual instance as a feature-name → value dict. |
| `prediction_changed` | `bool` | Whether the original model's prediction changed for the counterfactual. |
| `exclude_from_diff` | `list[str] \| None` | Feature names to ignore when computing the diff (e.g. IDs, timestamps). |

**Returns** `str` — Plain-text explanation.

If `prediction_changed` is `False`, the explanation notes that the surrogate may not fully capture the original model's decision boundary in this region.

---

## Local Explanations

### `explain_local(instance, version=None)`

Per-feature contribution breakdown for a single instance.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `instance` | `list` | A single feature vector in schema order. |
| `version` | `str \| None` | Surrogate version UUID. `None` uses the latest version. |

**Returns** `dict | None` — Dict with keys:
- `prediction` — surrogate's output for this instance.
- `feature_contributions` — list of `{feature, contribution, abs_contribution}` dicts, sorted by `abs_contribution` descending.
- `intercept` — model intercept (`sum(contributions) + intercept ≈ raw decision function`).
- `probabilities` — class probabilities for classification; `None` for regression.
- `per_class_contributions` — per-class breakdown for multiclass; `None` otherwise.

---

### `explain_local_batch(instances, version=None)`

Per-feature contribution breakdown for multiple instances in one API call.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `instances` | `list[list]` | List of feature vectors, each in schema order. |
| `version` | `str \| None` | Surrogate version UUID. `None` uses the latest version. |

**Returns** `dict | None` — Dict with keys:
- `results` — list of per-instance dicts, each with the same structure as `explain_local` (minus `model_version` and `task`).
- `model_version`, `task`, `schema_warning`.

Returns `None` on failure.

```python
results = explain_local_batch(instances=df.values.tolist())
for item in results["results"]:
    top = item["feature_contributions"][0]
    print(f"prediction={item['prediction']}  top feature={top['feature']} ({top['contribution']:+.3f})")
```

---

## Feature Importances & Model Insights

### `get_feature_importances(version=None)`

Return feature importances (linear coefficients) for a trained surrogate.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `version` | `str \| None` | Surrogate version UUID. `None` uses the latest version. |

**Returns** `dict | None` — Dict with keys:
- `feature_importances`: list of `{feature, coefficient, abs_coefficient}` dicts, sorted by `abs_coefficient` descending.
- `per_class_importances`: for multiclass classification, a list of `{class_label, importances}` dicts; `None` otherwise.
- `model_version`, `task`, `note`.

Coefficients are in the scaled feature space — magnitudes are comparable across features but are not in original units.

---

### `get_model_summary(version=None)`

Return a combined report of feature importances, fidelity metrics, and model metadata for a single surrogate.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `version` | `str \| None` | Surrogate version UUID. `None` uses the latest version. |

**Returns** `dict | None` — Dict with keys:
- `model_version` — UUID of the surrogate.
- `task` — `"classification"` or `"regression"`.
- `trained_at` — ISO timestamp.
- `name`, `comments` — User-supplied labels (may be `None`).
- `feature_names` — Features used by this model.
- `metrics` — Fidelity metrics dict (e.g. `{"r2": 0.92}` or `{"f1": 0.87, "accuracy": 0.88}`).
- `mlflow_run_id` — MLflow run ID for this training run, for correlation with the MLflow UI (may be `None` for models trained before this field was added).
- `feature_importances` — Same structure as `get_feature_importances`.
- `per_class_importances` — Per-class breakdown for multiclass; `None` otherwise.
- `note` — Explanation of coefficient scaling.

```python
summary = get_model_summary()
print(summary["metrics"])          # {"r2": 0.92}
print(summary["feature_importances"][0])  # highest-impact feature
```

---

### `diff_models(version_a, version_b)`

Compare two surrogate models — coefficient shifts and metric deltas.

Both surrogates must have the same task type (both classification or both regression) and share at least one feature; otherwise the API returns an error.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `version_a` | `str` | UUID of the first (baseline) surrogate. |
| `version_b` | `str` | UUID of the second (comparison) surrogate. |

**Returns** `dict | None` — Dict with keys:
- `version_a`, `version_b` — The UUIDs passed in.
- `task` — Shared task type.
- `metric_diff` — Dict mapping metric name → `{a, b, delta}` (e.g. `{"r2": {"a": 0.87, "b": 0.92, "delta": 0.05}}`).
- `coefficient_diff` — List of `{feature, a, b, delta}` dicts for shared features, sorted by `abs(delta)` descending.
- `per_class_coefficient_diff` — For multiclass models, a list of `{class_label, coefficient_diff}` dicts showing per-class shifts; `None` for binary or regression.
- `features_added` — Features present in `version_b` but not `version_a`.
- `features_removed` — Features present in `version_a` but not `version_b`.

```python
diff = diff_models(version_a="aaa-...", version_b="bbb-...")
for entry in diff["coefficient_diff"]:
    print(f"{entry['feature']}: {entry['a']:.3f} → {entry['b']:.3f} (Δ {entry['delta']:+.3f})")
```

---

## Account

### `health_check()`

Check API connectivity and version. Does not require an API key and does not count against usage quota.

**Returns** `dict | None` — Dict with `status`, `model_loaded`, and `version`, or `None` on failure (including network errors).

---

### `get_usage()`

Return the current tier, usage counts, and quota for the authenticated user.

**Returns** `dict | None` — Usage/quota dict, or `None` on failure.

---

### `rotate_key()`

Rotate the API key, revoking all previously issued keys.

**Returns** `str | None` — The new API key, or `None` on failure.

```python
new_key = rotate_key()
# Update PROXYML_API_KEY in your environment with new_key before making further calls
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PROXYML_API_KEY` | *(required)* | API key. Must be set before using the SDK. |
| `PROXYML_BASE_URL` | `https://api.proxyml.ai/api/v1` | Base URL for the API. |
