# API Reference

All functions are importable from the top-level `proxyml` package.

```python
import proxyml
# or
from proxyml import get_schema, put_schema, synthesize_data, ...
```

---

## Schema

### `get_schema(df, immutable_cols=None)`

Infer a ProxyML schema from a pandas DataFrame.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `df` | `pd.DataFrame` | Source data. Column dtypes determine feature types. |
| `immutable_cols` | `list[str] \| None` | Columns excluded from counterfactual search. |

**Returns** `dict` — Schema dict suitable for passing to `put_schema`.

**Notes**
- Float columns → `continuous` (mean, std, min, max)
- Bool columns → `categorical` (true/false frequencies)
- Integer columns → `count` (Poisson lambda, max). Change to `categorical_ordinal` manually for ordered categories.
- Object/string columns → `categorical` (category frequencies)

The returned dict includes a `_note` key with a reminder to review integer columns. Remove it before uploading if desired.

---

### `put_schema(schema)`

Upload a schema to the ProxyML API.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `schema` | `dict` | Schema dict, typically produced by `get_schema`. |

**Returns** `dict | None` — API response on success, `None` on failure.

---

## Data Synthesis

### `synthesize_data(num_points=100, sample=None, as_df=True)`

Generate synthetic data points using the uploaded schema.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `num_points` | `int` | Number of data points to generate. |
| `sample` | `list \| None` | If provided, generates neighbors blended around this instance. If `None`, samples from the global distribution. |
| `as_df` | `bool` | If `True` (default), returns a typed `pd.DataFrame`. If `False`, returns the raw API response dict. |

**Returns** `pd.DataFrame | dict | None`

---

## Surrogate Model

### `train_surrogate(samples, predictions, feature_names, task="auto", test_size=0.2, name=None, comments=None)`

Train a surrogate model on scored synthetic data.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `samples` | `list` | List of input feature vectors (list of lists). |
| `predictions` | `list` | Corresponding predictions from your black-box model. |
| `feature_names` | `list[str] \| None` | Column names matching the order of features in `samples`. Pass `None` to use all schema features in schema order. |
| `task` | `str` | `"auto"`, `"classification"`, or `"regression"`. `"auto"` infers from `predictions`. |
| `test_size` | `float` | Fraction of data held out for evaluation. Default `0.2`. |
| `name` | `str \| None` | Optional human-readable label for this version. |
| `comments` | `str \| None` | Optional free-text notes stored with the model. |

**Returns** `dict | None` — API response including `version` (UUID), `trained_at`, `task`, `metrics`, and any training `warning`, or `None` on failure.

---

### `predict(samples, version=None)`

Score a single instance using a trained surrogate model.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `samples` | `list` | A single input feature vector. |
| `version` | `str \| None` | Surrogate version UUID. `None` uses the latest version. |

**Returns** `dict | None` — API response with `prediction` and `model_version`, or `None` on failure.

---

### `predict_batch(instances, version=None)`

Score multiple instances in a single API call.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `instances` | `list[list]` | List of input feature vectors (one per row). |
| `version` | `str \| None` | Surrogate version UUID. `None` uses the latest version. |

**Returns** `dict | None` — API response with a `predictions` list (one value per input row) and `model_version`, or `None` on failure.

```python
result = predict_batch(instances=[[1.2, 0.5], [3.1, 0.8]])
# {"predictions": [0.74, 0.31], "model_version": "surrogate-<uuid>-regression"}
```

---

### `list_models()`

Return metadata for all trained surrogate models, newest first.

**Returns** `list[dict] | None` — List of model metadata dicts, each with keys `version`, `task`, `name`, `comments`, `feature_names`, `metrics`, and `trained_at`. Returns `None` on failure.

---

### `delete_model(model_id)`

Delete a surrogate model by its UUID.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `model_id` | `str` | UUID of the surrogate model to delete. |

**Returns** `bool` — `True` on success, `False` if the model was not found or the request failed.

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
- `None` if no counterfactual was found (prints a warning message) or on API error.

Immutable columns (set in the schema) are excluded from the counterfactual search automatically.

---

### `find_counterfactuals(instances, target, n_neighbors=10000, perturbation_scale=0.1, version=None, as_df=True)`

Search for counterfactuals for multiple instances in a single API call.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `instances` | `list[list]` | List of input feature vectors (one per row). |
| `target` | | The desired prediction label or value, applied to all instances. For regression, pass a `float` or `[min, max]` range. |
| `n_neighbors` | `int` | Number of perturbations per instance. Default `10000`. |
| `perturbation_scale` | `float` | Controls perturbation range. Default `0.1`. |
| `version` | `str \| None` | Surrogate version UUID. `None` uses the latest version. |
| `as_df` | `bool` | If `True` (default), returns a list of results. If `False`, returns the raw API response dict. |

**Returns** `list[pd.DataFrame | None] | dict | None`
- With `as_df=True`: a list with one entry per input instance — a typed `pd.DataFrame` if a counterfactual was found, or `None` if not (a warning is printed for each missing result).
- With `as_df=False`: the raw API response dict.
- `None` on API error.

```python
results = find_counterfactuals(instances=[[1.2, 0.5], [3.1, 0.8]], target=1)
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
- `features_added` — Features present in `version_b` but not `version_a`.
- `features_removed` — Features present in `version_a` but not `version_b`.

```python
diff = diff_models(version_a="aaa-...", version_b="bbb-...")
for entry in diff["coefficient_diff"]:
    print(f"{entry['feature']}: {entry['a']:.3f} → {entry['b']:.3f} (Δ {entry['delta']:+.3f})")
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PROXYML_API_KEY` | *(required)* | API key. Must be set before using the SDK. |
| `PROXYML_BASE_URL` | `https://api.proxyml.ai/api/v1` | Base URL for the API. |
