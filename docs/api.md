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

### `train_surrogate(samples, predictions, feature_names, task="auto", test_size=0.2)`

Train a surrogate model on scored synthetic data.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `samples` | `list` | List of input feature vectors (list of lists). |
| `predictions` | `list` | Corresponding predictions from your black-box model. |
| `feature_names` | `list[str] \| None` | Column names matching the order of features in `samples`. |
| `task` | `str` | `"auto"`, `"classification"`, or `"regression"`. `"auto"` infers from `predictions`. |
| `test_size` | `float` | Fraction of data held out for evaluation. Default `0.2`. |

**Returns** `dict | None` — API response including surrogate version and evaluation metrics, or `None` on failure.

---

### `predict(samples, version=None)`

Score samples using a trained surrogate model.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `samples` | `list` | List of input feature vectors. |
| `version` | `int \| None` | Surrogate version to use. `None` (or `0`) uses the latest version. Versions start at `1`. |

**Returns** `dict | None` — API response with predictions, or `None` on failure.

---

## Counterfactual Explanation

### `find_counterfactual(sample, target, n_neighbors=10000, perturbation_scale=0.1, version=None, as_df=True)`

Search for a minimal-change counterfactual that achieves a target prediction.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `sample` | `list` | The input instance to explain. |
| `target` | | The desired prediction label or value. |
| `n_neighbors` | `int` | Number of perturbations to evaluate during search. Higher values increase coverage. Default `10000`. |
| `perturbation_scale` | `float` | Controls how far perturbations stray from the original instance. Default `0.1`. |
| `version` | `int \| None` | Surrogate version. `None` uses latest. |
| `as_df` | `bool` | If `True` (default), returns a `pd.DataFrame`. If `False`, returns raw API response. |

**Returns** `pd.DataFrame | dict | None`
- `pd.DataFrame` (or `dict`) with the counterfactual instance on success.
- `None` if no counterfactual was found (prints a warning message) or on API error.

**Notes**

Immutable columns (set in the schema) are excluded from the counterfactual search automatically.

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

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PROXYML_API_KEY` | *(required)* | API key. Must be set before importing `proxyml`. |
| `PROXYML_BASE_URL` | `https://api.proxyml.ai/api/v1` | Base URL for the API. |
