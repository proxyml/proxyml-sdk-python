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

### `get_schema_like(df, baseline)`

Summarize a DataFrame using the column types of an existing schema (for example one from `get_model_schema()`), instead of inferring them from dtypes. `check_data_drift` does this for you; call it directly when you want to build the summary yourself.

- A column the baseline has is summarized as the baseline's type when its values fit it (an integer column becomes `NumericOrdinalFeature` or `CategoricalOrdinalFeature` if that is what the model uses). Ordinal categories keep the baseline's order, with any new category appended.
- Values that do not fit (text where the baseline has a continuous feature, a negative "count", non-integer "ordinal" values) are summarized as the type they look like, so a real type change is still reported.
- Columns the baseline does not have are summarized as `get_schema()` would. Columns that are entirely empty are left out.

---

### `put_schema(schema, name, n_samples=None, source=None)`

Upload a schema to the ProxyML API under a name. Schemas are named resources — `synthesize_data` and `train_surrogate` reference them by `schema_name`.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `schema` | `FeatureSchema` | Typically produced by `get_schema`. |
| `name` | `str` | Name to store the schema under. Overwrites any existing schema with the same name. |
| `n_samples` | `int \| None` | Rows the schema's statistics were computed from. Recorded with models trained on this schema so data drift checks can allow for noisy statistics from a small training set (alposit 0.4.0+; older servers ignore it). `train_auto_surrogate` passes it automatically. |
| `source` | `str \| None` | `"data"` if the statistics were computed from real rows and not edited, `"hand_written"` if you wrote or tuned them by hand; leave it out if you are not sure. Recorded with models trained on this schema and reported as `baseline_source` on data drift checks. `train_auto_surrogate` passes `"data"` automatically. Models trained before this existed report `"unknown"` (alposit 0.6.0+; older servers ignore it). |

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

### `train_challenger(df, target, schema, *, complexity=Complexity.MODERATE, feature_names=None, task="auto", test_size=0.2, target_name="target", champion_predictions=None)`

Train a linear challenger model on `df` against `target`, locally. `target` can be real ground-truth labels (training a genuine challenger to compare against a champion model on real outcomes) or a black box's predictions (training a surrogate/explainer of that model) — the fit itself doesn't care which.

Rows where `target` is missing (NaN/None) are dropped before training, the CV split, and champion scoring — never silently included. The drop count and a human-readable scope-limitation note are recorded on the result.

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
| `target_name` | `str` | Human-readable name for `target`, used in `population_note`. Default `"target"`. |
| `champion_predictions` | `np.ndarray \| list \| None` | A champion model's predictions, one per row of `df`/`target` (same order, pre-drop). If given, scored via `score_champion()` against the same (row-dropped) `target`, and attached as `TrainedChallenger.champion_metrics`. Rows dropped for a missing `target` are dropped from this too, so champion and challenger are always evaluated on the same population. |

**Returns** `TrainedChallenger` — a dataclass with `pipeline` (the fitted scikit-learn `Pipeline`), `task`, `metrics`, `hyperparameters`, `export` (a `SurrogateExport`, structurally identical to what `export_surrogate()` returns for a server-trained surrogate — score either one with the same `proxyml_core.export.predict_from_export`), `n_samples_total`, `n_samples_dropped_unlabeled`, `population_note`, and `champion_metrics` (`None` unless `champion_predictions` was given).

```python
from proxyml.local import train_challenger

result = train_challenger(df, df.pop("approved"), schema, task="classification")
print(result.metrics)  # {"f1": 0.91, "accuracy": 0.90}
print(result.population_note)  # "Evaluated on all 300 row(s) — 'target' had no missing values."
```

---

### `train_auto_challenger(data, target_col, *, immutable_cols=None, complexity=Complexity.MODERATE, feature_names=None, task="auto", test_size=0.2, champion_predictions=None)`

Convenience wrapper around `get_schema()` + `train_challenger()` — loads `data`, infers a schema, and trains in one call. This only automates schema inference and the feature/target column split; it does not search across `LADDERS` for the best-fitting rung — `complexity` still defaults to `Complexity.MODERATE` and remains overridable.

Rows with a missing `target_col` value are dropped before training and champion scoring — see `train_challenger()` above. Schema inference (feature means/stds/categories) still runs over every row, including ones later dropped for a missing target — only training and evaluation are restricted to the labeled subset.

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
| `champion_predictions` | `np.ndarray \| list \| None` | A champion model's predictions, one per row of `data` (same order) — see `train_challenger()`. |

**Returns** `TrainedChallenger` — same as `train_challenger`.

```python
from proxyml.local import train_auto_challenger

result = train_auto_challenger("data.csv", "approved", task="classification")
```

---

### `score_champion(labels, predictions, *, task)`

Score a champion model's predictions against real labels, locally. Uses the exact same scoring code as `train_challenger()`'s internal fidelity metrics, so `champion_metrics` and a paired `TrainedChallenger.metrics` are computed identically — required for an apples-to-apples champion-vs-challenger comparison.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `labels` | `np.ndarray \| list` | Real ground-truth labels. |
| `predictions` | `np.ndarray \| list` | The champion model's predictions for those same labels. |
| `task` | `"classification" \| "regression"` | No `"auto"` here — pass the same task the paired challenger resolved to (e.g. `result.task`), since letting the two resolve independently risks them silently diverging. |

**Returns** `dict[str, float]` — `{"f1": ..., "accuracy": ...}` for classification or `{"r2": ...}` for regression, the same shape as `TrainedChallenger.metrics`.

Most callers won't need to call this directly — pass `champion_predictions` to `train_challenger()`/`train_auto_challenger()` instead, which calls this automatically against the correct (row-dropped) population. Call it directly only when scoring a champion independently of a specific training run.

```python
from proxyml.local import score_champion

champion_metrics = score_champion(result.pipeline.predict(X_test), champion_preds, task=result.task)
```

---

### `to_challenger_upload(result, *, n_samples=None, champion_metrics=None, sdk_version=None, proxyml_core_version=None)`

Assemble the JSON-serializable payload for `POST /app/projects/{id}/challenger`, ready for `json.dump` — upload it either by POSTing it directly, or by saving it to a file and using the dashboard's "Upload challenger" button.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `result` | `TrainedChallenger` | Output of `train_challenger()`/`train_auto_challenger()`. |
| `n_samples` | `int \| None` | Size of the evaluation set. Defaults to `result.n_samples_total - result.n_samples_dropped_unlabeled` (the labeled-row count); override only if you scored on some other population. |
| `champion_metrics` | `dict[str, float] \| None` | The champion's performance, from `score_champion()`. Defaults to `result.champion_metrics`. Pass `None` explicitly (with no `champion_predictions` used during training) to get a self-contained export of the challenger alone — e.g. to save/share before you have a champion to compare against. The upload endpoint itself still requires `champion_metrics` at upload time; this function just doesn't force you to have it up front. |
| `sdk_version` | `str \| None` | Defaults to the installed `proxyml` version. |
| `proxyml_core_version` | `str \| None` | Defaults to the installed `proxyml-core` version. |

**Returns** `dict[str, Any]` with keys `export`, `challenger_metrics`, `n_samples`, `n_samples_total`, `n_samples_dropped_unlabeled`, `population_note`, `complexity`, `sdk_version`, `proxyml_core_version`, and `champion_metrics` (when available) — the dropped-row count and scope limitation travel with the upload so a reviewer can see them.

```python
from proxyml.local import to_challenger_upload

payload = to_challenger_upload(result)  # n_samples and champion_metrics auto-derived
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

## Continuous Integration (`proxyml.ci`)

Post a challenger comparison as a PR comment from a CI job — e.g. a GitHub Actions run that trains a challenger against real held-out data on every pull request. Authenticated with a project-scoped `PROXYML_CI_TOKEN` (mint one from the project's page in the ProxyML dashboard) rather than `PROXYML_API_KEY`, since a CI token only ever authorizes writes to the one project it was minted for — the account-wide API key has no way to prove it owns a given dashboard project.

### `upload_challenger(payload, *, ci_token=None)`

Uploads a challenger comparison from a headless environment. Same trust model as the dashboard's manual upload — self-reported, unsigned metrics.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `payload` | `dict` | Output of `proxyml.local.to_challenger_upload()`. |
| `ci_token` | `str \| None` | Overrides `PROXYML_CI_TOKEN` if provided. |

**Returns** `dict | None` — `{"status": "ok", "challenger_version": ..., "task": ...}`, or `None` on failure.

---

### `get_challenger_summary(*, ci_token=None)`

Fetches a CI-ready summary of the latest challenger comparison for the project authorized by the given CI token: which model won, the primary metric and its delta, and a ready-to-post markdown snippet.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `ci_token` | `str \| None` | Overrides `PROXYML_CI_TOKEN` if provided. |

**Returns** `dict | None` with keys `winner` (`"challenger" \| "champion" \| "tie"`), `primary_metric`, `champion_value`, `challenger_value`, `delta`, `challenger_version`, `trained_at`, `data_fingerprint_match`, and `markdown_summary`. `None` if no challenger has been uploaded to this project yet, or on API error.

---

### `proxyml.ci.post_challenger_comment(*, ci_token=None, github_token=None, repo=None, pr_number=None)`

Posts or updates a sticky PR comment summarizing the latest challenger comparison. Fetches the summary via `get_challenger_summary()`, then finds-and-updates a prior comment (matched by a hidden HTML marker) if one exists on the PR, so repeated runs edit in place instead of piling up duplicates — the same way coverage bots behave.

Reads `GITHUB_TOKEN`, `GITHUB_REPOSITORY`, and `GITHUB_EVENT_PATH` from the environment by default — all three are set automatically inside a GitHub Actions job triggered by a `pull_request` event, so no arguments are needed in the common case.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `ci_token` | `str \| None` | Overrides `PROXYML_CI_TOKEN` if provided. |
| `github_token` | `str \| None` | Overrides `GITHUB_TOKEN` if provided. |
| `repo` | `str \| None` | `"owner/name"`. Overrides `GITHUB_REPOSITORY` if provided. |
| `pr_number` | `int \| None` | Overrides the PR number read from `GITHUB_EVENT_PATH` if provided — required on non-`pull_request` events. |

**Returns** the created/updated GitHub comment JSON, or `None` if the challenger summary couldn't be fetched or the GitHub API call failed.

```python
from proxyml import local, ci
from proxyml.client import upload_challenger

result = local.train_challenger(df, df.pop("approved"), schema, task="classification")
payload = local.to_challenger_upload(result, champion_metrics=my_champion_metrics)
upload_challenger(payload)
ci.post_challenger_comment()
```

See [`examples/ci_challenger_example.py`](../examples/ci_challenger_example.py) and [`examples/github_actions_challenger.yml`](../examples/github_actions_challenger.yml) for a full workflow.

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

## Data Drift

Compare recent data against the data a model was trained on. Only summary statistics are sent (means, standard deviations, category names with their frequencies, count rates), never rows. Each feature is compared on its own (marginal distribution); changes in the *relationships* between features are not detected. Each verdict is stored, and the last 100 per model are kept.

### `check_data_drift(model_id, df, *, schema=None, label=None, thresholds=None, baseline_n_samples=None, align_to_model=True)`

Summarizes `df` locally and returns a verdict against the model's stored training schema.

- `status` is `"pass"`, `"drift"` or `"insufficient_sample"` (fewer rows than `min_samples`, default 100, so no verdict is given). `passed` is `True` only for `"pass"`. Drift is a normal return, not an error.
- `features` lists per-feature findings (metric values, noise floors, and human-readable `reasons`). Added, removed or type-changed columns are reported as findings too.
- `thresholds` overrides the policy per call: `mean_shift` (default 0.2, in baseline standard deviations), `std_ratio` (1.4), `category_tvd` (0.08), `count_rel_change` (0.2), `new_category_mass` (0.02), `noise_sigmas` (auto) and `min_samples` (100). A feature is flagged only when a metric exceeds its threshold **and** a sampling-noise allowance that grows as the sample shrinks, so lowering a threshold below that allowance has no effect. Lowering `noise_sigmas` is what makes checks more sensitive, at the cost of more false alarms.
- `baseline_n_samples`: the number of rows the model's training schema was computed from. Defaults to the count recorded when the schema was uploaded (`train_auto_surrogate` and `put_schema(..., n_samples=)` record it; the result's `baseline_n_source` says whether it came from `"request"` or `"schema"`). If neither is available the baseline is treated as exact, which makes small training sets (a few hundred rows) false-alarm noticeably more often, so pass it for models whose schema was uploaded without a count.
- `df` is summarized using the model's own column types (`align_to_model=True`, the default): the model's schema is fetched first and each column is summarized as the type the model gives it when its values fit, so hand-tuned types such as ordinal features no longer show up as false `type_changed` findings. Values that do not fit are summarized as what they look like, so a real type change is still reported. See `get_schema_like()`. If the schema cannot be fetched the check carries on with types inferred from dtypes; pass `align_to_model=False` to always infer them (no extra request), or `schema=` to supply the summary yourself.
- The result's `baseline_source` says how the model's schema was produced, as declared when it was uploaded: `"data"`, `"hand_written"` or `"unknown"` (the default, and what models trained before this was recorded report). A hand-written baseline can legitimately differ from the data it was meant to describe, so read its findings accordingly.

```python
result = proxyml.check_data_drift("my-model", todays_df, label="2026-11-24", baseline_n_samples=4200)
if not result["passed"]:
    for f in result["features"]:
        if f["status"] != "ok":
            print(f["name"], f["reasons"])
```

Returns `None` if the request failed (for example, an unknown model).

### `list_drift_checks(model_id, limit=20)`

Recent verdicts for a model, newest first (summary only).

### `get_drift_check(model_id, snapshot_id)`

One stored verdict with its per-feature findings.

### `delete_drift_check(model_id, snapshot_id)`

Deletes one stored verdict, for example a test run. Returns `True` on success and `False` if it wasn't found. Needs alposit 0.5.2+.

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
| `PROXYML_CI_TOKEN` | *(required for `upload_challenger`/`get_challenger_summary`)* | Project-scoped CI token — mint one from the ProxyML dashboard's project page. Distinct from `PROXYML_API_KEY`: it only ever authorizes writes to the one project it was minted for. See [Continuous Integration](#continuous-integration-proxymlci). |
| `GITHUB_TOKEN` | *(required for `proxyml.ci.post_challenger_comment`)* | Set automatically inside a GitHub Actions job that declares `env: {GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}}`. |
| `GITHUB_REPOSITORY` | *(auto-set in Actions)* | `"owner/name"` — read automatically by `proxyml.ci.post_challenger_comment` unless `repo=` is passed explicitly. |
| `GITHUB_EVENT_PATH` | *(auto-set in Actions)* | Path to the triggering event's JSON payload — read automatically to find the PR number unless `pr_number=` is passed explicitly. |
