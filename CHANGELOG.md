# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-07-12

### Added
- `proxyml-core` dependency — schema types, the export JSON contract, and (behind
  `proxyml[local]`) shared modeling code, now shared with the backend so both sides
  score exports with the exact same arithmetic
- `proxyml.local` (requires `pip install 'proxyml[local]'`) — `train_challenger`,
  `Complexity`, `Rung`, `LADDERS` for training a linear challenger surrogate locally,
  with no round-trip to the API. Its output is a `SurrogateExport`, structurally
  identical to what `export_surrogate()` returns for a server-trained surrogate

### Changed
- **breaking**: `get_schema()` (and the schema it builds) now returns a
  `proxyml_core.schema.FeatureSchema` of typed `Feature` objects instead of a plain
  dict; `gen_continuous_schema`/`gen_categorical_schema`/`gen_discrete_schema` are
  removed — construct `proxyml_core.schema` `Feature` subclasses directly if you
  need that level of control
- **breaking**: `put_schema()` now takes a `FeatureSchema` (not a dict) and returns
  one; `fetch_schema()` and `get_model_schema()` now return a `FeatureSchema`
  instead of a raw dict
- **breaking**: `export_surrogate()` now returns a `proxyml_core.export.SurrogateExport`
  instead of a raw dict — score it locally with zero sklearn via
  `proxyml_core.export.predict_from_export(export, sample)`, which (unlike the old
  `examples/surrogate_export_example.py` snippet) correctly handles every feature
  type (continuous, count, categorical, both ordinal types) and multiclass

## [0.2.2] - 2026-05-25

### Changed
- **breaking**: `schema_name` is now a required keyword argument on `synthesize_data`
  and `train_surrogate` — it previously defaulted to `"default"`, which made it easy
  to silently train against the wrong schema

## [0.2.1] - 2026-05-14

### Added
- `health_check()` — does not require auth or count against quota

### Fixed
- Network reliability, import hygiene, and a stale test fixture

## [0.2.0] - 2026-05-07

### Added
- `explain_local_batch(instances, version=None)` — per-feature contribution breakdown for multiple instances in one API call (`POST /explain/local/batch`)
- `update_model(version, name=..., comments=...)` — update a surrogate's name or comments without retraining (`PATCH /surrogate/models/{version}`)
- `patch()` — internal HTTP helper for PATCH requests

### Changed
- `list_models()` now accepts `limit` (default 50, max 200) and `offset` parameters for pagination
- `list_models()` now returns `{"models": [...], "total": N}` instead of a bare list — **breaking change**
- `diff_models()` response now includes `per_class_coefficient_diff` for multiclass models
- `get_model_summary()` and model list entries now include `mlflow_run_id` for correlation with the MLflow UI

### Fixed
- Corrected `predict()` parameter name in quickstart example (`sample=` not `samples=`)
- Corrected `find_counterfactuals()` parameter name in quickstart example (`samples=` not `instances=`)

## [0.1.0] - 2026-04-12

### Added
- `get_schema` / `put_schema` — infer and upload feature schemas from a pandas DataFrame
- `synthesize_data` — generate synthetic data points via neighbor sampling or blended synthesis
- `train_surrogate` — train a surrogate model from samples and black-box predictions
- `predict` — score samples using a trained surrogate model
- `find_counterfactual` — search for a minimal-change counterfactual explanation
- `interpret_counterfactual` — produce a human-readable summary of counterfactual differences
