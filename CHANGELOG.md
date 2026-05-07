# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
