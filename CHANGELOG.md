# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-12

### Added
- `get_schema` / `put_schema` — infer and upload feature schemas from a pandas DataFrame
- `synthesize_data` — generate synthetic data points via neighbor sampling or blended synthesis
- `train_surrogate` — train a surrogate model from samples and black-box predictions
- `predict` — score samples using a trained surrogate model
- `find_counterfactual` — search for a minimal-change counterfactual explanation
- `interpret_counterfactual` — produce a human-readable summary of counterfactual differences
