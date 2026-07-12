# proxyml

Python SDK for the [ProxyML API](https://proxyml.ai).

<img width="577" height="115" alt="creditg_car_counterfactual" src="https://github.com/user-attachments/assets/8f913c95-1190-4b3c-ba7d-6690e9ebb3e0" />

## Why ProxyML?

Most explainability tools require sending your data to a third-party server. 
ProxyML never sees your training data. You generate synthetic data locally, 
score it with your own model, and only the surrogate model and summary 
statistics are uploaded — your data stays yours.

## Installation

```bash
pip install proxyml
```

Want to train a challenger model locally, with no round-trip to the API? Install the `local` extra (adds scikit-learn and scipy):

```bash
pip install 'proxyml[local]'
```

## Setup

ProxyML requires an API key. Set it as an environment variable before importing the package:

```bash
export PROXYML_API_KEY="your-api-key"
```

Optionally override the base URL (defaults to `https://api.proxyml.ai/api/v1`):

```bash
export PROXYML_BASE_URL="https://api.proxyml.ai/api/v1"
```

## Quick Start

```python
import pandas as pd
import proxyml
from proxyml import get_schema

# 1. Load your dataset and generate a schema
df = pd.read_csv("data.csv")
schema = get_schema(df, immutable_cols=["age", "gender"])

# 2. Upload the schema under a name — synthesis and training reference it by name
SCHEMA_NAME = "my_schema"
proxyml.put_schema(schema, name=SCHEMA_NAME)

# 3. Generate synthetic training data
synth_df = proxyml.synthesize_data(num_points=500, schema_name=SCHEMA_NAME)

# 4. Score synthetic data with your black-box model
predictions = my_model.predict(synth_df.values.tolist())

# 5. Train a surrogate model
proxyml.train_surrogate(
    samples=synth_df.values.tolist(),
    predictions=predictions,
    feature_names=list(synth_df.columns),
    schema_name=SCHEMA_NAME,
)

# 6. Find a counterfactual explanation
sample = df.iloc[0].tolist()
cf = proxyml.find_counterfactual(sample=sample, target=1, version=None)

if cf is not None:
    original = synth_df.iloc[0].to_dict()
    cf_dict = cf.iloc[0].to_dict()
    current_pred = my_model.predict([sample])[0]
    cf_pred = my_model.predict([cf.values.tolist()[0]])[0]

    explanation = proxyml.interpret_counterfactual(
        sample=original,
        counterfactual=cf_dict,
        prediction_changed=(current_pred != cf_pred),
    )
    print(explanation)
```

See [`docs/quickstart.md`](docs/quickstart.md) for a full walkthrough and [`docs/api.md`](docs/api.md) for complete API reference.

## Core Concepts

**Surrogate model** — A fast, interpretable model trained to approximate your black-box model's behavior on synthetic data. Once trained, it can be queried directly via the ProxyML API.

**Challenger model** — Trained the same way as a surrogate, but locally (`proxyml.local`, no round-trip to the API) and against either real ground-truth labels (a genuine challenger to compare against a champion model) or a black box's predictions (a surrogate/explainer). Its export is structurally identical to a server-trained surrogate's, so the two can be compared and scored with the exact same arithmetic.

**Counterfactual explanation** — Given a prediction, a counterfactual is the minimal change to input features that would produce a different prediction. It answers: *"What would have to be different for the outcome to change?"*

**Schema** — Describes the statistical properties of each feature (type, range, distribution). Used to generate realistic synthetic data and constrain counterfactual search.

## API Reference

| Function | Description |
|---|---|
| `get_schema(df, immutable_cols)` | Infer a `FeatureSchema` from a DataFrame |
| `put_schema(schema, name)` | Upload a schema to the API under a name |
| `synthesize_data(num_points, sample, as_df, schema_name)` | Generate synthetic data points |
| `train_surrogate(samples, predictions, feature_names, task, test_size, schema_name)` | Train a surrogate model |
| `train_auto_surrogate(data, target_col, ...)` | Load data + train a surrogate in one call, skipping synthesis |
| `export_surrogate(version)` | Export a surrogate for offline scoring via `predict_from_export` |
| `predict(sample, version)` | Score a single sample with the surrogate model |
| `find_counterfactual(sample, target, ...)` | Find a counterfactual for a given sample |
| `interpret_counterfactual(sample, counterfactual, ...)` | Generate a human-readable explanation |
| `proxyml.local.train_challenger(df, target, schema, ...)` | Train a challenger model locally — no API round-trip (`pip install 'proxyml[local]'`) |
| `proxyml.local.train_auto_challenger(data, target_col, ...)` | Load data + train a local challenger in one call |

Full documentation: [`docs/api.md`](docs/api.md)

## Examples

- [`examples/basic_usage.py`](examples/basic_usage.py) — Schema upload, data synthesis, surrogate training
- [`examples/counterfactual_example.py`](examples/counterfactual_example.py) — Counterfactual search and interpretation
- [`examples/regression_example.py`](examples/regression_example.py) — Regression with immutable features
- [`examples/multiclass_example.py`](examples/multiclass_example.py) — Multi-class classification with per-class feature importances
- [`examples/testing_example.py`](examples/testing_example.py) — Using a surrogate as a reference model in CI
- [`examples/surrogate_export_example.py`](examples/surrogate_export_example.py) — Exporting a surrogate and reproducing its predictions locally

## License

Apache 2.0 — see [LICENSE](LICENSE).
