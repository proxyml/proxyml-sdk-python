# proxyml

Python SDK for the [ProxyML API](https://proxyml.ai).

> **Status:** Early access — server endpoints coming soon.  
> [Request early access](mailto:contact@proxyml.ai) or star this repo to follow progress.

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

# 2. Upload the schema
proxyml.put_schema(schema)

# 3. Generate synthetic training data
synth_df = proxyml.synthesize_data(num_points=500)

# 4. Score synthetic data with your black-box model
predictions = my_model.predict(synth_df.values.tolist())

# 5. Train a surrogate model
proxyml.train_surrogate(
    samples=synth_df.values.tolist(),
    predictions=predictions,
    feature_names=list(synth_df.columns),
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

**Counterfactual explanation** — Given a prediction, a counterfactual is the minimal change to input features that would produce a different prediction. It answers: *"What would have to be different for the outcome to change?"*

**Schema** — Describes the statistical properties of each feature (type, range, distribution). Used to generate realistic synthetic data and constrain counterfactual search.

## API Reference

| Function | Description |
|---|---|
| `get_schema(df, immutable_cols)` | Generate a schema dict from a DataFrame |
| `put_schema(schema)` | Upload a schema to the API |
| `synthesize_data(num_points, sample, as_df)` | Generate synthetic data points |
| `train_surrogate(samples, predictions, feature_names, task, test_size)` | Train a surrogate model |
| `predict(samples, version)` | Score samples with the surrogate model |
| `find_counterfactual(sample, target, ...)` | Find a counterfactual for a given sample |
| `interpret_counterfactual(sample, counterfactual, ...)` | Generate a human-readable explanation |

Full documentation: [`docs/api.md`](docs/api.md)

## Examples

- [`examples/basic_usage.py`](examples/basic_usage.py) — Schema upload, data synthesis, surrogate training
- [`examples/counterfactual_example.py`](examples/counterfactual_example.py) — Counterfactual search and interpretation

## License

Apache 2.0 — see [LICENSE](LICENSE).
