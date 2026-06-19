# Quick Start

## Prerequisites

Install the package and set your API key:

```bash
pip install proxyml
export PROXYML_API_KEY="your-api-key"
```

## Step 1 — Define a Schema

A schema tells ProxyML how to treat each feature: its type, range, and whether it can be changed during counterfactual search.

```python
import pandas as pd
from proxyml import get_schema, put_schema

df = pd.read_csv("data.csv")

# Schemas are named — synthesis and training reference them by this name
SCHEMA_NAME = "my_schema"

# Mark features that should not be changed in counterfactuals
schema = get_schema(df, immutable_cols=["age", "gender"])

# Review and adjust the schema before uploading
# Integer columns default to `count` type — change to `categorical_ordinal`
# for ordered categories like ratings or class labels
print(schema)

# Upload to ProxyML
put_schema(schema, name=SCHEMA_NAME)
```

The schema is auto-generated from column dtypes:

| dtype | Schema type | Notes |
|---|---|---|
| `float` | `continuous` | mean, std, min, max |
| `int` | `count` | Poisson lambda, max |
| `bool` | `categorical` | true/false frequencies |
| `object` | `categorical` | category frequencies |

Change `count` to `categorical_ordinal` for ordered integers (ratings, labels).

## Step 2 — Synthesize Training Data

Generate synthetic data that respects your schema's distributions:

```python
from proxyml import synthesize_data

# Sample from the learned distribution
synth_df = synthesize_data(num_points=1000, schema_name=SCHEMA_NAME)

# Or generate neighbors around a specific instance
instance = df.iloc[0].tolist()
local_df = synthesize_data(num_points=200, sample=instance, schema_name=SCHEMA_NAME)
```

## Step 3 — Train a Surrogate Model

Score the synthetic data with your black-box model, then train a surrogate:

```python
from proxyml import train_surrogate

predictions = my_model.predict(synth_df.values.tolist())

result = train_surrogate(
    samples=synth_df.values.tolist(),
    predictions=predictions,
    feature_names=list(synth_df.columns),
    task="auto",       # "auto", "classification", or "regression"
    test_size=0.2,
    schema_name=SCHEMA_NAME,
)
print(result)  # version UUID, metrics, etc.
version = result["version"]
```

## Step 4 — Find a Counterfactual

Given a sample and a desired outcome, find the minimal feature changes needed:

```python
from proxyml import find_counterfactual, interpret_counterfactual

sample = df.iloc[0].tolist()
target_label = 1  # the outcome you want

cf_df = find_counterfactual(
    sample=sample,
    target=target_label,
    n_neighbors=10000,
    perturbation_scale=0.1,
    version=None,  # None = latest surrogate version
)

if cf_df is not None:
    original = df.iloc[0].to_dict()
    cf_dict = cf_df.iloc[0].to_dict()

    original_pred = my_model.predict([sample])[0]
    cf_pred = my_model.predict([cf_df.values.tolist()[0]])[0]

    explanation = interpret_counterfactual(
        sample=original,
        counterfactual=cf_dict,
        prediction_changed=(original_pred != cf_pred),
        exclude_from_diff=["id", "timestamp"],
    )
    print(explanation)
```

## Step 5 — Query the Surrogate Directly

```python
from proxyml import predict

result = predict(sample=sample, version=None)
print(result)  # {"prediction": 1, "model_version": "surrogate-<uuid>-classification"}
```

---

## Batch Operations

For large workloads, use batch endpoints to make a single API call instead of one per instance.

### Batch Predictions

```python
from proxyml import predict_batch

samples = df.values.tolist()  # multiple rows
result = predict_batch(samples=samples)
print(result["predictions"])    # one prediction per row
```

### Batch Counterfactuals

```python
from proxyml import find_counterfactuals

samples = df.iloc[:10].values.tolist()
results = find_counterfactuals(samples=samples, target=1)

# results is a list — one DataFrame (or None) per input instance
for i, cf in enumerate(results):
    if cf is not None:
        print(f"Instance {i}: {cf.iloc[0].to_dict()}")
    else:
        print(f"Instance {i}: no counterfactual found")
```

### Batch Local Explanations

```python
from proxyml import explain_local_batch

instances = df.iloc[:10].values.tolist()
result = explain_local_batch(instances=instances)

for item in result["results"]:
    top = item["feature_contributions"][0]
    print(f"prediction={item['prediction']}  top feature={top['feature']} ({top['contribution']:+.3f})")
```

---

## Model Management & Inspection

### Inspect a Model

`get_model_summary` returns feature importances, fidelity metrics, and metadata in one call:

```python
from proxyml import get_model_summary

summary = get_model_summary()          # latest version
print(summary["metrics"])              # e.g. {"r2": 0.92}
print(summary["feature_importances"]) # ranked by impact
```

### Compare Two Versions

After retraining on updated data, use `diff_models` to see what changed:

```python
from proxyml import diff_models

diff = diff_models(version_a="<old-uuid>", version_b="<new-uuid>")

print(diff["metric_diff"])  # {"r2": {"a": 0.87, "b": 0.92, "delta": 0.05}}

for entry in diff["coefficient_diff"]:
    print(f"{entry['feature']}: {entry['a']:.3f} → {entry['b']:.3f} (Δ {entry['delta']:+.3f})")

print("Added features:", diff["features_added"])
print("Removed features:", diff["features_removed"])
```

### List and Delete Models

```python
from proxyml import list_models, delete_model, update_model

result = list_models()             # returns {"models": [...], "total": N}
print(f"{result['total']} total models")
for m in result["models"]:
    print(m["version"], m["task"], m["metrics"], m["trained_at"])

# Paginate through all models
all_models, offset = [], 0
while True:
    page = list_models(limit=50, offset=offset)
    all_models.extend(page["models"])
    if len(all_models) >= page["total"]:
        break
    offset += 50

# Update a model's label without retraining
update_model("<uuid>", name="prod-v3", comments="retrained on May data")

# Delete a specific version
delete_model("<uuid>")
```

---

## Next Steps

- See [`api.md`](api.md) for complete parameter documentation.
- See [`../examples/`](../examples/) for runnable scripts.
