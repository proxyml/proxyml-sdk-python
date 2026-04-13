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

# Mark features that should not be changed in counterfactuals
schema = get_schema(df, immutable_cols=["age", "gender"])

# Review and adjust the schema before uploading
# Integer columns default to `count` type — change to `categorical_ordinal`
# for ordered categories like ratings or class labels
print(schema)

# Upload to ProxyML
put_schema(schema)
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
synth_df = synthesize_data(num_points=1000)

# Or generate neighbors around a specific instance
instance = df.iloc[0].tolist()
local_df = synthesize_data(num_points=200, sample=instance)
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
)
print(result)  # accuracy, version number, etc.
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

results = predict(samples=[sample], version=None)
print(results)
```

## Next Steps

- See [`api.md`](api.md) for complete parameter documentation.
- See [`../examples/`](../examples/) for runnable scripts.
