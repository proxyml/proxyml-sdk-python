"""
Counterfactual explanation with ProxyML.

Assumes you have already run basic_usage.py (schema uploaded, surrogate trained).

Prerequisites:
    pip install proxyml scikit-learn
    export PROXYML_API_KEY="your-api-key"
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

import proxyml
from proxyml import get_schema

# ---------------------------------------------------------------------------
# Setup (same as basic_usage.py — run this if surrogate not already trained)
# ---------------------------------------------------------------------------
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

black_box = RandomForestClassifier(n_estimators=100, random_state=42)
black_box.fit(df.values, y)

schema = get_schema(df, immutable_cols=None)
proxyml.put_schema(schema)

synth_df = proxyml.synthesize_data(num_points=500)
predictions = black_box.predict(synth_df.values).tolist()
train_result = proxyml.train_surrogate(
    samples=synth_df.values.tolist(),
    predictions=predictions,
    feature_names=list(synth_df.columns),
    task="classification",
)
surrogate_version = train_result.get("version") if train_result else None

# ---------------------------------------------------------------------------
# Find a counterfactual for the first sample
# ---------------------------------------------------------------------------
sample = df.iloc[0].tolist()
original_label = black_box.predict([sample])[0]
target_label = 1 - original_label  # flip the prediction

print(f"Original prediction: {original_label}  →  Target: {target_label}")

cf_df = proxyml.find_counterfactual(
    sample=sample,
    target=target_label,
    n_neighbors=10000,
    perturbation_scale=0.1,
    version=surrogate_version,
)

if cf_df is None:
    print("No counterfactual found.")
else:
    print("\nCounterfactual found:")
    print(cf_df.T)

    # Check whether the original model also changed its prediction
    cf_pred = black_box.predict(cf_df.values)[0]
    prediction_changed = (cf_pred != original_label)
    print(f"\nOriginal model prediction on counterfactual: {cf_pred} (changed: {prediction_changed})")

    # Generate a human-readable explanation
    original_dict = df.iloc[0].to_dict()
    cf_dict = cf_df.iloc[0].to_dict()

    explanation = proxyml.interpret_counterfactual(
        sample=original_dict,
        counterfactual=cf_dict,
        prediction_changed=prediction_changed,
        exclude_from_diff=None,
    )
    print(f"\nExplanation:\n{explanation}")
