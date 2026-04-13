"""
Basic ProxyML usage: schema upload, data synthesis, surrogate training.

Prerequisites:
    pip install proxyml scikit-learn
    export PROXYML_API_KEY="your-api-key"
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

import proxyml
from proxyml import get_schema

# ---------------------------------------------------------------------------
# 1. Load a dataset and train a black-box model
# ---------------------------------------------------------------------------
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

black_box = RandomForestClassifier(n_estimators=100, random_state=42)
black_box.fit(df.values, y)

# ---------------------------------------------------------------------------
# 2. Generate and upload a schema
# ---------------------------------------------------------------------------
schema = get_schema(df, immutable_cols=None)
print("Schema generated. Features:", [f["name"] for f in schema["features"]])

result = proxyml.put_schema(schema)
print("Schema upload result:", result)

# ---------------------------------------------------------------------------
# 3. Synthesize training data for the surrogate
# ---------------------------------------------------------------------------
synth_df = proxyml.synthesize_data(num_points=500)
print(f"Synthesized {len(synth_df)} rows with columns: {list(synth_df.columns)}")

# ---------------------------------------------------------------------------
# 4. Score synthetic data with the black-box model
# ---------------------------------------------------------------------------
predictions = black_box.predict(synth_df.values).tolist()
print(f"Label distribution: {np.unique(predictions, return_counts=True)}")

# ---------------------------------------------------------------------------
# 5. Train a surrogate model
# ---------------------------------------------------------------------------
train_result = proxyml.train_surrogate(
    samples=synth_df.values.tolist(),
    predictions=predictions,
    feature_names=list(synth_df.columns),
    task="classification",
    test_size=0.2,
)
print("Surrogate training result:", train_result)

# ---------------------------------------------------------------------------
# 6. Query the surrogate
# ---------------------------------------------------------------------------
sample = df.iloc[0].tolist()
pred_result = proxyml.predict(samples=sample, version=None)
print("Sample 0 classification:", y[0])
print("Local prediction for sample 0:", black_box.predict_proba([sample])[0])
print("Surrogate prediction for sample 0:", pred_result)
