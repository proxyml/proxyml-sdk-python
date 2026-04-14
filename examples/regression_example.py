"""
Example of using ProxyML for regression.

Prerequisites:
    pip install proxyml scikit-learn
    export PROXYML_API_KEY="your-api-key"
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import fetch_california_housing

import proxyml
from proxyml import get_schema

# ---------------------------------------------------------------------------
# 1. Load a dataset and train a black-box model
# ---------------------------------------------------------------------------
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

black_box = GradientBoostingRegressor(n_estimators=100, random_state=42)
black_box.fit(df.values, y)

# ---------------------------------------------------------------------------
# 2. Generate and upload a schema
# ---------------------------------------------------------------------------

schema = get_schema(df, immutable_cols = ['Latitude', 'Longitude', 'HouseAge'])
print("\nSchema generated. Features:", [f["name"] for f in schema["features"]])

result = proxyml.put_schema(schema)
print("\nSchema upload result:", result)

# ---------------------------------------------------------------------------
# 3. Synthesize training data for the surrogate
# ---------------------------------------------------------------------------
synth_df = proxyml.synthesize_data(num_points=500)
print(f"\nSynthesized {len(synth_df)} rows with columns: {list(synth_df.columns)}")

# ---------------------------------------------------------------------------
# 4. Score synthetic data with the black-box model
# ---------------------------------------------------------------------------
predictions = black_box.predict(synth_df.values).tolist()
print(f"\nPrediction range: {min(predictions):.2f} – {max(predictions):.2f}")

# ---------------------------------------------------------------------------
# 5. Train a surrogate model
# ---------------------------------------------------------------------------
train_result = proxyml.train_surrogate(
    samples=synth_df.values.tolist(),
    predictions=predictions,
    feature_names=list(synth_df.columns),
    task="auto",  # or "regression"
    test_size=0.2,
)
print("\nSurrogate training result:", train_result)
print("\nSurrogate feature importance:", proxyml.get_feature_importances(version=train_result["version"]))

# ---------------------------------------------------------------------------
# 6. Query the surrogate
# ---------------------------------------------------------------------------
sample = df.iloc[0].tolist()
pred_result = proxyml.predict(samples=sample, version=train_result["version"])
print("\nSample 0 price:", y[0])
print("\nLocal prediction for sample 0:", black_box.predict([sample])[0])
print("\nSurrogate prediction for sample 0:", pred_result)
