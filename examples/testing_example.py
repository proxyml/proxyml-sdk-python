"""
Example of using ProxyML to test a model in lower environments.

Ever wonder how to write unit tests for a machine learning model? 
You don't want to bundle real data with your tests, but you feel like you need to do more than just assert the model returns a prediction.

Train a surrogate in ProxyML and use it as a reference implementation!

Prerequisites:
    pip install proxyml scikit-learn
    export PROXYML_API_KEY="your-api-key"
"""
import proxyml
from proxyml import get_schema

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import fetch_california_housing


# Scenario: we have a model in production, but we need a test to validate it as part of a deployment pipeline
print("Fetching dataset...")
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target
print(df)

# Our current production model
print("\nTraining production model...")
production_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
production_model.fit(df.values, y)

# Setting up our surrogate in ProxyML
problem_name = "California Housing Regressor Validator"

# Upload data schema
print("\nGenerating and uploading data schema...")
schema = get_schema(df, immutable_cols = ['Latitude', 'Longitude', 'HouseAge'])
proxyml.put_schema(schema, name=problem_name)

# Generate synthetic training data
print("\nGenerating surrogate training data...")
synth_df = proxyml.synthesize_data(num_points=500, schema_name=problem_name)
predictions = production_model.predict(synth_df.values).tolist()

# Train surrogate
print("\nTraining ProxyML surrogate...")
train_result = proxyml.train_surrogate(
    samples=synth_df.values.tolist(),
    predictions=predictions,
    feature_names=list(synth_df.columns),
    task="auto",  # or "regression"
    test_size=0.2,
    name=problem_name,
    schema_name=problem_name,
    comments="Validates agreement between expected predictions and actual predictions"
)


def test_model_predictions_within_expected_range(
        dev_model, 
        schema_name, 
        surrogate_version,
        tolerance = 0.1  # in target units e.g. 0.1 = $10,000 for California Housing
    ):
    synthetic = proxyml.synthesize_data(num_points=100, schema_name=schema_name)
    dev_predictions = dev_model.predict(synthetic.values)
    surrogate_response = proxyml.predict_batch(synthetic.values.tolist(), version=surrogate_version)
    surrogate_predictions = surrogate_response['predictions']
    
    # predictions should broadly agree
    mean_diff = np.mean(np.abs(dev_predictions - surrogate_predictions))
    # Assert that the mean difference is less than or equal to the tolerance
    assert mean_diff <= tolerance, f"Mean difference {mean_diff} exceeds tolerance {tolerance}"

# Our current development model
print("\nTraining local development model...")
development_model = GradientBoostingRegressor(n_estimators=101, random_state=43)
development_model.fit(df.values, y)

print("\nValidating development model against expected production model results...")
test_model_predictions_within_expected_range(
    dev_model=development_model,
    schema_name=problem_name,
    surrogate_version=train_result['version']
)
