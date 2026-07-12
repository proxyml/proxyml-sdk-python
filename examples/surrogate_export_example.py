"""
surrogate_export_example - demonstrates exporting a surrogate from ProxyML and using it locally

Prerequisites:
    pip install proxyml scikit-learn catboost pandas
    export PROXYML_API_KEY="your-api-key"
"""
from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from catboost import CatBoostRegressor, Pool

import proxyml
from proxyml import get_schema
from proxyml_core.export import predict_from_export

# Fetch the Ames housing dataset
housing = fetch_openml(name="house_prices", as_frame=True, parser='auto')

# Extract features and target
X = housing.data
y = housing.target
# We'll use imputation to handle missing values - in a "real"
# model you'd probably wrap everything together in a pipline
imputer = KNNImputer(n_neighbors=10, weights="distance")
# separate numeric and categorical
numeric_cols = X.select_dtypes(include='number').columns
categorical_cols = X.select_dtypes(exclude='number').columns

# impute numeric with KNN
X_numeric_imputed = pd.DataFrame(
    imputer.fit_transform(X[numeric_cols]),
    columns=numeric_cols,
    index=X.index
)

# impute categorical with most frequent
cat_imputer = SimpleImputer(strategy='most_frequent')
X_categorical_imputed = pd.DataFrame(
    cat_imputer.fit_transform(X[categorical_cols]),
    columns=categorical_cols,
    index=X.index
)

# recombine
X_imputed = pd.concat([X_numeric_imputed, X_categorical_imputed], axis=1)
# restore original column order
X_imputed = X_imputed[X.columns]

# Features we'll allow our black box model to consider
BLACK_BOX_FEATURES = [
    '1stFlrSF',
    'Alley',
    'BldgType',
    'BsmtCond',
    'BsmtExposure',
    'BsmtFinSF1',
    'BsmtFinType1',
    'BsmtQual',
    'CentralAir',
    'Condition1',
    'Condition2',
    'Electrical',
    'ExterCond',
    'ExterQual',
    'Exterior1st',
    'Exterior2nd',
    'Fence',
    'FireplaceQu',
    'Fireplaces',
    'Foundation',
    'Functional',
    'GarageArea',
    'GarageCars',
    'GarageCond',
    'GarageFinish',
    'GarageQual',
    'GarageType',
    'GrLivArea',
    'Heating',
    'HeatingQC',
    'HouseStyle',
    'KitchenQual',
    'LandContour',
    'LandSlope',
    'LotArea',
    'LotConfig',
    'LotFrontage',
    'MSSubClass',
    'MSZoning',
    'MasVnrArea',
    'MasVnrType',
    'Neighborhood',
    'OpenPorchSF',
    'OverallCond',
    'OverallQual',
    'PavedDrive',
    'PoolArea',
    'RoofMatl',
    'RoofStyle',
    'ScreenPorch',
    'Street',
    'TotalBsmtSF',
    'WoodDeckSF',
    'YearBuilt',
    'YearRemodAdd',
]

# Features we'll allow our explanation surrogate to consider - 
# we want to understand how our black box model arrives at its
# predictions, so we'd likely use every feature it uses
EXPLANATION_FEATURES = list(BLACK_BOX_FEATURES)  


# identify categorical columns in your feature set
cat_cols_in_black_box_features = [
    col for col in BLACK_BOX_FEATURES
    if pd.api.types.is_string_dtype(X_imputed[col])
    or pd.api.types.is_object_dtype(X_imputed[col])
]

black_box_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    random_seed=42,
    verbose=0
)

# use Pool instead of raw arrays - it handles column names explicitly
train_pool = Pool(
    data=X_imputed[BLACK_BOX_FEATURES],
    label=y,
    cat_features=cat_cols_in_black_box_features
)
black_box_model.fit(train_pool)

black_box_model.fit(
    X_imputed[BLACK_BOX_FEATURES], 
    y,
    cat_features=cat_cols_in_black_box_features
)

# Building our explainer surrogate
explanation_name = 'AmesHousingExplanation'
explanation_schema = get_schema(X_imputed, immutable_cols=None)
_ = proxyml.put_schema(explanation_schema, name=explanation_name)

explanation_synthetic_df = proxyml.synthesize_data(num_points=500, schema_name=explanation_name)
synth_pool = Pool(
    data=explanation_synthetic_df[BLACK_BOX_FEATURES],
    cat_features=cat_cols_in_black_box_features
)
black_box_predictions = black_box_model.predict(synth_pool).tolist()
explanation_train_result = explanation_surrogate = proxyml.train_surrogate(
    samples=explanation_synthetic_df.values.tolist(),
    schema_name=explanation_name,
    predictions=black_box_predictions,
    feature_names=EXPLANATION_FEATURES,
    task="auto",
    test_size=0.2,
)
print("\nSurrogate training result:", explanation_train_result)
print("\nSurrogate feature importance:", proxyml.get_feature_importances(version=explanation_train_result["version"]))

exported_explanation_surrogate = proxyml.export_surrogate(version=explanation_train_result['version'])

# predict_from_export reconstructs surrogate predictions from the export alone —
# no ProxyML SDK or API required, and no sklearn either (proxyml_core.export is
# pure arithmetic). It handles every feature type the export can carry
# (continuous, count, categorical, both ordinal types) and multiclass, not just
# the continuous/categorical subset a hand-rolled version might cover.

# verify against the API - get a sample of data and compare remote and local predictions
sample = X_imputed.sample().to_dict(orient='records')[0]

local_prediction = predict_from_export(exported_explanation_surrogate, sample)
api_prediction = proxyml.predict(sample=list(sample.values()))

print(f"Local prediction:  ${local_prediction:,.0f}")
print(f"API prediction:    ${api_prediction['prediction']:,.0f}")
print(f"Difference:        ${abs(local_prediction - api_prediction['prediction']):,.2f}")