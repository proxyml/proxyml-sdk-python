from proxyml.client import (
    put_schema,
    synthesize_data,
    train_surrogate,
    predict,
    find_counterfactual,
    interpret_counterfactual,
    get_feature_importances,
)
from proxyml.schema import (
    get_schema,
    gen_continuous_schema,
    gen_categorical_schema,
    gen_discrete_schema,
)

__all__ = [
    "put_schema",
    "synthesize_data",
    "train_surrogate",
    "predict",
    "find_counterfactual",
    "interpret_counterfactual",
    "get_schema",
    "gen_continuous_schema",
    "gen_categorical_schema",
    "gen_discrete_schema",
]
