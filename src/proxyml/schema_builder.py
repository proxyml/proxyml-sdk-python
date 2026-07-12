import logging

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype

from proxyml_core.schema import CategoricalFeature, ContinuousFeature, CountFeature, Feature, FeatureSchema

logger = logging.getLogger(__name__)


def _continuous_feature(s: pd.Series, name: str | None = None, immutable: bool = False) -> ContinuousFeature:
    return ContinuousFeature(
        name=name or s.name,
        immutable=immutable,
        mean=np.nanmean(s).item(),
        std=np.nanstd(s).item(),
        min=np.nanmin(s).item(),
        max=np.nanmax(s).item(),
    )


def _categorical_feature(s: pd.Series, name: str | None = None, immutable: bool = False) -> CategoricalFeature:
    counts = s.value_counts(normalize=True)
    return CategoricalFeature(name=name or s.name, immutable=immutable, valid_categories=counts.to_dict())


def _count_feature(s: pd.Series, name: str | None = None, immutable: bool = False) -> CountFeature:
    return CountFeature(
        name=name or s.name,
        immutable=immutable,
        lambda_=np.nanmean(s).item(),
        max=np.nanmax(s).item(),
    )


def get_schema(df: pd.DataFrame, immutable_cols: list[str] | None = None) -> FeatureSchema:
    """
    Generates a data schema for a pandas DataFrame, based on the data types of the columns.

    Args:
        df (pandas DataFrame): data to characterize
        immutable_cols (list): list of columns to consider immutable. These columns will
            have schema entries, but the surrogate will _not_ use them for inference.
    Returns:
        FeatureSchema. Review and adjust as needed — integer columns default to count
        type; consider categorical_ordinal for ordered categories like ratings or
        class labels.
    """
    if immutable_cols:
        unknown = set(immutable_cols) - set(df.columns)
        if unknown:
            logger.warning("immutable_cols not found in DataFrame and will be ignored: %s", sorted(unknown))

    features: list[Feature] = []
    for col in df.columns:
        immutable = immutable_cols is not None and col in immutable_cols
        if is_float_dtype(df[col]):
            feature = _continuous_feature(df[col], immutable=immutable)
        elif df[col].dtype == bool:
            feature = _categorical_feature(df[col], immutable=immutable)
        elif is_integer_dtype(df[col]):
            feature = _count_feature(df[col], immutable=immutable)
        else:
            feature = _categorical_feature(df[col], immutable=immutable)
        features.append(feature)
    return FeatureSchema(features=features)
