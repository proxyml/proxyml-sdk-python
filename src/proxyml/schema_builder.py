import logging

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype

from proxyml_core.schema import (
    CategoricalFeature,
    CategoricalOrdinalFeature,
    ContinuousFeature,
    CountFeature,
    Feature,
    FeatureSchema,
    NumericOrdinalFeature,
)

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


# ── summarizing data against an existing schema's types ──────────────────────
#
# get_schema() infers types from dtypes, so a column the model's schema treats as
# ordinal (or count, or categorical) can come out as something else. For a drift
# check that is a false "type_changed". get_schema_like() summarizes each column
# as the *baseline's* type instead, but only when the values really fit it. A
# value that does not fit is summarized as what it looks like, so a genuine type
# change still shows up. These are the same rules as the dashboard's CSV check.


def _as_numbers(s: pd.Series) -> pd.Series | None:
    """The column as numbers, or None if any value is not numeric (booleans are not)."""
    if s.dtype == bool:
        return None
    nums = pd.to_numeric(s, errors="coerce")
    return None if nums.isna().any() else nums


def _label(v):
    """A category label as get_schema() would store it: 1.0 -> 1, numpy scalars -> Python."""
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def _frequencies(s: pd.Series) -> dict:
    counts: dict = {}
    for v, share in s.value_counts(normalize=True).items():
        counts[_label(v)] = counts.get(_label(v), 0.0) + float(share)
    return counts


def _as_observed(s: pd.Series, name: str, nums: pd.Series | None, immutable: bool) -> Feature:
    if nums is not None:
        return _continuous_feature(nums, name=name, immutable=immutable)
    return CategoricalFeature(name=name, immutable=immutable, valid_categories=_frequencies(s))


def _summarize_like(base: Feature, s: pd.Series) -> Feature:
    name, immutable = base.name, base.immutable
    nums = _as_numbers(s)
    if isinstance(base, ContinuousFeature):
        if nums is None:
            return _as_observed(s, name, nums, immutable)
        return _continuous_feature(nums.astype(float), name=name, immutable=immutable)
    if isinstance(base, CountFeature):
        if nums is None or not (nums % 1 == 0).all() or (nums < 0).any():
            return _as_observed(s, name, nums, immutable)
        return CountFeature(name=name, immutable=immutable, lambda_=float(nums.mean()), max=int(nums.max()))
    if isinstance(base, NumericOrdinalFeature):
        if nums is None or not (nums % 1 == 0).all():
            return _as_observed(s, name, nums, immutable)
        return NumericOrdinalFeature(
            name=name, immutable=immutable,
            categories=sorted({int(v) for v in nums}),
            mean=float(nums.mean()), std=float(nums.std(ddof=0)),
        )
    if isinstance(base, CategoricalOrdinalFeature):
        freq = _frequencies(s)
        by_text = {str(k): v for k, v in freq.items()}
        known = {str(c) for c in base.categories}
        # Keep the baseline's category order and append values it has never seen,
        # so a new category is visible to the comparison.
        extra = [k for k in freq if str(k) not in known]
        categories = [*base.categories, *extra]
        probabilities = [by_text.get(str(c), 0.0) for c in base.categories] + [freq[k] for k in extra]
        return CategoricalOrdinalFeature(name=name, immutable=immutable, categories=categories, probabilities=probabilities)
    return CategoricalFeature(name=name, immutable=immutable, valid_categories=_frequencies(s))


def get_schema_like(df: pd.DataFrame, baseline: FeatureSchema) -> FeatureSchema:
    """
    Summarizes a DataFrame using the column types of an existing schema.

    Use this to compare recent data against a model's training schema when that
    schema has hand-tuned types (ordinal features, say) that get_schema() cannot
    infer from dtypes. Columns the baseline has are summarized as the baseline's
    type when their values fit it; values that do not fit are summarized as the
    type they look like, so a real type change is still reported. Columns the
    baseline does not have are summarized as get_schema() would. Columns that are
    entirely empty are left out.

    Args:
        df (pandas DataFrame): data to characterize
        baseline (FeatureSchema): schema whose types to follow, e.g. from get_model_schema()
    Returns:
        FeatureSchema (statistics only, never rows).
    """
    known = {f.name: f for f in baseline.features}
    features: list[Feature] = []
    for col in df.columns:
        s = df[col].dropna()
        if col in known:
            if len(s) == 0:
                continue
            features.append(_summarize_like(known[col], s))
        else:
            features.extend(get_schema(df[[col]]).features)
    return FeatureSchema(features=features)
