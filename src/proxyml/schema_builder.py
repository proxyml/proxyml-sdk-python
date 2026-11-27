import logging

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype

from proxyml_core.schema import (
    ArrayFeature,
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


# ── array columns (embeddings etc.) ──────────────────────────────────────────
#
# A column whose cells are equal-length numeric vectors becomes an ArrayFeature:
# a summary (width, centroid, rms radius) and nothing else. The vectors are never
# uploaded, and the server never sees them: in every request the array's slot is
# sent as null (see proxyml.client). Always immutable.

# Fewer rows than this and the centroid of a handful of vectors sits close to the
# vectors themselves, so the "summary" would leak individual rows. Matches the
# smallest cluster size the array-feature spikes treated as safe.
MIN_ARRAY_ROWS = 50


def _is_vector(v) -> bool:
    return isinstance(v, (list, tuple, np.ndarray))


def _array_matrix(s: pd.Series, name: str) -> np.ndarray:
    """Stack a column of vectors into an (n_rows, width) float matrix, or raise ValueError."""
    if any(v is None or (not _is_vector(v) and pd.isna(v)) for v in s):
        raise ValueError(f"Array column '{name}' has missing rows; drop or fill them first")
    try:
        mat = np.array([np.asarray(v, dtype=float) for v in s], dtype=float)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Array column '{name}' must contain equal-length numeric vectors ({e})") from None
    if mat.ndim != 2:
        raise ValueError(f"Array column '{name}' must contain equal-length, one-dimensional numeric vectors")
    if mat.shape[1] < 2:
        raise ValueError(f"Array column '{name}': vectors must have at least 2 values")
    if not np.isfinite(mat).all():
        raise ValueError(f"Array column '{name}' contains NaN or infinite values")
    return mat


def _is_array_column(s: pd.Series) -> bool:
    """True if every non-null cell of an object column is a list/tuple/ndarray (and there is at least one)."""
    if s.dtype != object:
        return False
    cells = [v for v in s if not (v is None or (not _is_vector(v) and pd.isna(v)))]
    return bool(cells) and all(_is_vector(v) for v in cells)


def _array_feature(s: pd.Series, name: str | None = None) -> ArrayFeature:
    name = name or s.name
    mat = _array_matrix(s, name)
    if len(mat) < MIN_ARRAY_ROWS:
        raise ValueError(
            f"Array column '{name}' has {len(mat)} rows; at least {MIN_ARRAY_ROWS} are needed, "
            "because the centroid of a few vectors is nearly the vectors themselves"
        )
    mean = mat.mean(axis=0)
    radius = float(np.sqrt(((mat - mean) ** 2).sum(axis=1).mean()))
    return ArrayFeature(name=name, width=mat.shape[1], mean=mean.tolist(), rms_radius=radius)


def get_schema(df: pd.DataFrame, immutable_cols: list[str] | None = None) -> FeatureSchema:
    """
    Generates a data schema for a pandas DataFrame, based on the data types of the columns.

    Args:
        df (pandas DataFrame): data to characterize
        immutable_cols (list): list of columns to consider immutable. These columns will
            have schema entries, but the surrogate will _not_ use them for inference.
    Returns:
        FeatureSchema. Columns of equal-length numeric vectors (lists, tuples or arrays,
        e.g. embeddings) become immutable ArrayFeatures holding only a summary; they need
        at least 50 rows. Review and adjust as needed — integer columns default to count
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
        if _is_array_column(df[col]):
            features.append(_array_feature(df[col]))  # always immutable
            continue
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
    if isinstance(base, ArrayFeature):
        if _is_array_column(s):
            return _array_feature(s, name=name)  # a different width shows up as a type_changed finding
        # not vectors any more: summarize as what it looks like, so the change is reported
        return get_schema(s.to_frame(name=name)).features[0]
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
