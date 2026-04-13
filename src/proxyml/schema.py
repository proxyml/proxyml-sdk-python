import logging
logger = logging.getLogger(__name__)

import os
import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype


def gen_continuous_schema(s: pd.Series, name: str | None = None) -> dict:
    return {
        'type': 'continuous',
        'name': name or s.name,
        'mean': np.nanmean(s).item(),
        'std': np.nanstd(s).item(),
        'min': np.nanmin(s).item(),
        'max': np.nanmax(s).item()
    }


def gen_categorical_schema(s: pd.Series, name: str | None = None) -> dict:
    counts = s.value_counts(normalize=True)                                                                                                                                         
    return {                                        
        'type': 'categorical',
        'name': name or s.name,          
        'valid_categories': counts.to_dict()
    }


def gen_discrete_schema(s: pd.Series, name: str | None = None) -> dict:
    return {                                        
        'type': 'count',
        'name': name or s.name,    
        'lambda': np.nanmean(s).item(),
        'max': np.nanmax(s).item()
    }


def get_schema(df: pd.DataFrame, immutable_cols: list[str] | None) -> dict:
    schema = {
        'features': list(),
        '_note': (
            'Auto-generated schema. Review and adjust types as needed. '
            'Integer columns default to count type - consider categorical_ordinal '
            'for ordered categories like ratings or class labels.'
        )
    }
    for col in df.columns:
        if is_float_dtype(df[col]):
            col_schema = gen_continuous_schema(df[col])
        elif df[col].dtype == bool:
            col_schema = gen_categorical_schema(df[col])
        elif is_integer_dtype(df[col]):
            col_schema = gen_discrete_schema(df[col])
        else:
            col_schema = gen_categorical_schema(df[col])
        col_schema['immutable'] = immutable_cols is not None and col in immutable_cols
        schema['features'].append(col_schema)
    return schema