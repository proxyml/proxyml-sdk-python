import logging
logger = logging.getLogger(__name__)

import requests
import os
import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype


PROXYML_BASE_URL = os.getenv("PROXYML_BASE_URL", "https://api.proxyml.ai/api/v1")

_BOOL_STRINGS = {"true", "false"}


def _headers() -> dict:
    api_key = os.getenv("PROXYML_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "PROXYML_API_KEY is not set. "
            "Set the environment variable before using the SDK."
        )
    return {
        'Content-Type': 'application/json',
        'X-API-KEY': api_key,
    }


def post(endpoint: str, payload: dict) -> requests.models.Response:
    r = requests.post(
        url=f'{PROXYML_BASE_URL}{endpoint}',
        json=payload,
        headers=_headers()
    )
    return r


def put(endpoint: str, payload: dict) -> requests.models.Response:
    r = requests.put(
        url=f'{PROXYML_BASE_URL}{endpoint}',
        json=payload,
        headers=_headers()
    )
    return r





def put_schema(schema: dict):
    r = put(endpoint='/schema', payload=schema)
    if r.status_code == 200:
        logger.info("Schema uploaded successfully")
        return r.json()
    logger.error(
        "Schema upload failed with status %s: %s",
        r.status_code,
        r.text
    )
    return None


def _cast_column(series: pd.Series, ftype: str) -> pd.Series:                                                                                                                         
    if ftype in ("continuous",):                                                                                                                                                      
        return series.astype(float)                                                                                                                                                   
    if ftype in ("count", "numeric_ordinal"):                                                                                                                                         
        return series.astype(int)
    if ftype in ("categorical", "categorical_ordinal"):                                                                                                                               
        # convert "true"/"false" strings back to booleans when appropriate
        unique = {str(v).lower() for v in series.dropna().unique()}                                                                                                                   
        if unique <= _BOOL_STRINGS:                                                                                                                                                   
            return series.map({"true": True, "false": False, True: True, False: False})                                                                                               
    return series                                                                                                                                                                     
                                                            
                                                                                                                                                                                        
def synthesize_data(num_points: int = 100, sample: list | None, as_df: bool = True):
    if sample is None:                                                                                                                                                                
        r = post(endpoint='/synthesize/neighbors', payload={'n': num_points})                                                                                                         
    else:
        r = post(endpoint='/synthesize/blended', payload={'n': num_points, 'instance': [col.item() for col in sample]})                                                               
    if r.status_code == 200:                                                                                                                                                          
        payload = r.json()                                                                                                                                                            
        if as_df:                                                                                                                                                                     
            df = pd.DataFrame(payload['samples'], columns=payload['feature_names'])
            for col, ftype in zip(payload['feature_names'], payload['feature_types']):                                                                                                
                df[col] = _cast_column(df[col], ftype)
            return df                                                                                                                                                                 
        return payload
    logger.error(
        "Data synthesis failed with status %s: %s",
        r.status_code,
        r.text
    )
    return None 


def train_surrogate(
        samples: list, 
        predictions: list, 
        feature_names: list[str] | None,
        task: str = 'auto', 
        test_size: float = 0.2
    ):
    r = post(endpoint='/surrogate/train', payload={'samples': samples, 'predictions': predictions, 'feature_names': feature_names, 'task': task, 'test_size': test_size})
    if r.status_code == 200:
        return r.json()
    logger.error(
        "Surrogate training failed with status %s: %s",
        r.status_code,
        r.text
    )
    return None 


def predict(samples: list, version: int | None):  # Defaults to latest version if not specified
    payload = {'inputs': samples}
    if version:  # Also rejects version 0 (versions start at 1)
        payload['version'] = version 
    r = post(endpoint='/surrogate/predict', payload=payload)
    if r.status_code == 200:
        return r.json()
    logger.error(
        "Surrogate prediction failed with status %s: %s",
        r.status_code,
        r.text
    )
    return None 


def find_counterfactual(sample, target, n_neighbors: int = 10000, perturbation_scale: float = 0.1, version: int | None, as_df: bool = True):
    payload = {
        'instance': sample,
        'target_label': target,
        'n_neighbors': n_neighbors,
        'perturbation_scale': perturbation_scale,
    }
    if version:  # Also rejects version 0 (versions start at 1)
        payload['version'] = version 
    r = post(endpoint='/explain/counterfactual', payload=payload)
    if r.status_code == 200:
        payload = r.json()                                                                                                                                                            
        if as_df:
            if payload['counterfactual'] is None:
                print(f"No counterfactual found: {payload.get('warning')}")
                return None
            df = pd.DataFrame([payload['counterfactual']], columns=payload['feature_names'])
            for col, ftype in zip(payload['feature_names'], payload['feature_types']):
                df[col] = _cast_column(df[col], ftype)
            return df
        return payload
    logger.error(
        "Counterfactual failed with status %s: %s",
        r.status_code,
        r.text
    )
    return None 


def interpret_counterfactual(
    sample: dict,
    counterfactual: dict,
    prediction_changed: bool,
    exclude_from_diff: list[str] | None
) -> str:
    if not exclude_from_diff:
        exclude_from_diff = list()
    diffs = {
        k: (sample[k], counterfactual[k])
        for k in sample
        if sample[k] != counterfactual[k]
        and k not in exclude_from_diff
    }
    
    if not diffs:
        return "No meaningful differences found between sample and counterfactual."
    
    changes = ", ".join(
        f"{feature} from {original} to {cf_value}"
        for feature, (original, cf_value) in diffs.items()
    )
    
    if prediction_changed:
        return (
            f"Changing {changes} may result in a different prediction. "
            f"Note: this is based on a surrogate model and should be treated as approximate."
        )
    else:
        return (
            f"A counterfactual was found suggesting {changes}, however "
            f"the original model's prediction did not change. "
            f"This may indicate the surrogate model does not fully capture "
            f"the original model's decision boundary in this region."
        )
