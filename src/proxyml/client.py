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


def get(endpoint: str, params: dict) -> requests.models.Response:
    r = requests.get(
        url=f'{PROXYML_BASE_URL}{endpoint}',
        headers=_headers(),
        params=params
    )
    return r


def delete(endpoint: str) -> requests.models.Response:
    r = requests.delete(
        url=f'{PROXYML_BASE_URL}{endpoint}',
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


def synthesize_data(num_points: int = 100, sample: list | None = None, as_df: bool = True):
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
        test_size: float = 0.2,
        name: str | None = None,
        comments: str | None = None,
    ):
    payload = {
        'samples': samples,
        'predictions': predictions,
        'feature_names': feature_names,
        'task': task,
        'test_size': test_size,
    }
    if name is not None:
        payload['name'] = name
    if comments is not None:
        payload['comments'] = comments
    r = post(endpoint='/surrogate/train', payload=payload)
    if r.status_code == 200:
        return r.json()
    logger.error(
        "Surrogate training failed with status %s: %s",
        r.status_code,
        r.text
    )
    return None


def predict(sample: list, version: str | None = None):
    payload = {'inputs': sample}
    if version is not None:
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


def find_counterfactual(sample, target, n_neighbors: int = 10000, perturbation_scale: float = 0.1, version: str | None = None, as_df: bool = True):
    payload = {
        'instance': sample,
        'target_label': target.item() if hasattr(target, 'item') else target,
        'n_neighbors': n_neighbors,
        'perturbation_scale': perturbation_scale,
    }
    if version is not None:
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


def predict_batch(samples: list[list], version: str | None = None):
    payload = {'inputs': samples}
    if version is not None:
        payload['version'] = version
    r = post(endpoint='/surrogate/predict/batch', payload=payload)
    if r.status_code == 200:
        return r.json()
    logger.error(
        "Batch prediction failed with status %s: %s",
        r.status_code,
        r.text,
    )
    return None


def find_counterfactuals(
    samples: list[list],
    target,
    n_neighbors: int = 10000,
    perturbation_scale: float = 0.1,
    version: str | None = None,
    as_dfs: bool = True,
):
    payload = {
        'instances': samples,
        'target_label': target.item() if hasattr(target, 'item') else target,
        'n_neighbors': n_neighbors,
        'perturbation_scale': perturbation_scale,
    }
    if version is not None:
        payload['version'] = version
    r = post(endpoint='/explain/counterfactual/batch', payload=payload)
    if r.status_code == 200:
        data = r.json()
        if as_dfs:
            feature_names = data['feature_names']
            feature_types = data['feature_types']
            results = []
            for item in data['results']:
                if item['counterfactual'] is None:
                    if item.get('warning'):
                        print(f"No counterfactual found: {item['warning']}")
                    results.append(None)
                else:
                    df = pd.DataFrame([item['counterfactual']], columns=feature_names)
                    for col, ftype in zip(feature_names, feature_types):
                        df[col] = _cast_column(df[col], ftype)
                    results.append(df)
            return results
        return data
    logger.error(
        "Batch counterfactual failed with status %s: %s",
        r.status_code,
        r.text,
    )
    return None


def get_model_summary(version: str | None = None):
    params = {'version': version} if version is not None else {}
    r = get(endpoint='/explain/summary', params=params)
    if r.status_code == 200:
        return r.json()
    logger.error(
        "Model summary failed with status %s: %s",
        r.status_code,
        r.text,
    )
    return None


def diff_models(version_a: str, version_b: str):
    r = get(endpoint='/explain/diff', params={'version_a': version_a, 'version_b': version_b})
    if r.status_code == 200:
        return r.json()
    logger.error(
        "Model diff failed with status %s: %s",
        r.status_code,
        r.text,
    )
    return None


def get_feature_importances(version: str | None = None):
    params = {'version': version} if version is not None else {}
    r = get(endpoint='/explain/importance', params=params)
    if r.status_code == 200:
        return r.json()
    logger.error(
        "Feature importances failed with status %s: %s",
        r.status_code,
        r.text,
    )
    return None


def list_models() -> list[dict] | None:
    """Return metadata for all trained surrogate models, newest first."""
    r = get(endpoint='/surrogate/models', params={})
    if r.status_code == 200:
        return r.json()['models']
    logger.error(
        "List models failed with status %s: %s",
        r.status_code,
        r.text,
    )
    return None


def delete_model(model_id: str) -> bool:
    """Delete a surrogate model by its UUID. Returns True on success, False if not found."""
    r = delete(endpoint=f'/surrogate/models/{model_id}')
    if r.status_code == 204:
        return True
    if r.status_code == 404:
        logger.warning("Surrogate model %s not found", model_id)
        return False
    logger.error(
        "Delete model failed with status %s: %s",
        r.status_code,
        r.text,
    )
    return False
