import logging
logger = logging.getLogger(__name__)
from typing import Any

import requests
import os
import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype


PROXYML_BASE_URL = os.getenv("PROXYML_BASE_URL", "https://api.proxyml.ai/api/v1")

_BOOL_STRINGS = {"true", "false"}


def _headers() -> dict:
    """
    Constructs the request headers required for making calls to the ProxyML API.
    """
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
    """
    POSTs a request to a ProxyML API endpoint.

    Args:
        endpoint (str): ProxyML API endpoint.  PROXYML_BASE_URL is prepended e.g. use endpoint='/schemas' not 'https://api.proxyml.ai/api/v1/schemas'.
        payload (dict): JSON payload to POST.

    Returns:
        requests Response object.
    """
    r = requests.post(
        url=f'{PROXYML_BASE_URL}{endpoint}',
        json=payload,
        headers=_headers()
    )
    return r


def put(endpoint: str, payload: dict) -> requests.models.Response:
    """
    PUTs a request to a ProxyML API endpoint.

    Args:
        endpoint (str): ProxyML API endpoint.  PROXYML_BASE_URL is prepended e.g. use endpoint='/schemas' not 'https://api.proxyml.ai/api/v1/schemas'.
        payload (dict): JSON payload to POST.

    Returns:
        requests Response object.
    """    
    r = requests.put(
        url=f'{PROXYML_BASE_URL}{endpoint}',
        json=payload,
        headers=_headers()
    )
    return r


def get(endpoint: str, params: dict) -> requests.models.Response:
    """
    GETs a request to a ProxyML API endpoint.

    Args:
        endpoint (str): ProxyML API endpoint.  PROXYML_BASE_URL is prepended e.g. use endpoint='/schemas' not 'https://api.proxyml.ai/api/v1/schemas'.
        params (dict): JSON payload to POST.

    Returns:
        requests Response object.
    """
    r = requests.get(
        url=f'{PROXYML_BASE_URL}{endpoint}',
        headers=_headers(),
        params=params
    )
    return r


def delete(endpoint: str) -> requests.models.Response:
    """
    DELETE request to a ProxyML API endpoint.

    Args:
        endpoint (str): ProxyML API endpoint.  PROXYML_BASE_URL is prepended e.g. use endpoint='/schemas' not 'https://api.proxyml.ai/api/v1/schemas'.

    Returns:
        requests Response object.
    """
    r = requests.delete(
        url=f'{PROXYML_BASE_URL}{endpoint}',
        headers=_headers()
    )
    return r


def put_schema(schema: dict, name: str = "default") -> Any:
    """
    Uploads a data schema.

    Args:
        schema (dict): data schema object
        name (str): optional name for the schema, defaults to "default".
    Returns:
        API response JSON object, or None if the return code was not 200.
    """
    r = put(endpoint=f'/schema/{name}', payload=schema)
    if r.status_code == 200:
        logger.info("Schema '%s' uploaded successfully", name)
        return r.json()
    logger.error(
        "Schema upload failed with status %s: %s",
        r.status_code,
        r.text
    )
    return None


def fetch_schema(name: str = "default") -> dict | None:
    """Retrieve a stored feature schema by name."""
    r = get(endpoint=f'/schema/{name}', params={})
    if r.status_code == 200:
        return r.json()
    logger.error(
        "Fetch schema failed with status %s: %s",
        r.status_code,
        r.text,
    )
    return None


def list_schemas() -> list[dict] | None:
    """Return all named schemas for the authenticated user."""
    r = get(endpoint='/schemas', params={})
    if r.status_code == 200:
        return r.json()['schemas']
    logger.error(
        "List schemas failed with status %s: %s",
        r.status_code,
        r.text,
    )
    return None


def delete_schema(name: str) -> bool:
    """Delete a named schema. Returns True on success, False if not found."""
    r = delete(endpoint=f'/schema/{name}')
    if r.status_code == 204:
        return True
    if r.status_code == 404:
        logger.warning("Schema '%s' not found", name)
        return False
    logger.error(
        "Delete schema failed with status %s: %s",
        r.status_code,
        r.text,
    )
    return False


def _cast_column(series: pd.Series, ftype: str) -> pd.Series:
    """
    Casts a pandas Series to a Python type based on a specified ftype.

    Args:
        series (pd.Series): data to cast
        ftype (str): type of conversion, one of "continuous", "count", "numeric_ordinal", "categorical", or "categorical_ordinal".
    Returns:
        pd.Series cast to specific ftype.
    """
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


def synthesize_data(num_points: int = 100, sample: list | None = None, as_df: bool = True, schema_name: str = "default") -> Any:
    """
    Synthesizes data based on a data schema.

    Args:
        num_points (int): number of samples to synthesize, defaults to 100.
        sample (list): if specified, synthesized data will be a blend of samples generated from the schema and a number of perturbations around this sample. If not specified (default), all synthesized data is generated by sampling from the data schema.
        as_df (bool): if True (default), data are returned as a pandas DataFrame.
        schema_name (str): name of the data schema to use to generate the data, defaults to "default."
    Returns:
        JSON object representing the synthesized data if as_df=False, a pandas DataFrame if as_df=True, or None if an error occurred.
    """
    if sample is None:
        r = post(endpoint='/synthesize/neighbors', payload={'n': num_points, 'schema_name': schema_name})
    else:
        r = post(endpoint='/synthesize/blended', payload={'n': num_points, 'instance': [col.item() for col in sample], 'schema_name': schema_name})
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
        schema_name: str = "default",
        name: str | None = None,
        comments: str | None = None,
    ) -> Any:
    """
    Trains a surrogate model to predict a "black box" model's predictions.

    Args:
        samples (list): list of samples that were provided to the black box for inference, e.g. the output from synthesize_data().
        predictions (list): the inferences on the samples, made by the black box model.
        feature_names (list): names of the features (columns) in the data.
        task (str): specifies the modeling task, one of "classification," "regression," or "auto" in which case ProxyML will attempt to automatically determine the modeling task.
        test_size (float): fraction of the data to set aside for test data, defaults to 0.2.
        schema_name (str): name of the data schema to use, defaults to "default."
        name (str): an optional name for the surrogate.
        comments (str): an optional comment string for the surrogate.
    Returns:
        JSON object denoting surrogate training result, or None if an error occurred.
    """
    payload = {
        'samples': samples,
        'predictions': predictions,
        'feature_names': feature_names,
        'task': task,
        'test_size': test_size,
        'schema_name': schema_name,
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


def predict(sample: list, version: str | None = None) -> Any:
    """
    Calls a surrogate model to make a single prediction.
    
    Args:
        sample (list): a single sample
        version (str): name of the surrogate model to use.
    Returns:
        JSON object denoting result of the surrogate prediction, or None if an error occurred.
    """
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


def find_counterfactual(sample, target, n_neighbors: int = 10000, perturbation_scale: float = 0.1, version: str | None = None, as_df: bool = True) -> Any:
    """
    Attempts to find a counterfactual sample: a sample that is near to a given sample in featurespace, but with a specified prediction. Counterfactuals are
    not guaranteed i.e., it's possible to not be able to find a nearby sample with the desired label / target value.

    Args:
        sample (list): point around which neighbors will be found.
        target (Any): the desired prediction outcome, can be a string or number for classification or a target predicted value for regression.
        n_neighbors (int): the number of neighbors to search, defaults to 10000.
        pertubation_scale (float): controls the "wiggle" in the features around the provided sample. Defaults to 0.1.
        version (str): name of the surrogate model to run, defaults to "default."
        as_df (bool): if True, return the counterfactual as a dataframe. If False, returns JSON object.
    Returns:
        JSON counterfactual if as_df=False, pandas DataFrame if as_df=True, None if an error occurred.
    """
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
    """
    Simple string interpretation of a counterfactual result. No API calls are required.

    Args:
        sample (dict): the original prediction i.e. the sample from which counterfactuals were drawn.
        counterfactual (dict): the counterfactual returned by ProxyML.
        prediction_changed (bool): True if the prediction changed, False if it did not (or didn't change as much as desired).
        exclude_from_diff (list): list of keys that should not be considered in interpreting the counterfactual.
    Returns:
        String summarizing the differences (if any) observed between sample and counterfactual.
    """
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


def predict_batch(samples: list[list], version: str | None = None) -> Any:
    """
    Calls a surrogate model to make batch predictions.
    
    Args:
        sample (list): a list of samples
        version (str): name of the surrogate model to use.
    Returns:
        JSON object denoting result of the surrogate prediction, or None if an error occurred.
    """
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
) -> Any:
    """
    Attempts to find counterfactual samples: samples that are near a given sample in featurespace, but with a specified prediction. Counterfactuals are
    not guaranteed i.e., it's possible to not be able to find a nearby sample with the desired label / target value.

    Args:
        samples (list): list of points around which neighbors will be found.
        target (Any): the desired prediction outcome, can be a string or number for classification or a target predicted value for regression.
        n_neighbors (int): the number of neighbors to search, defaults to 10000.
        pertubation_scale (float): controls the "wiggle" in the features around the provided sample. Defaults to 0.1.
        version (str): name of the surrogate model to run, defaults to "default."
        as_df (bool): if True, return the counterfactual as a dataframe. If False, returns JSON object.
    Returns:
        JSON counterfactuals if as_df=False, pandas DataFrame if as_df=True, None if an error occurred.
    """    
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


def get_model_summary(version: str | None = None) -> Any:
    """
    Retrieves a summary of a given surrogate.

    Args:
        version (str): name of the surrogate model to retrieve.
    Returns:
        JSON object summarizing the surrogate, or None if an error occurred.
    """
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


def get_model_schema(version: str) -> Any:
    """
    Retrieves the schema associated with a particular surrogate.
    
    Args:
        version (str): name of the surrogate model.
    Returns:
        JSON object representation of the model's schema, or None if an error occurred.
    """
    r = get(endpoint=f'/surrogate/models/{version}/schema', params=dict())
    if r.status_code == 200:
        return r.json()
    logger.error(
        "Schema retrieval failed with status %s: %s",
        r.status_code,
        r.text,
    )
    return None


def diff_models(version_a: str, version_b: str) -> Any:
    """
    Returns a summary of the differences between two surrogates.  The surrogates must have the same model task (i.e. a classification model can't be compared with a regression model), and they must
    have at least one feature in common.

    Args:
        version_a (str): name of one surrogate
        version_b (str): name of the other surrogate
    Returns:
        JSON object summarizing the difference between version_a and version_b, or None if an error occurred.
    """
    r = get(endpoint='/explain/diff', params={'version_a': version_a, 'version_b': version_b})
    if r.status_code == 200:
        return r.json()
    logger.error(
        "Model diff failed with status %s: %s",
        r.status_code,
        r.text,
    )
    return None


def get_feature_importances(version: str | None = None) -> Any:
    """
    Convenience function to directly retrieve feature importances from a surrogate model.

    Args:
        version (str): ID of the surrogate
    Returns:
        JSON object of feature importances, or None if an error occurred.
    """
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


def get_usage() -> dict | None:
    """Return current tier, usage counts, and quota for the authenticated user."""
    r = get(endpoint='/account/usage', params={})
    if r.status_code == 200:
        return r.json()
    logger.error(
        "Get usage failed with status %s: %s",
        r.status_code,
        r.text,
    )
    return None


def rotate_key() -> str | None:
    """Rotate the API key, revoking all old keys. Returns the new key string or None on failure."""
    r = post(endpoint='/account/keys/rotate', payload={})
    if r.status_code == 201:
        return r.json()['api_key']
    logger.error(
        "Key rotation failed with status %s: %s",
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


def explain_local(instance: list, version: str | None = None) -> dict | None:
    """Per-feature contribution breakdown for a single instance.

    Returns a dict with ``prediction``, ``feature_contributions`` (sorted by
    abs_contribution descending), ``intercept``, optional ``probabilities``
    (classification), and optional ``per_class_contributions`` (multiclass).
    """
    payload: dict = {"instance": instance}
    if version is not None:
        payload["version"] = version
    r = post(endpoint="/explain/local", payload=payload)
    if r.status_code == 200:
        return r.json()
    logger.error(
        "Local explanation failed with status %s: %s",
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
