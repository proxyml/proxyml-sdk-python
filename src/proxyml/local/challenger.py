"""Train a challenger model locally, with no round-trip to the API.

Linear-only by design (LogisticRegressionCV / RidgeCV, matching the server's
default surrogate): a challenger's complexity ladder varies regularization
strength rather than model family, so results stay explainable by the same
closed-form coefficient math the server uses, and comparable to a
server-trained surrogate via the exact same export contract — whether the
training target was real ground truth or a black box's predictions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from enum import Enum
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from proxyml.schema_builder import get_schema
from proxyml_core.export import SurrogateExport
from proxyml_core.modeling.estimators import (
    binarize_if_probabilities,
    extract_hyperparameters,
    get_default_classifier,
    get_default_regressor,
    is_classification,
)
from proxyml_core.modeling.extract import extract_export_data
from proxyml_core.modeling.preprocess import build_preprocessor
from proxyml_core.modeling.scoring import score_predictions
from proxyml_core.schema import Feature, FeatureSchema


class Complexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    FLEXIBLE = "flexible"


@dataclass(kw_only=True)
class Rung:
    complexity: Complexity
    build_classifier: Callable[[], BaseEstimator]
    build_regressor: Callable[[], BaseEstimator]
    description: str


def _simple_classifier() -> BaseEstimator:
    return LogisticRegressionCV(
        Cs=np.logspace(-2, 0, 10),
        l1_ratios=(0,),
        solver="lbfgs",
        class_weight="balanced",
        max_iter=500,
        cv=5,
        n_jobs=-1,
        scoring="accuracy",
        use_legacy_attributes=False,
    )


def _simple_regressor() -> BaseEstimator:
    return RidgeCV(alphas=np.logspace(0, 4, 10), cv=5)


def _flexible_classifier() -> BaseEstimator:
    return LogisticRegressionCV(
        Cs=np.logspace(-4, 4, 25),
        l1_ratios=(0,),
        solver="lbfgs",
        class_weight="balanced",
        max_iter=1000,
        cv=5,
        n_jobs=-1,
        scoring="accuracy",
        use_legacy_attributes=False,
    )


def _flexible_regressor() -> BaseEstimator:
    return RidgeCV(alphas=np.logspace(-4, 4, 25), cv=5)


LADDERS: dict[Complexity, Rung] = {
    Complexity.SIMPLE: Rung(
        complexity=Complexity.SIMPLE,
        build_classifier=_simple_classifier,
        build_regressor=_simple_regressor,
        description="Strong regularization — biased toward fewer effectively-nonzero coefficients.",
    ),
    Complexity.MODERATE: Rung(
        complexity=Complexity.MODERATE,
        build_classifier=get_default_classifier,
        build_regressor=get_default_regressor,
        description="Matches the server's default surrogate — the baseline rung.",
    ),
    Complexity.FLEXIBLE: Rung(
        complexity=Complexity.FLEXIBLE,
        build_classifier=_flexible_classifier,
        build_regressor=_flexible_regressor,
        description="Wider regularization search grid for a closer per-sample fit.",
    ),
}


@dataclass(kw_only=True)
class TrainedChallenger:
    pipeline: Pipeline
    task: Literal["classification", "regression"]
    complexity: Complexity
    metrics: dict[str, float]
    hyperparameters: dict[str, Any]
    export: SurrogateExport
    n_samples_total: int
    n_samples_dropped_unlabeled: int
    population_note: str
    target_fingerprint: str
    champion_metrics: dict[str, float] | None = None


def _fingerprint_values(values: np.ndarray | list) -> str:
    """Hash an array of labels, deterministically, without the data ever leaving this process.

    ``.tolist()`` converts numpy scalars to native Python types before
    serializing, so the hash doesn't drift across numpy versions with
    different scalar repr behavior. Order is preserved (not sorted) since
    it encodes row alignment — that's exactly what a "same data?" check
    needs to be sensitive to.
    """
    canonical = json.dumps(np.asarray(values).tolist())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _population_note(target_name: str, n_total: int, n_labeled: int, n_dropped: int) -> str:
    if n_dropped == 0:
        return f"Evaluated on all {n_total} row(s) — '{target_name}' had no missing values."
    return (
        f"Evaluated on {n_labeled} of {n_total} row(s) with a non-null '{target_name}' value "
        f"({n_dropped} unlabeled row(s) dropped before training/scoring). "
        "Labeled-vs-unlabeled selection may not be random; treat this as a declared "
        "scope limitation on the evaluation population, not a claim about performance "
        "on the full dataset."
    )


def train_challenger(
    df: pd.DataFrame,
    target: np.ndarray | list,
    schema: FeatureSchema,
    *,
    complexity: Complexity = Complexity.MODERATE,
    feature_names: list[str] | None = None,
    task: Literal["classification", "regression", "auto"] = "auto",
    test_size: float = 0.2,
    target_name: str = "target",
    champion_predictions: np.ndarray | list | None = None,
) -> TrainedChallenger:
    """Train a linear challenger model on ``df`` against ``target``, locally.

    ``target`` can be either real ground-truth labels (training a genuine
    challenger to compare against a champion model on real outcomes) or a
    black box's predictions (training a surrogate/explainer of that model) —
    the fit itself doesn't care which. No round-trip to the API — everything
    happens in-process via ``proxyml_core.modeling``. The result's ``export``
    is a ``SurrogateExport``, structurally identical to what
    ``export_surrogate()`` returns for a server-trained surrogate, so the two
    can be compared with the same ``proxyml_core.export.predict_from_export``
    arithmetic.

    Rows where ``target`` is missing (NaN/None) are dropped before training,
    the CV split, and champion scoring — never silently included. The drop
    count and a human-readable scope-limitation note are recorded on the
    result (``n_samples_total``, ``n_samples_dropped_unlabeled``,
    ``population_note``). If ``champion_predictions`` is given, it must have
    one entry per row of ``df``/``target`` (same order) so the identical rows
    are dropped from both sides — champion and challenger are always
    evaluated on the same labeled population, never on different ones.

    Args:
        df: samples to train on, one column per schema feature.
        target: the value to predict for each row of ``df`` — ground-truth
            labels or a black box's output, in the same order as ``df``.
        schema: FeatureSchema describing ``df``'s columns (e.g. from ``get_schema``).
        complexity: which rung of ``LADDERS`` to train at.
        feature_names: subset of ``schema.features`` to train on; omit for all.
        task: "classification", "regression", or "auto" to infer from ``target``.
        test_size: fraction of data held out to compute fidelity metrics.
        target_name: human-readable name for ``target``, used in
            ``population_note`` (e.g. the column name, if known).
        champion_predictions: a champion model's predictions, one per row of
            ``df``/``target`` (same order). If given, scored via
            ``score_champion()`` against the same (row-dropped) ``target``,
            and the result is attached as ``TrainedChallenger.champion_metrics``.
    """
    target_arr = np.asarray(target)
    target_fingerprint = _fingerprint_values(target_arr)
    if champion_predictions is not None and len(champion_predictions) != len(target_arr):
        raise ValueError(
            f"champion_predictions must have one entry per row of target "
            f"({len(target_arr)} rows, got {len(champion_predictions)}) — same order — so "
            f"rows with a missing {target_name!r} value can be dropped from both the "
            f"challenger and the champion, keeping them evaluated on the same population."
        )

    labeled_mask = ~pd.isna(target_arr)
    n_total = len(target_arr)
    n_labeled = int(labeled_mask.sum())
    n_dropped = n_total - n_labeled
    if n_labeled == 0:
        raise ValueError(f"All {n_total} row(s) have a missing {target_name!r} value; nothing to train on")

    df = df.iloc[labeled_mask].reset_index(drop=True)
    target_arr = target_arr[labeled_mask]
    champion_predictions_labeled = (
        np.asarray(champion_predictions)[labeled_mask] if champion_predictions is not None else None
    )

    rung = LADDERS[complexity]

    features: list[Feature] = schema.features
    if feature_names is not None:
        name_to_feature = {f.name: f for f in features}
        features = [name_to_feature[n] for n in feature_names]
    col_order = [f.name for f in features]

    X = df[col_order].to_numpy(dtype=object)
    y = target_arr

    if task == "auto":
        classification = is_classification(y)
    else:
        classification = task == "classification"
    resolved_task: Literal["classification", "regression"] = (
        "classification" if classification else "regression"
    )

    if classification:
        y = binarize_if_probabilities(y)

    champion_metrics = None
    if champion_predictions_labeled is not None:
        champion_metrics = score_champion(y, champion_predictions_labeled, task=resolved_task)

    preprocessor = build_preprocessor(features)
    estimator = rung.build_classifier() if classification else rung.build_regressor()
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y if classification else None,
        )
    except ValueError:
        # stratify fails when a class has too few samples to split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    pipeline.fit(X_train, y_train)
    hyperparameters = extract_hyperparameters(pipeline.named_steps["estimator"])
    y_pred = pipeline.predict(X_test)

    metrics = score_predictions(y_test, y_pred, task=resolved_task)

    export = extract_export_data(pipeline, features, resolved_task)
    export = replace(export, hyperparameters=hyperparameters, metrics=metrics)

    return TrainedChallenger(
        pipeline=pipeline,
        task=resolved_task,
        complexity=complexity,
        metrics=metrics,
        hyperparameters=hyperparameters,
        export=export,
        n_samples_total=n_total,
        n_samples_dropped_unlabeled=n_dropped,
        population_note=_population_note(target_name, n_total, n_labeled, n_dropped),
        target_fingerprint=target_fingerprint,
        champion_metrics=champion_metrics,
    )


def score_champion(
    labels: np.ndarray | list,
    predictions: np.ndarray | list,
    *,
    task: Literal["classification", "regression"],
) -> dict[str, float]:
    """Score a champion model's predictions against real labels, locally.

    Uses the exact same scoring code as ``train_challenger()``'s internal
    fidelity metrics, so ``champion_metrics`` and a paired
    ``TrainedChallenger.metrics`` are computed identically — required for an
    apples-to-apples champion-vs-challenger comparison. Pass the same task
    the paired challenger resolved to (e.g. ``result.task``); there's no
    ``task="auto"`` here, since letting the two resolve independently risks
    them silently diverging.

    Returns ``{"f1":..., "accuracy":...}`` for classification or ``{"r2":...}``
    for regression — the same shape as ``TrainedChallenger.metrics``.

    If you're calling this decoupled from ``train_challenger()`` (i.e. not
    via its ``champion_predictions=`` param), pass the same ``labels`` you
    used here as ``champion_labels=`` to ``to_challenger_upload()`` — that
    lets the upload endpoint confirm the challenger and champion were
    actually scored on the same data, catching an accidental mismatched
    file before it silently produces a misleading comparison.
    """
    return score_predictions(np.asarray(labels), np.asarray(predictions), task=task)


def to_challenger_upload(
    result: TrainedChallenger,
    *,
    n_samples: int | None = None,
    champion_metrics: dict[str, float] | None = None,
    champion_labels: np.ndarray | list | None = None,
    sdk_version: str | None = None,
    proxyml_core_version: str | None = None,
) -> dict[str, Any]:
    """Assemble the JSON-serializable payload for a challenger upload.

    Matches the shape ProxyML's dashboard/API expects at
    ``POST /app/projects/{id}/challenger`` — handles the mechanical assembly
    (serializing the export, stamping SDK/core versions, converting
    ``complexity`` to a plain string) so callers don't have to hand-roll it.
    The result is plain ``dict``/``str``/``float`` data, ready for
    ``json.dump`` — upload it either by POSTing it directly, or by saving it
    to a file and using the dashboard's "Upload challenger" button.

    ``champion_metrics`` is optional: pass ``None`` (the default) to fall
    back to ``result.champion_metrics`` (populated automatically if you
    passed ``champion_predictions`` to ``train_challenger()``/
    ``train_auto_challenger()``) — or, if that's also ``None``, to get a
    self-contained export of the challenger alone, e.g. to save/share it
    before you have a champion to compare against. The upload endpoint
    itself still requires ``champion_metrics`` at upload time; this function
    just doesn't force you to have it up front.

    Args:
        result: output of ``train_challenger()``/``train_auto_challenger()``.
        n_samples: size of the evaluation set both ``result.metrics`` and
            ``champion_metrics`` were scored on. Defaults to
            ``result.n_samples_total - result.n_samples_dropped_unlabeled``
            (the labeled-row count) — override only if you scored on some
            other population.
        champion_metrics: the champion's real-world performance, from
            ``score_champion()`` — same metric keys as ``result.metrics``.
            Defaults to ``result.champion_metrics``.
        champion_labels: the ``labels`` array you passed to a standalone
            ``score_champion()`` call, if ``champion_metrics`` didn't come
            from ``train_challenger()``'s internal ``champion_predictions=``
            path. Used only to compute ``champion_data_fingerprint`` — the
            labels themselves are never included in the payload. If
            ``champion_metrics`` resolves from ``result.champion_metrics``
            instead, the fingerprint defaults to ``result.target_fingerprint``
            (guaranteed identical, since that internal path scores against
            the exact same data).
        sdk_version: defaults to the installed ``proxyml`` version.
        proxyml_core_version: defaults to the installed ``proxyml-core`` version.
    """
    if n_samples is None:
        n_samples = result.n_samples_total - result.n_samples_dropped_unlabeled
    used_internal_champion_metrics = champion_metrics is None
    if champion_metrics is None:
        champion_metrics = result.champion_metrics
    if sdk_version is None:
        sdk_version = _pkg_version("proxyml")
    if proxyml_core_version is None:
        proxyml_core_version = _pkg_version("proxyml-core")

    payload: dict[str, Any] = {
        "export": result.export.to_dict(),
        "challenger_metrics": result.metrics,
        "n_samples": n_samples,
        "n_samples_total": result.n_samples_total,
        "n_samples_dropped_unlabeled": result.n_samples_dropped_unlabeled,
        "population_note": result.population_note,
        "complexity": result.complexity.value,
        "sdk_version": sdk_version,
        "proxyml_core_version": proxyml_core_version,
    }
    if champion_metrics is not None:
        payload["champion_metrics"] = champion_metrics
        payload["challenger_data_fingerprint"] = result.target_fingerprint
        if champion_labels is not None:
            payload["champion_data_fingerprint"] = _fingerprint_values(champion_labels)
        elif used_internal_champion_metrics:
            payload["champion_data_fingerprint"] = result.target_fingerprint
    return payload


def train_auto_challenger(
    data: str | Path | pd.DataFrame,
    target_col: str,
    *,
    immutable_cols: list[str] | None = None,
    complexity: Complexity = Complexity.MODERATE,
    feature_names: list[str] | None = None,
    task: Literal["classification", "regression", "auto"] = "auto",
    test_size: float = 0.2,
    champion_predictions: np.ndarray | list | None = None,
) -> TrainedChallenger:
    """Load data, infer a schema, and train a linear challenger in one call.

    Convenience wrapper around ``get_schema()`` + ``train_challenger()`` — it
    only automates schema inference and the feature/target column split,
    nothing more. ``complexity`` still defaults to ``Complexity.MODERATE`` and
    remains overridable; this does not search across ``LADDERS`` to find the
    best-fitting rung.

    Rows with a missing ``target_col`` value are dropped before training and
    champion scoring — see ``train_challenger()`` for details. Schema
    inference (feature means/stds/categories) still runs over every row,
    including ones later dropped for a missing target — only training and
    evaluation are restricted to the labeled subset.

    Args:
        data: a CSV path, or an already-loaded DataFrame containing both the
            feature columns and ``target_col``.
        target_col: name of the column to train against — either real
            ground-truth labels or a black box's predictions.
        immutable_cols: passed through to ``get_schema()``.
        complexity: which rung of ``LADDERS`` to train at.
        feature_names: subset of feature columns to train on; omit for all.
        task: "classification", "regression", or "auto" to infer from ``target_col``.
        test_size: fraction of data held out to compute fidelity metrics.
        champion_predictions: a champion model's predictions, one per row of
            ``data`` (same order) — see ``train_challenger()``.
    """
    df = data if isinstance(data, pd.DataFrame) else pd.read_csv(data)
    target = df[target_col]
    features_df = df.drop(columns=[target_col])

    schema = get_schema(features_df, immutable_cols=immutable_cols)
    return train_challenger(
        features_df,
        target,
        schema,
        complexity=complexity,
        feature_names=feature_names,
        task=task,
        test_size=test_size,
        target_name=target_col,
        champion_predictions=champion_predictions,
    )
