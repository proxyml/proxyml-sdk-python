"""Train a challenger model locally, with no round-trip to the API.

Linear-only by design (LogisticRegressionCV / RidgeCV, matching the server's
default surrogate): a challenger's complexity ladder varies regularization
strength rather than model family, so results stay explainable by the same
closed-form coefficient math the server uses, and comparable to a
server-trained surrogate via the exact same export contract — whether the
training target was real ground truth or a black box's predictions.
"""

from __future__ import annotations

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


def train_challenger(
    df: pd.DataFrame,
    target: np.ndarray | list,
    schema: FeatureSchema,
    *,
    complexity: Complexity = Complexity.MODERATE,
    feature_names: list[str] | None = None,
    task: Literal["classification", "regression", "auto"] = "auto",
    test_size: float = 0.2,
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

    Args:
        df: samples to train on, one column per schema feature.
        target: the value to predict for each row of ``df`` — ground-truth
            labels or a black box's output, in the same order as ``df``.
        schema: FeatureSchema describing ``df``'s columns (e.g. from ``get_schema``).
        complexity: which rung of ``LADDERS`` to train at.
        feature_names: subset of ``schema.features`` to train on; omit for all.
        task: "classification", "regression", or "auto" to infer from ``target``.
        test_size: fraction of data held out to compute fidelity metrics.
    """
    rung = LADDERS[complexity]

    features: list[Feature] = schema.features
    if feature_names is not None:
        name_to_feature = {f.name: f for f in features}
        features = [name_to_feature[n] for n in feature_names]
    col_order = [f.name for f in features]

    X = df[col_order].to_numpy(dtype=object)
    y = np.asarray(target)

    if task == "auto":
        classification = is_classification(y)
    else:
        classification = task == "classification"
    resolved_task: Literal["classification", "regression"] = (
        "classification" if classification else "regression"
    )

    if classification:
        y = binarize_if_probabilities(y)

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
    """
    return score_predictions(np.asarray(labels), np.asarray(predictions), task=task)


def to_challenger_upload(
    result: TrainedChallenger,
    *,
    n_samples: int,
    champion_metrics: dict[str, float] | None = None,
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

    ``champion_metrics`` is optional: pass ``None`` (the default) to get a
    self-contained export of the challenger alone — e.g. to save/share it
    before you have a champion to compare against — and fill in
    ``champion_metrics`` later. The upload endpoint itself still requires
    ``champion_metrics`` at upload time; this function just doesn't force you
    to have it up front.

    Args:
        result: output of ``train_challenger()``/``train_auto_challenger()``.
        n_samples: size of the evaluation set both ``result.metrics`` and
            ``champion_metrics`` were scored on. Not derived automatically —
            ``TrainedChallenger`` doesn't retain its internal held-out split,
            and ``champion_metrics`` typically comes from a separate
            ``score_champion()`` call the two need to share, so the caller is
            the only one who actually knows this number.
        champion_metrics: the champion's real-world performance, from
            ``score_champion()`` — same metric keys as ``result.metrics``.
            Omit if you don't have it yet.
        sdk_version: defaults to the installed ``proxyml`` version.
        proxyml_core_version: defaults to the installed ``proxyml-core`` version.
    """
    if sdk_version is None:
        sdk_version = _pkg_version("proxyml")
    if proxyml_core_version is None:
        proxyml_core_version = _pkg_version("proxyml-core")

    payload: dict[str, Any] = {
        "export": result.export.to_dict(),
        "challenger_metrics": result.metrics,
        "n_samples": n_samples,
        "complexity": result.complexity.value,
        "sdk_version": sdk_version,
        "proxyml_core_version": proxyml_core_version,
    }
    if champion_metrics is not None:
        payload["champion_metrics"] = champion_metrics
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
) -> TrainedChallenger:
    """Load data, infer a schema, and train a linear challenger in one call.

    Convenience wrapper around ``get_schema()`` + ``train_challenger()`` — it
    only automates schema inference and the feature/target column split,
    nothing more. ``complexity`` still defaults to ``Complexity.MODERATE`` and
    remains overridable; this does not search across ``LADDERS`` to find the
    best-fitting rung.

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
    )
