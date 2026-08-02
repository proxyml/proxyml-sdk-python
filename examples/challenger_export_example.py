"""
challenger_export_example - trains a challenger locally and exports it as a
JSON payload ready to upload to the ProxyML dashboard's champion-vs-challenger
comparison.

Prerequisites:
    pip install 'proxyml[local]' scikit-learn pandas
"""
import json

import pandas as pd
from sklearn.datasets import load_breast_cancer

from proxyml.local import Complexity, to_challenger_upload, train_auto_challenger

# A real dataset with a real binary target - stands in for your own labeled data.
data = load_breast_cancer(as_frame=True)
df = data.frame.rename(columns={"target": "diagnosis"})

# champion_predictions should be YOUR actual production model's predictions,
# one per row of df (same order) - that's what makes the comparison
# meaningful. This example has no real champion handy, so it trains a
# differently-regularized challenger purely as a stand-in so the two metric
# sets aren't identical. Do not do this in real usage - swap in your actual
# champion's predictions.
champion_stand_in = train_auto_challenger(df, "diagnosis", task="classification", complexity=Complexity.SIMPLE)
champion_predictions = champion_stand_in.pipeline.predict(
    df.drop(columns=["diagnosis"]).to_numpy(dtype=object)
)

# Train the challenger you actually want to upload and compare against your
# champion. complexity controls the challenger's own regularization strength
# (SIMPLE/MODERATE/FLEXIBLE) - it has no bearing on the champion side. Any
# row with a missing "diagnosis" value is dropped before training, and the
# identical rows are dropped from champion_predictions too, so the champion
# and challenger are always scored on the same population - never pass
# champion_metrics computed on a different set of rows than the challenger.
challenger_result = train_auto_challenger(
    df, "diagnosis", task="classification", complexity=Complexity.MODERATE,
    champion_predictions=champion_predictions,
)
print("Challenger metrics:", challenger_result.metrics)
print("Champion metrics (stand-in):", challenger_result.champion_metrics)
if challenger_result.n_samples_dropped_unlabeled:
    print("Population note:", challenger_result.population_note)

# to_challenger_upload() assembles the JSON-serializable payload the
# dashboard/API expects - export serialization, SDK/core version stamping,
# n_samples/champion_metrics (auto-derived from challenger_result above),
# and complexity-as-a-string are all handled for you.
payload = to_challenger_upload(challenger_result)

with open("challenger.json", "w") as f:
    json.dump(payload, f, indent=2)
print("\nWrote challenger.json - upload it via the ProxyML dashboard's "
      "'Upload challenger' button on a challenger-comparison project, or "
      "POST it directly to /app/projects/{project_id}/challenger.")

# If you don't have champion_predictions yet (e.g. you want to hand this
# export to whoever owns the champion model, or just archive it), omit
# champion_predictions entirely - you get a self-contained export of the
# challenger alone, and can fill champion_metrics in later via
# to_challenger_upload(result, champion_metrics=...):
#
#   result = train_auto_challenger(df, "diagnosis", task="classification")
#   payload = to_challenger_upload(result)

# --- What if the champion was scored on different data? ---
# train_challenger()/train_auto_challenger() fingerprint the labels they're
# scored against (a SHA-256 hash, in TrainedChallenger.target_fingerprint).
# to_challenger_upload() compares that against the champion's own
# fingerprint and raises immediately if they don't match - catching a
# champion accidentally scored against a different train/test split, an
# extra dropna(), or the wrong file entirely, before the mismatched
# comparison is ever uploaded or shared. Simulated here by flipping every
# label, standing in for a champion scored on a subtly different population:
mismatched_champion_labels = ~df["diagnosis"].astype(bool).to_numpy()
try:
    to_challenger_upload(
        challenger_result,
        champion_metrics=challenger_result.champion_metrics,
        champion_labels=mismatched_champion_labels,
    )
except ValueError as e:
    print("\nCaught a mismatched champion/challenger comparison:", e)
