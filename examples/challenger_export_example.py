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

from proxyml.local import Complexity, score_champion, to_challenger_upload, train_auto_challenger

# A real dataset with a real binary target - stands in for your own labeled data.
data = load_breast_cancer(as_frame=True)
df = data.frame.rename(columns={"target": "diagnosis"})

# Train the challenger you actually want to upload and compare against your
# champion. complexity controls the challenger's own regularization strength
# (SIMPLE/MODERATE/FLEXIBLE) - it has no bearing on the champion side.
challenger_result = train_auto_challenger(df, "diagnosis", task="classification", complexity=Complexity.MODERATE)
print("Challenger metrics:", challenger_result.metrics)

# champion_metrics should come from scoring YOUR actual production model's
# predictions against real labels, via score_champion(labels, predictions,
# task=...) - that's what makes the comparison meaningful. This example has
# no real champion handy, so it trains a second, differently-regularized
# challenger purely as a stand-in so the two metric sets aren't identical.
# Do not do this in real usage - swap in your actual champion's predictions.
champion_stand_in = train_auto_challenger(df, "diagnosis", task="classification", complexity=Complexity.SIMPLE)
champion_metrics = score_champion(df["diagnosis"], champion_stand_in.pipeline.predict(
    df.drop(columns=["diagnosis"]).to_numpy(dtype=object)
), task="classification")
print("Champion metrics (stand-in):", champion_metrics)

# to_challenger_upload() assembles the JSON-serializable payload the
# dashboard/API expects - export serialization, SDK/core version stamping,
# and complexity-as-a-string are all handled for you.
payload = to_challenger_upload(
    challenger_result,
    n_samples=len(df),
    champion_metrics=champion_metrics,
)

with open("challenger.json", "w") as f:
    json.dump(payload, f, indent=2)
print("\nWrote challenger.json - upload it via the ProxyML dashboard's "
      "'Upload challenger' button on a challenger-comparison project, or "
      "POST it directly to /app/projects/{project_id}/challenger.")

# If you don't have champion_metrics yet (e.g. you want to hand this export
# to whoever owns the champion model, or just archive it), omit
# champion_metrics entirely - you get a self-contained export of the
# challenger alone, and can fill champion_metrics in later:
#
#   payload = to_challenger_upload(challenger_result, n_samples=len(df))
