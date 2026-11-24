"""
ci_data_drift_example - compares a recent batch of data against the data a
ProxyML model was trained on, and fails the job if it has drifted.

Only summary statistics (means, standard deviations, category names with
their frequencies, count rates) are sent to ProxyML, never rows. Each feature
is compared on its own; changes in the relationships between features are
not detected.

Usage:
    python ci_data_drift_example.py MODEL_ID recent_data.csv [--label LABEL]
        [--baseline-rows N] [--fail-on-insufficient]

Exit status: 0 = pass, 1 = drift (or the check failed), 2 = too few rows for
a verdict (only with --fail-on-insufficient; otherwise 0 with a warning).

Environment:
    PROXYML_API_KEY - your account API key (the drift endpoints are not
                       covered by the project-scoped CI token).

Prerequisites:
    pip install proxyml pandas
"""
import argparse
import sys

import pandas as pd

import proxyml

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("model_id")
parser.add_argument("csv", help="recent data, with the same columns as the training data")
parser.add_argument("--label", help="free-text tag stored with the check, e.g. a date or commit SHA")
parser.add_argument(
    "--baseline-rows", type=int,
    help="rows the model's training data had; widens the noise allowance. Recommended for small training sets.",
)
parser.add_argument("--fail-on-insufficient", action="store_true")
args = parser.parse_args()

result = proxyml.check_data_drift(
    args.model_id,
    pd.read_csv(args.csv),
    label=args.label,
    baseline_n_samples=args.baseline_rows,
    # Tune the policy here if the defaults are too strict or too loose for you, e.g.
    # thresholds={"mean_shift": 0.3, "category_tvd": 0.15},
)
if result is None:
    sys.exit("Drift check failed - see logged error above.")

print(f"Data drift check: {result['status']} ({result['n_samples']} rows)")
for f in result["features"]:
    if f["status"] != "ok":
        print(f"  {f['name']}: {f['status']} - {'; '.join(f['reasons'])}")

if result["status"] == "drift":
    sys.exit(1)
if result["status"] == "insufficient_sample":
    print("Warning: too few rows for a verdict.", file=sys.stderr)
    sys.exit(2 if args.fail_on_insufficient else 0)
