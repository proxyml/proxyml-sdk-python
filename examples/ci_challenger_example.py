"""
ci_challenger_example - trains a challenger locally, uploads it to ProxyML
via a project-scoped CI token, and posts the result as a PR comment.

Intended to run inside a CI job (see examples/github_actions_challenger.yml)
on every pull request, so a reviewer sees "challenger beats/trails champion
by X" without leaving GitHub.

Prerequisites:
    pip install 'proxyml[local]' scikit-learn pandas

Environment (set as CI secrets — see github_actions_challenger.yml):
    PROXYML_CI_TOKEN   - project-scoped token, minted from the ProxyML
                          dashboard's project page. NOT the account's
                          general PROXYML_API_KEY.
    GITHUB_TOKEN        - set automatically by GitHub Actions.
"""
from sklearn.datasets import load_breast_cancer

from proxyml import ci
from proxyml.client import upload_challenger
from proxyml.local import Complexity, to_challenger_upload, train_auto_challenger

# A real dataset with a real binary target - stands in for your own
# versioned held-out data (checked into the repo, or pulled from wherever
# this CI job's evaluation data lives).
data = load_breast_cancer(as_frame=True)
df = data.frame.rename(columns={"target": "diagnosis"})

# champion_predictions should be YOUR actual production model's predictions
# on df, one per row (same order) - that's what makes the comparison
# meaningful. This example has no real champion handy, so it trains a
# differently-regularized challenger purely as a stand-in. Do not do this in
# real usage - swap in your actual champion's predictions.
champion_stand_in = train_auto_challenger(df, "diagnosis", task="classification", complexity=Complexity.SIMPLE)
champion_predictions = champion_stand_in.pipeline.predict(
    df.drop(columns=["diagnosis"]).to_numpy(dtype=object)
)

challenger_result = train_auto_challenger(
    df, "diagnosis", task="classification", complexity=Complexity.MODERATE,
    champion_predictions=champion_predictions,
)
print("Challenger metrics:", challenger_result.metrics)
print("Champion metrics (stand-in):", challenger_result.champion_metrics)

payload = to_challenger_upload(challenger_result)
result = upload_challenger(payload)  # reads PROXYML_CI_TOKEN from the environment
if result is None:
    raise SystemExit("Challenger upload failed — see logged error above.")
print(f"Uploaded challenger version {result['challenger_version']}")

# Reads GITHUB_TOKEN/GITHUB_REPOSITORY/GITHUB_EVENT_PATH from the Actions
# environment automatically. A second run on the same PR edits this comment
# in place rather than adding a duplicate.
comment = ci.post_challenger_comment()
if comment is None:
    raise SystemExit("Posting the PR comment failed — see logged error above.")
print(f"Posted PR comment: {comment['html_url']}")
