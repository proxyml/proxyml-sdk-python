# Challenger Flow

How to compare a challenger model against your production (champion) model, end to end: train locally, upload the result, and read the verdict in the dashboard, in CI, or from an AI agent.

The pieces are documented individually in [`api.md`](api.md) and [`quickstart.md`](quickstart.md). This page is the connective tissue: what happens in what order, what leaves your machine, and how to wire it into a pull request.

---

## The idea

A **champion** is the model you run in production. A **challenger** is a simple, interpretable linear model trained on the same real, labeled data. Comparing the two answers "how much performance would we give up (or gain) with something we can fully explain?"

Training and scoring happen **on your machine**. Your data is never uploaded. ProxyML receives only the fitted model, summary metrics, and a hash of the labels (see [What leaves your machine](#what-leaves-your-machine)).

```
 your data + champion predictions
          │
          ▼
  train_auto_challenger()        local, needs proxyml[local]
          │  TrainedChallenger
          ▼
   to_challenger_upload()        builds the JSON payload
          │
     ┌────┴───────────────┐
     ▼                    ▼
 dashboard upload     upload_challenger()      CI, with PROXYML_CI_TOKEN
 (manual)                  │
                           ▼
                  get_challenger_summary()
                           │
                           ▼
              post_challenger_comment()        sticky PR comment
```

---

## 1. Train locally

```python
from proxyml.local import train_auto_challenger

result = train_auto_challenger(
    "holdout.csv",
    "approved",                          # ground-truth label column
    task="classification",
    champion_predictions=champion_preds, # one per row of holdout.csv, same order
)
print(result.metrics)           # challenger, e.g. {"f1": 0.91, "accuracy": 0.90}
print(result.champion_metrics)  # champion, scored on the identical rows
```

Install the extra first: `pip install 'proxyml[local]'`.

Two rules matter here:

- **`champion_predictions` must be your real production model's predictions** on the same rows, in the same order. The comparison is only as honest as this input.
- **Use real, held-out labels.** Evaluating on data either model was trained on flatters whichever one saw it.

Rows with a missing label are dropped before training and scoring, for both models, so they're always judged on the same population. `result.n_samples_dropped_unlabeled` and `result.population_note` record how many were dropped. That number travels with the upload so a reviewer can see it.

If you already have a `FeatureSchema` and want control over the feature/target split, use `train_challenger(df, target, schema)` instead. Both default to `Complexity.MODERATE`; pass `SIMPLE` or `FLEXIBLE` for stronger or weaker regularization.

## 2. Build the upload payload

```python
from proxyml.local import to_challenger_upload

payload = to_challenger_upload(result)
```

The payload contains the exported model, both sets of metrics, sample counts, and two label fingerprints. When `champion_predictions` was passed to training, everything is derived automatically.

If you scored the champion separately with `score_champion()`, pass `champion_metrics=` and `champion_labels=` so the fingerprint can be computed. If the two fingerprints disagree, `to_challenger_upload()` raises `ValueError` immediately, before any network call.

## 3. Upload it

### Manually, through the dashboard

Save the payload and use the project's **Upload challenger** button:

```python
import json
json.dump(payload, open("challenger.json", "w"))
```

### From CI, with a project token

```python
from proxyml.client import upload_challenger

if upload_challenger(payload) is None:
    raise SystemExit("upload failed")
```

This authenticates with `PROXYML_CI_TOKEN`, **not** `PROXYML_API_KEY`.

**Minting a CI token:** open the project in the dashboard, go to *Challenger comparison → CI token → Mint*, and store the value as a CI secret named `PROXYML_CI_TOKEN`. It's shown once.

A CI token can only upload to and read from the one project it was minted for, so a leaked token can't touch your other projects or your account. Tokens are stored hashed and can be revoked from the same page.

## 4. Read the verdict

```python
from proxyml.client import get_challenger_summary

summary = get_challenger_summary()
summary["winner"]                 # "challenger" | "champion" | "tie"
summary["primary_metric"]         # "f1" or "r2"
summary["delta"]                  # challenger minus champion
summary["data_fingerprint_match"] # were both scored on the same labels?
summary["markdown_summary"]       # ready to paste anywhere
```

`get_challenger_summary()` returns `None` if nothing has been uploaded yet.

**Trust note:** metrics are self-reported and unsigned, the same as a dashboard upload. ProxyML records what you send. It can't verify that your champion predictions are the real production ones. The fingerprint check catches an accidental mismatch (different rows or ordering). It isn't a defense against someone fabricating inputs.

## 5. Put it on every pull request

```python
from proxyml import ci

ci.post_challenger_comment()
```

This fetches the summary and posts it as a PR comment. Re-runs edit the same comment in place (it's matched by a hidden marker), so a PR with ten pushes has one comment, not ten.

A complete workflow:

- [`examples/ci_challenger_example.py`](../examples/ci_challenger_example.py): train, upload, and comment in one script.
- [`examples/github_actions_challenger.yml`](../examples/github_actions_challenger.yml): drop-in GitHub Actions workflow. It needs the `PROXYML_CI_TOKEN` secret, plus `pull-requests: write` permission for the built-in `GITHUB_TOKEN`.

The example script uses a stand-in champion so it runs anywhere. Replace that with your real model's predictions before relying on the output.

### Other CI systems (GitLab, Bitbucket, etc.)

`ci.post_challenger_comment()` only supports GitHub. On other hosts, the upload and the summary work the same way. Only the posting step differs. `get_challenger_summary()` returns a host-neutral `markdown_summary`, so a short script can hand it to whatever your CI provides:

```python
# summary.py: writes the comparison to a file
import sys
from proxyml.client import get_challenger_summary

summary = get_challenger_summary()
if summary is None:
    sys.exit("no challenger summary available")
open("challenger-summary.md", "w").write(summary["markdown_summary"])
```

Then post the file as a merge request note. For example, in GitLab CI (an MR pipeline job):

```yaml
challenger:
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
  script:
    - pip install 'proxyml[local]' scikit-learn pandas
    - python train_and_upload.py        # train + upload_challenger(); needs PROXYML_CI_TOKEN
    - python summary.py
    - >
      curl --fail --request POST
      --header "PRIVATE-TOKEN: $GITLAB_COMMENT_TOKEN"
      --data-urlencode "body@challenger-summary.md"
      "$CI_API_V4_URL/projects/$CI_PROJECT_ID/merge_requests/$CI_MERGE_REQUEST_IID/notes"
```

Things to know:

- **You need a token with permission to comment.** `GITLAB_COMMENT_TOKEN` here is a project access token (API scope) stored as a masked CI variable. GitLab's built-in job token generally can't write MR notes. Scope the token as narrowly as your plan allows.
- **It isn't sticky.** Each run adds a new note instead of editing the previous one, so a busy MR collects one per push. If that bothers you, list the MR's notes, find the one containing a marker string you add to the body, and update it instead of posting a new one.
- **Bitbucket works the same way** with its pull request comments API and an app password or repository access token. Pipelines expose the repo and PR as `BITBUCKET_REPO_FULL_NAME` and `BITBUCKET_PR_ID`.
- **These requests aren't tested against current GitLab and Bitbucket docs.** Check the endpoint and auth details for your version, especially on self-hosted instances.

---

## From an AI agent (MCP)

[`mcp-proxyml`](https://pypi.org/project/mcp-proxyml/) 0.1.8+ exposes the same flow as tools, so an agent can do it without leaving the conversation:

| Tool | Does |
|---|---|
| `proxyml_train_challenger` | Trains locally and returns an `upload_payload` (needs the `challenger` extra) |
| `proxyml_score_champion` | Scores champion predictions; returns metrics plus a `data_fingerprint` |
| `proxyml_upload_challenger` | Uploads a payload via `/ci/challenger` using `PROXYML_CI_TOKEN` |
| `proxyml_get_challenger_summary` | Reads the latest comparison via `/ci/challenger/summary` |

The two CI tools send the project-scoped token only. The account-wide `PROXYML_API_KEY` is deliberately never sent to those endpoints.

---

## What leaves your machine

| Sent to ProxyML | Never sent |
|---|---|
| Fitted linear model (coefficients, feature names, schema summary) | Your rows, features, or raw labels |
| Challenger and champion metrics (F1/accuracy or R²) | Champion predictions |
| Sample counts and dropped-row count | Feature values |
| SHA-256 hashes of the label arrays | Anything that could rebuild the labels from the hash |
| SDK and core versions | |

The exported model does include feature names, category names, and per-feature means and standard deviations, since scoring needs them. Treat those as metadata about your data. If a column name or category value is itself sensitive, rename it before training.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `ValueError` about fingerprints in `to_challenger_upload()` | Champion and challenger were scored on different labels, rows, or order. Re-score on the same array. |
| `upload_challenger()` returns `None` | Missing or revoked `PROXYML_CI_TOKEN`, project deleted, or a validation error. The cause is logged. |
| `get_challenger_summary()` returns `None` | No challenger uploaded yet for this project. |
| `data_fingerprint_match` is false | The two sets of metrics came from different data. Don't trust the delta. |
| Upload rejected on a categorical probability sum | Schema category probabilities must sum to 1 within about 1e-5. Re-infer the schema rather than hand-editing it. |
| Comment doesn't appear on the PR | The workflow lacks `pull-requests: write`, or it's running on a fork (fork PRs get a read-only token). |

---

## See also

- [`quickstart.md`](quickstart.md#local-training): local training basics
- [`api.md`](api.md#continuous-integration-proxymlci): full parameter reference
- Blog: [Challenger mode is live](https://proxyml.ai/blog/challenger-mode-is-live.html) and [Same data or it doesn't count](https://proxyml.ai/blog/same-data-or-it-doesnt-count.html)
