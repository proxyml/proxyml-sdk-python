"""Post a champion/challenger comparison to a GitHub pull request from CI.

Typical use in a GitHub Actions job, after training and uploading a
challenger::

    from proxyml import local, ci
    from proxyml.client import upload_challenger

    result = local.train_challenger(...)
    payload = local.to_challenger_upload(result, champion_metrics=...)
    upload_challenger(payload)
    ci.post_challenger_comment()

Reads GITHUB_TOKEN, GITHUB_REPOSITORY, and GITHUB_EVENT_PATH from the
environment by default — all three are set automatically inside a GitHub
Actions job triggered by a pull_request event, alongside PROXYML_CI_TOKEN
(set as a repo secret), so the common case needs no arguments at all.

This never runs server-side — ProxyML's backend doesn't talk to GitHub —
so it lives in the SDK, not proxyml_core.
"""

import json
import logging
import os

import requests

from proxyml.client import get_challenger_summary

logger = logging.getLogger(__name__)

_TIMEOUT = 30
_COMMENT_MARKER = "<!-- proxyml-challenger-comment -->"


def _github_headers(github_token: str) -> dict:
    return {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _resolve_github_token(github_token: str | None) -> str:
    token = github_token or os.getenv("GITHUB_TOKEN", "")
    if not token:
        raise EnvironmentError(
            "GITHUB_TOKEN is not set and no github_token was provided. GitHub Actions "
            "sets this automatically for a job that declares 'env: {GITHUB_TOKEN: "
            "${{ secrets.GITHUB_TOKEN }}}' — add that to the step if it's missing."
        )
    return token


def _resolve_repo(repo: str | None) -> str:
    resolved = repo or os.getenv("GITHUB_REPOSITORY", "")
    if not resolved:
        raise EnvironmentError(
            "GITHUB_REPOSITORY is not set and no repo was provided. "
            "This is set automatically inside GitHub Actions."
        )
    return resolved


def _resolve_pr_number(pr_number: int | None) -> int:
    if pr_number is not None:
        return pr_number
    event_path = os.getenv("GITHUB_EVENT_PATH")
    if not event_path:
        raise EnvironmentError(
            "GITHUB_EVENT_PATH is not set and no pr_number was provided. "
            "This is set automatically inside GitHub Actions on pull_request events "
            "— on other event types, pass pr_number explicitly."
        )
    with open(event_path) as f:
        event = json.load(f)
    number = (event.get("pull_request") or {}).get("number", event.get("number"))
    if number is None:
        raise ValueError(
            f"Could not find a pull request number in {event_path}. This workflow may "
            "not be running on a pull_request event — pass pr_number explicitly."
        )
    return number


def _find_existing_comment(repo: str, pr_number: int, headers: dict) -> int | None:
    """Look for a prior challenger comment on this PR, so re-runs edit it instead of piling up."""
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    r = requests.get(url, headers=headers, timeout=_TIMEOUT)
    r.raise_for_status()
    for comment in r.json():
        if _COMMENT_MARKER in comment.get("body", ""):
            return comment["id"]
    return None


def post_challenger_comment(
    *,
    ci_token: str | None = None,
    github_token: str | None = None,
    repo: str | None = None,
    pr_number: int | None = None,
) -> dict | None:
    """Post or update a sticky PR comment summarizing the latest challenger comparison.

    Call this after uploading a challenger via proxyml.client.upload_challenger()
    — it fetches the CI-ready summary for the project authorized by the CI
    token, then posts it to the pull request. A second run on the same PR
    edits the existing comment in place (matched via a hidden HTML marker)
    rather than adding a duplicate, the same way coverage bots behave.

    Args:
        ci_token: overrides PROXYML_CI_TOKEN if provided.
        github_token: overrides GITHUB_TOKEN if provided.
        repo: "owner/name", overrides GITHUB_REPOSITORY if provided.
        pr_number: overrides the PR number GitHub Actions reports via GITHUB_EVENT_PATH.

    Returns:
        The created/updated GitHub comment JSON, or None if the challenger
        summary couldn't be fetched or the GitHub API call failed.
    """
    summary = get_challenger_summary(ci_token=ci_token)
    if summary is None:
        return None

    headers = _github_headers(_resolve_github_token(github_token))
    resolved_repo = _resolve_repo(repo)
    resolved_pr = _resolve_pr_number(pr_number)
    body = f"{_COMMENT_MARKER}\n{summary['markdown_summary']}"

    try:
        existing_id = _find_existing_comment(resolved_repo, resolved_pr, headers)
        base_url = f"https://api.github.com/repos/{resolved_repo}/issues"
        if existing_id is not None:
            r = requests.patch(
                f"{base_url}/comments/{existing_id}", headers=headers, json={"body": body}, timeout=_TIMEOUT,
            )
        else:
            r = requests.post(
                f"{base_url}/{resolved_pr}/comments", headers=headers, json={"body": body}, timeout=_TIMEOUT,
            )
        r.raise_for_status()
    except requests.exceptions.RequestException as exc:
        logger.error("Failed to post challenger comment to GitHub: %s", exc)
        return None
    return r.json()
