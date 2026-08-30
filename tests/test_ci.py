import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from proxyml.ci import (
    _COMMENT_MARKER,
    _resolve_github_token,
    _resolve_pr_number,
    _resolve_repo,
    post_challenger_comment,
)


def _mock_response(status_code, json_body):
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = json_body
    r.raise_for_status = MagicMock()
    if status_code >= 400:
        r.raise_for_status.side_effect = requests.exceptions.HTTPError(f"{status_code} error")
    return r


# ---------------------------------------------------------------------------
# env resolution
# ---------------------------------------------------------------------------

def test_resolve_github_token_raises_when_missing(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(EnvironmentError, match="GITHUB_TOKEN"):
        _resolve_github_token(None)


def test_resolve_github_token_explicit_overrides_env(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "env-token")
    assert _resolve_github_token("explicit-token") == "explicit-token"


def test_resolve_repo_raises_when_missing(monkeypatch):
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    with pytest.raises(EnvironmentError, match="GITHUB_REPOSITORY"):
        _resolve_repo(None)


def test_resolve_repo_reads_env(monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "acme/widgets")
    assert _resolve_repo(None) == "acme/widgets"


def test_resolve_pr_number_explicit_skips_event_file(monkeypatch):
    monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
    assert _resolve_pr_number(42) == 42


def test_resolve_pr_number_raises_without_event_path(monkeypatch):
    monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
    with pytest.raises(EnvironmentError, match="GITHUB_EVENT_PATH"):
        _resolve_pr_number(None)


def test_resolve_pr_number_reads_pull_request_event(tmp_path, monkeypatch):
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps({"pull_request": {"number": 17}}))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    assert _resolve_pr_number(None) == 17


def test_resolve_pr_number_raises_on_non_pr_event(tmp_path, monkeypatch):
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps({"ref": "refs/heads/main"}))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    with pytest.raises(ValueError, match="pull request number"):
        _resolve_pr_number(None)


# ---------------------------------------------------------------------------
# post_challenger_comment
# ---------------------------------------------------------------------------

@patch("proxyml.ci.get_challenger_summary")
def test_post_challenger_comment_returns_none_without_summary(mock_summary):
    mock_summary.return_value = None
    assert post_challenger_comment(github_token="t", repo="acme/widgets", pr_number=1) is None


@patch("proxyml.ci.requests.post")
@patch("proxyml.ci.requests.get")
@patch("proxyml.ci.get_challenger_summary")
def test_post_challenger_comment_creates_new_comment_when_none_exists(mock_summary, mock_get, mock_post):
    mock_summary.return_value = {"markdown_summary": "**ProxyML challenger beats champion** by 0.19 r2"}
    mock_get.return_value = _mock_response(200, [])  # no existing comments
    mock_post.return_value = _mock_response(201, {"id": 99, "body": "posted"})

    result = post_challenger_comment(github_token="gh-token", repo="acme/widgets", pr_number=7)

    assert result == {"id": 99, "body": "posted"}
    mock_post.assert_called_once()
    url = mock_post.call_args.args[0]
    assert url == "https://api.github.com/repos/acme/widgets/issues/7/comments"
    body = mock_post.call_args.kwargs["json"]["body"]
    assert _COMMENT_MARKER in body
    assert "0.19 r2" in body


@patch("proxyml.ci.requests.patch")
@patch("proxyml.ci.requests.get")
@patch("proxyml.ci.get_challenger_summary")
def test_post_challenger_comment_updates_existing_sticky_comment(mock_summary, mock_get, mock_patch):
    mock_summary.return_value = {"markdown_summary": "updated result"}
    mock_get.return_value = _mock_response(
        200, [{"id": 55, "body": f"{_COMMENT_MARKER}\nold result"}, {"id": 56, "body": "unrelated comment"}],
    )
    mock_patch.return_value = _mock_response(200, {"id": 55, "body": "updated"})

    result = post_challenger_comment(github_token="gh-token", repo="acme/widgets", pr_number=7)

    assert result == {"id": 55, "body": "updated"}
    mock_patch.assert_called_once()
    url = mock_patch.call_args.args[0]
    assert url == "https://api.github.com/repos/acme/widgets/issues/comments/55"


@patch("proxyml.ci.requests.post")
@patch("proxyml.ci.requests.get")
@patch("proxyml.ci.get_challenger_summary")
def test_post_challenger_comment_returns_none_on_github_error(mock_summary, mock_get, mock_post):
    mock_summary.return_value = {"markdown_summary": "result"}
    mock_get.return_value = _mock_response(200, [])
    mock_post.return_value = _mock_response(403, {"message": "forbidden"})

    assert post_challenger_comment(github_token="gh-token", repo="acme/widgets", pr_number=7) is None
