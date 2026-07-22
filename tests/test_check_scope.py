import json
import os
import sys
import tempfile
import urllib.request
from unittest import mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".github", "scripts"))
import check_scope  # noqa: E402,I001


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fake_response(data, status=200):
    return mock.Mock(
        __enter__=mock.Mock(return_value=mock.Mock(read=lambda: data, status=status, reason="OK")),
        __exit__=mock.Mock(return_value=False),
    )


# ---------------------------------------------------------------------------
# Tests for _fetch_pr_body_from_api
# ---------------------------------------------------------------------------

class TestFetchPrBodyFromApi:
    def test_returns_body_on_success(self):
        pr_data = json.dumps({"body": "SCOPE: kronyx/\n\nSome details"}).encode("utf-8")
        with mock.patch("urllib.request.urlopen", return_value=fake_response(pr_data)):
            body, error = check_scope._fetch_pr_body_from_api("owner/repo", "1", "token")
        assert body == "SCOPE: kronyx/\n\nSome details"
        assert error is None

    def test_returns_empty_string_when_body_key_missing(self):
        pr_data = json.dumps({"title": "My PR"}).encode("utf-8")
        with mock.patch("urllib.request.urlopen", return_value=fake_response(pr_data)):
            body, error = check_scope._fetch_pr_body_from_api("owner/repo", "1", "token")
        assert body == ""
        assert error is None

    def test_handles_http_error(self):
        err = urllib.error.HTTPError("url", 404, "Not Found", {}, fp=mock.Mock())
        with mock.patch("urllib.request.urlopen", side_effect=err):
            body, error = check_scope._fetch_pr_body_from_api("owner/repo", "1", "token")
        assert body is None
        assert "HTTP 404" in error

    def test_handles_url_error(self):
        err = urllib.error.URLError("connection refused")
        with mock.patch("urllib.request.urlopen", side_effect=err):
            body, error = check_scope._fetch_pr_body_from_api("owner/repo", "1", "token")
        assert body is None
        assert "URL error" in error

    def test_requires_all_params(self):
        body, error = check_scope._fetch_pr_body_from_api("", "1", "token")
        assert body is None
        assert "missing" in error

        body, error = check_scope._fetch_pr_body_from_api("owner/repo", "", "token")
        assert body is None
        assert "missing" in error

        body, error = check_scope._fetch_pr_body_from_api("owner/repo", "1", "")
        assert body is None
        assert "missing" in error


# ---------------------------------------------------------------------------
# Tests for _load_from_event_file
# ---------------------------------------------------------------------------

class TestLoadFromEventFile:
    def test_loads_body_from_valid_event(self):
        event = {
            "pull_request": {
                "body": "SCOPE: kronyx/viz/"
            }
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(event, f)
            path = f.name

        with mock.patch.dict(os.environ, {"GITHUB_EVENT_PATH": path}, clear=False):
            body, ep, exists = check_scope._load_from_event_file()
        os.unlink(path)

        assert body == "SCOPE: kronyx/viz/"
        assert path in ep
        assert exists is True

    def test_returns_none_when_event_path_missing(self):
        with mock.patch.dict(os.environ, {"GITHUB_EVENT_PATH": ""}, clear=False):
            body, ep, exists = check_scope._load_from_event_file()
        assert body is None
        assert exists is False

    def test_returns_none_when_file_not_found(self):
        with mock.patch.dict(os.environ, {"GITHUB_EVENT_PATH": "/nonexistent.json"}, clear=False):
            body, ep, exists = check_scope._load_from_event_file()
        assert body is None
        assert exists is False


# ---------------------------------------------------------------------------
# Tests for load_pr_body (priority chain)
# ---------------------------------------------------------------------------

class TestLoadPrBody:
    def test_prefers_api_over_event_and_env(self):
        pr_data = json.dumps({"body": "SCOPE: api/"}).encode("utf-8")
        env = {
            "GITHUB_REPOSITORY": "owner/repo",
            "PR_NUMBER": "1",
            "GITHUB_TOKEN": "token",
            "PR_BODY": "SCOPE: env/",
        }

        event = {"pull_request": {"body": "SCOPE: event/"}}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(event, f)
            env["GITHUB_EVENT_PATH"] = f.name

        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch("urllib.request.urlopen", return_value=fake_response(pr_data)):
                body, source, event_path, event_exists = check_scope.load_pr_body()

        os.unlink(env["GITHUB_EVENT_PATH"])

        assert body == "SCOPE: api/"
        assert source == "api"
        assert event_exists is False

    def test_falls_back_to_event_when_api_fails(self):
        env = {
            "GITHUB_REPOSITORY": "owner/repo",
            "PR_NUMBER": "1",
            "GITHUB_TOKEN": "token",
        }

        event = {"pull_request": {"body": "SCOPE: event/"}}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(event, f)
            env["GITHUB_EVENT_PATH"] = f.name

        err = urllib.error.HTTPError("url", 403, "Forbidden", {}, fp=mock.Mock())
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch("urllib.request.urlopen", side_effect=err):
                body, source, event_path, event_exists = check_scope.load_pr_body()

        os.unlink(env["GITHUB_EVENT_PATH"])

        assert body == "SCOPE: event/"
        assert source == "event_file"
        assert event_exists is True

    def test_falls_back_to_env_when_api_and_event_missing(self):
        env = {
            "GITHUB_REPOSITORY": "owner/repo",
            "PR_NUMBER": "1",
            "GITHUB_TOKEN": "token",
            "PR_BODY": "SCOPE: env/",
        }

        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch("urllib.request.urlopen", side_effect=Exception("no network")):
                body, source, event_path, event_exists = check_scope.load_pr_body()

        assert body == "SCOPE: env/"
        assert source == "env"
        assert event_exists is False

    def test_returns_empty_when_all_fail(self):
        env = {
            "GITHUB_REPOSITORY": "owner/repo",
            "PR_NUMBER": "1",
            "GITHUB_TOKEN": "token",
        }

        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch("urllib.request.urlopen", side_effect=Exception("no network")):
                body, source, event_path, event_exists = check_scope.load_pr_body()

        assert body == ""
        assert source == "none"
        assert event_exists is False

    def test_api_returns_none_treated_as_failure(self):
        """Even if API returns None body, fallback chain continues."""
        env = {
            "GITHUB_REPOSITORY": "owner/repo",
            "PR_NUMBER": "1",
            "GITHUB_TOKEN": "token",
        }

        event = {"pull_request": {"body": "SCOPE: event/"}}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(event, f)
            env["GITHUB_EVENT_PATH"] = f.name

        pr_data = json.dumps({"body": None}).encode("utf-8")
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch("urllib.request.urlopen", return_value=fake_response(pr_data)):
                body, source, event_path, event_exists = check_scope.load_pr_body()

        os.unlink(env["GITHUB_EVENT_PATH"])

        assert body == "SCOPE: event/"
        assert source == "event_file"


# ---------------------------------------------------------------------------
# Tests for parse_scope
# ---------------------------------------------------------------------------

class TestParseScope:
    def test_simple_scope_line(self):
        assert check_scope.parse_scope("SCOPE: kronyx/viz/") == ["kronyx/viz/"]

    def test_multiple_patterns(self):
        assert check_scope.parse_scope("SCOPE: kronyx/losses.py, kronyx/utils.py, tests/") == [
            "kronyx/losses.py",
            "kronyx/utils.py",
            "tests/",
        ]

    def test_case_insensitive(self):
        assert check_scope.parse_scope("scope: kronyx/") == ["kronyx/"]
        assert check_scope.parse_scope("Scope: kronyx/") == ["kronyx/"]
        assert check_scope.parse_scope("SCOPE: kronyx/") == ["kronyx/"]

    def test_multiline_description(self):
        body = "Title\n\nSCOPE: kronyx/viz/, kronyx/model.py\n\nSome details here."
        assert check_scope.parse_scope(body) == ["kronyx/viz/", "kronyx/model.py"]

    def test_leading_whitespace(self):
        assert check_scope.parse_scope("  SCOPE: kronyx/  ") == ["kronyx/"]

    def test_empty_patterns_raises(self):
        with pytest.raises(ValueError, match="no patterns"):
            check_scope.parse_scope("SCOPE:   ")

    def test_missing_scope_raises(self):
        with pytest.raises(ValueError, match="missing a required 'SCOPE:'"):
            check_scope.parse_scope("No scope here.")


# ---------------------------------------------------------------------------
# Tests for get_changed_files
# ---------------------------------------------------------------------------

class TestGetChangedFiles:
    def test_returns_sorted_files(self):
        mock_result = mock.Mock(stdout="docs/README.md\nkronyx/model.py\ntests/test_model.py\n")
        with mock.patch("subprocess.run", return_value=mock_result):
            files = check_scope.get_changed_files()
        assert files == ["docs/README.md", "kronyx/model.py", "tests/test_model.py"]

    def test_empty_stdout_returns_empty_list(self):
        mock_result = mock.Mock(stdout="")
        with mock.patch("subprocess.run", return_value=mock_result):
            files = check_scope.get_changed_files()
        assert files == []

    def test_strips_whitespace(self):
        mock_result = mock.Mock(stdout="  kronyx/model.py  \n\n  tests/test_dense.py  \n")
        with mock.patch("subprocess.run", return_value=mock_result):
            files = check_scope.get_changed_files()
        assert files == ["kronyx/model.py", "tests/test_dense.py"]


# ---------------------------------------------------------------------------
# Tests for is_in_scope
# ---------------------------------------------------------------------------

class TestIsInScope:
    def test_directory_prefix(self):
        assert check_scope.is_in_scope("kronyx/viz/arch.py", ["kronyx/viz/"]) is True
        assert check_scope.is_in_scope("kronyx/viz_bad.py", ["kronyx/viz/"]) is False

    def test_glob_exact_match(self):
        assert check_scope.is_in_scope("kronyx/model.py", ["kronyx/model.py"]) is True
        assert check_scope.is_in_scope("kronyx/losses.py", ["kronyx/model.py"]) is False

    def test_glob_wildcard(self):
        assert check_scope.is_in_scope("tests/test_dense.py", ["tests/*"]) is True
        assert check_scope.is_in_scope("src/test_dense.py", ["tests/*"]) is False

    def test_multiple_patterns(self):
        patterns = ["kronyx/viz/", "kronyx/model.py", "tests/*"]
        assert check_scope.is_in_scope("kronyx/viz/plot.py", patterns) is True
        assert check_scope.is_in_scope("kronyx/model.py", patterns) is True
        assert check_scope.is_in_scope("tests/test_dense.py", patterns) is True
        assert check_scope.is_in_scope("README.md", patterns) is False


# ---------------------------------------------------------------------------
# Tests for debug_pr_body_source
# ---------------------------------------------------------------------------

class TestDebugPrBodySource:
    def test_outputs_sections(self, capsys):
        check_scope.debug_pr_body_source("body", "api", "/path.json", True)
        out = capsys.readouterr().out
        assert "PR body source     : api" in out
        assert "GITHUB_EVENT_PATH : '/path.json'" in out
        assert "Event file exists  : True" in out
        assert "PR body length     : 4" in out
        assert "Detected SCOPE line" in out
