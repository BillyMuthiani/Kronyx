#!/usr/bin/env python3
"""Scope-check script for CI.

Parses a SCOPE: line from the PR body and compares it against the actual
changed files in the PR. Exits non-zero if the PR body is missing a scope
declaration or if any changed file is not covered by the declared scope.

Usage:
    python check_scope.py

Environment variables required:
    GITHUB_TOKEN   — token with repo scope (for posting PR comments).
    GITHUB_REPOSITORY — owner/repo slug (provided by Actions automatically).
    PR_NUMBER      — the pull request number (provided by Actions automatically).
    PR_BODY        — the full text of the PR description/body.

Output:
    Prints the parsed scope and out-of-scope files to stdout.
    Posts a comment on the PR if issues are found.
    Exit code 0 if scope is valid, non-zero otherwise.
"""

import fnmatch
import json
import os
import re
import subprocess
import sys
import urllib.request


def parse_scope(pr_body: str) -> list[str]:
    """Extract the SCOPE: declaration from the PR body.

    Looks for a line that starts with 'SCOPE:' (case-insensitive, optional
    leading whitespace). The value after the colon is a comma-separated list
    of file paths or glob patterns.

    Args:
        pr_body: Full text of the PR description.

    Returns:
        List of stripped scope patterns.

    Raises:
        ValueError: If no SCOPE: line is present in the body.
    """
    for line in pr_body.splitlines():
        if re.match(r"^\s*SCOPE\s*:", line, re.IGNORECASE):
            raw = line.split(":", 1)[1]
            patterns = [p.strip() for p in raw.split(",") if p.strip()]
            if not patterns:
                raise ValueError("SCOPE: line found but contains no patterns.")
            return patterns
    raise ValueError(
        "PR body is missing a required 'SCOPE:' line. "
        "Add one listing every file or glob your change touches, e.g.:\n"
        "SCOPE: kronyx/losses.py, kronyx/utils.py, tests/"
    )


def get_changed_files() -> list[str]:
    """Return the list of files changed in this PR vs the base branch.

    Uses git diff against origin/main. Assumes the repository was fetched
    with enough history for origin/main to exist.

    Returns:
        Sorted list of changed file paths relative to repo root.
    """
    result = subprocess.run(
        ["git", "diff", "--name-only", "origin/main...HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    files = [f.strip() for f in result.stdout.splitlines() if f.strip()]
    return sorted(files)


def is_in_scope(file_path: str, scope_patterns: list[str]) -> bool:
    """Check whether a file path matches any of the declared scope patterns.

    Supports exact matches and fnmatch-style glob patterns (e.g. 'tests/*'
    matches 'tests/test_losses.py'). Directory prefixes ending with '/'
    are treated as recursive directory matches.

    Args:
        file_path: Path of the changed file, relative to repo root.
        scope_patterns: List of patterns declared in SCOPE:.

    Returns:
        True if the file is covered by at least one pattern.
    """
    for pattern in scope_patterns:
        if pattern.endswith("/"):
            if file_path.startswith(pattern):
                return True
        elif fnmatch.fnmatch(file_path, pattern):
            return True
    return False


def post_comment(message: str) -> None:
    """Post a comment on the PR via the GitHub API.

    Reads GITHUB_REPOSITORY and PR_NUMBER from the environment and uses
    GITHUB_TOKEN for authentication.

    Args:
        message: Markdown body of the comment to post.
    """
    repo = os.environ.get("GITHUB_REPOSITORY")
    pr_number = os.environ.get("PR_NUMBER")
    token = os.environ.get("GITHUB_TOKEN")

    if not all([repo, pr_number, token]):
        print("Warning: missing GitHub context, cannot post comment.", file=sys.stderr)
        return

    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    payload = json.dumps({"body": message}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            print(f"Posted comment: HTTP {resp.status}")
    except Exception as exc:
        print(f"Warning: failed to post comment: {exc}", file=sys.stderr)


def main() -> int:
    pr_body = os.environ.get("PR_BODY", "")

    try:
        scope_patterns = parse_scope(pr_body)
    except ValueError as exc:
        print(f"SCOPE DECLARATION MISSING: {exc}")
        post_comment(
            f"## Scope-check failed\n\n{exc}\n\n"
            "Add a `SCOPE:` line to the PR description so the changed files "
            "can be verified against your declared intent."
        )
        return 1

    changed_files = get_changed_files()
    print(f"Declared scope: {', '.join(scope_patterns)}")
    print(f"Changed files : {len(changed_files)} file(s)")

    out_of_scope = [f for f in changed_files if not is_in_scope(f, scope_patterns)]

    if out_of_scope:
        print(f"Out-of-scope files ({len(out_of_scope)}):")
        for file_path in out_of_scope:
            print(f"  - {file_path}")

        matched = [f for f in changed_files if is_in_scope(f, scope_patterns)]
        print(f"In-scope files ({len(matched)}):")
        for file_path in matched:
            print(f"  - {file_path}")

        post_comment(
            "## Scope-check warning\n\n"
            "The following changed files are **not** covered by the declared "
            f"`SCOPE:` line (`{', '.join(scope_patterns)}`):\n\n"
            + "\n".join(f"- `{f}`" for f in out_of_scope)
            + "\n\nEither expand the `SCOPE:` line or revert these files "
              "before requesting review."
        )
        return 1

    print("All changed files are within declared scope.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
