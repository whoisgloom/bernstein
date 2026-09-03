"""The WIP/dump refusal gate at merge time.

Bug (2026-09-03, Outerloop multi-node proof): ``_save_partial_work``
(agent_lifecycle.py) stages a crashed/timed-out agent's whole worktree with
``git add -A`` and merges it under a ``[WIP] <session-id> partial work``
commit so real finished-but-uncommitted work is not lost on a kill. Observed
twice in real end-to-end runs, that unconditional ``git add -A`` also swept
up worktree-local scratch state (``.env``, lockfiles, ``.sdd/`` tool-internal
metadata, ``__pycache__``) and merged it onto the delivery branch, while the
run's own status still reported zero failures.

Each test below names the property it protects, and each names a way the
result could be wrong.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from bernstein.core.agents.spawner_merge import _partial_work_refusal


def _run(args: list[str], cwd: Path) -> None:
    subprocess.run(args, cwd=str(cwd), capture_output=True, check=True)


def _repo_with_agent_branch(
    root: Path,
    session_id: str,
    files: dict[str, str],
    *,
    commit_message: str = "agent work",
) -> None:
    """Commit ``files`` on ``agent/<session_id>``, branched off an empty main."""
    _run(["git", "init", "-b", "main"], root)
    _run(["git", "config", "user.email", "test@example.com"], root)
    _run(["git", "config", "user.name", "Test User"], root)
    _run(["git", "commit", "--allow-empty", "-m", "init"], root)

    _run(["git", "checkout", "-b", f"agent/{session_id}"], root)
    for relative, body in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    _run(["git", "add", "-A"], root)
    _run(["git", "commit", "-m", commit_message], root)
    _run(["git", "checkout", "main"], root)


def _session(session_id: str) -> Any:
    class _Stub:
        pass

    stub = _Stub()
    stub.id = session_id
    stub.task_ids = ["T-1"]
    return stub


def _refuse(root: Path, session_id: str) -> Any:
    return _partial_work_refusal(_session(session_id), root, f"agent/{session_id}")


def test_a_wip_marker_tip_commit_is_refused(tmp_path: Path) -> None:
    """The exact _save_partial_work signature is caught by subject alone."""
    _repo_with_agent_branch(
        tmp_path,
        "qa-fae05508",
        {"src/real_change.py": "x = 1\n"},
        commit_message="[WIP] qa-fae05508 partial work",
    )

    result = _refuse(tmp_path, "qa-fae05508")

    assert result is not None
    assert result.success is False
    assert "wip" in result.error.lower()


def test_a_bare_wip_marker_is_also_refused(tmp_path: Path) -> None:
    """A short-hand ``wip``/``[WIP]`` subject with no other text still matches."""
    _repo_with_agent_branch(tmp_path, "s2", {"src/x.py": "1\n"}, commit_message="wip")

    result = _refuse(tmp_path, "s2")

    assert result is not None
    assert result.success is False


def test_an_env_file_in_the_diff_is_refused_even_with_a_clean_subject(tmp_path: Path) -> None:
    """The path check catches a dump that never got a WIP-marked subject."""
    _repo_with_agent_branch(
        tmp_path,
        "s3",
        {"src/real_change.py": "x = 1\n", ".env": "SECRET=1\n"},
        commit_message="feat: add real change",
    )

    result = _refuse(tmp_path, "s3")

    assert result is not None
    assert result.success is False
    assert ".env" in result.error


def test_a_pycache_binary_in_the_diff_is_refused(tmp_path: Path) -> None:
    _repo_with_agent_branch(
        tmp_path,
        "s4",
        {"src/real_change.py": "x = 1\n", "src/__pycache__/real_change.cpython-314.pyc": "junk"},
        commit_message="feat: add real change",
    )

    result = _refuse(tmp_path, "s4")

    assert result is not None
    assert result.success is False
    assert "__pycache__" in result.error


def test_a_lockfile_in_the_diff_is_refused(tmp_path: Path) -> None:
    _repo_with_agent_branch(
        tmp_path,
        "s5",
        {"src/real_change.py": "x = 1\n", "uv.lock": "junk"},
        commit_message="feat: add real change",
    )

    result = _refuse(tmp_path, "s5")

    assert result is not None
    assert result.success is False
    assert "uv.lock" in result.error


def test_bernstein_tool_internal_state_in_the_diff_is_refused(tmp_path: Path) -> None:
    _repo_with_agent_branch(
        tmp_path,
        "s6",
        {"src/real_change.py": "x = 1\n", ".sdd/runtime/summary.md": "junk"},
        commit_message="feat: add real change",
    )

    result = _refuse(tmp_path, "s6")

    assert result is not None
    assert result.success is False


def test_a_clean_scoped_feature_commit_is_never_refused(tmp_path: Path) -> None:
    """The gate must not false-positive on the ordinary, correct case."""
    _repo_with_agent_branch(
        tmp_path,
        "s7",
        {"src/real_change.py": "x = 1\n", "tests/test_real_change.py": "def test_x(): pass\n"},
        commit_message="feat: add real change with tests",
    )

    result = _refuse(tmp_path, "s7")

    assert result is None


def test_an_env_named_config_file_that_is_not_dotenv_is_not_refused(tmp_path: Path) -> None:
    """The path pattern is anchored -- a real file named e.g. src/environment.py is not .env."""
    _repo_with_agent_branch(
        tmp_path,
        "s8",
        {"src/environment.py": "x = 1\n"},
        commit_message="feat: add environment module",
    )

    result = _refuse(tmp_path, "s8")

    assert result is None
