"""Regression tests for the completion-vs-crash-watcher race guard.

Context (Outerloop attempt-3, 2026-09-03): a slow final LLM turn on a
non-heartbeat adapter (qwen) let the orchestrator's stale/liveness watchers
decide the agent was dead from a tick-start ``tasks_snapshot`` -- while the
agent's own ``POST /tasks/{id}/complete`` was in flight. ``retry_or_fail_task``
then acted on the stale mid-flight status and destroyed a real success:
``DONE -> FAILED`` is a legal transition (the janitor-reopen edge), so the
server accepted the fail and a retry task fanned out work that had already
finished.

The guard: when the snapshot status is mid-flight, re-check the live status
at the task server before destroying anything. A terminal live status means
the completion landed inside the watcher's blind window -- skip, and let the
task's own terminal state stand.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
import pytest
from bernstein.core.task_lifecycle import retry_or_fail_task

from bernstein.core.tasks.models import Complexity, Scope, Task, TaskStatus, TaskType

_TASK_ID = "T-race"


def _task_payload(*, status: str) -> dict[str, Any]:
    """Serialised task payload mirroring the server's task response shape."""
    return {
        "id": _TASK_ID,
        "title": "Implement widget",
        "description": "Write the widget code.",
        "role": "backend",
        "priority": 2,
        "scope": "medium",
        "complexity": "medium",
        "estimated_minutes": 10,
        "status": status,
        "depends_on": [],
        "owned_files": [],
        "assigned_agent": "agent-1",
        "result_summary": None,
        "task_type": "feature",
        "model": "sonnet",
        "effort": "high",
        "retry_count": 0,
        "max_retries": 3,
        "retry_delay_s": 0.0,
        "terminal_reason": None,
        "metadata": {},
    }


def _snapshot_task(*, status: TaskStatus) -> Task:
    return Task(
        id=_TASK_ID,
        title="Implement widget",
        description="Write the widget code.",
        role="backend",
        status=status,
        scope=Scope.MEDIUM,
        complexity=Complexity.MEDIUM,
        task_type=TaskType.STANDARD,
        estimated_minutes=10,
        model="sonnet",
        effort="high",
        retry_count=0,
        max_retries=3,
    )


def _client_for(*, live_status: str | None, requests: list[tuple[str, str]] | None = None) -> httpx.Client:
    """Mock task-server client serving GET /tasks/{id} at ``live_status``."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if requests is not None:
            requests.append((request.method, path))
        if request.method == "GET" and path == f"/tasks/{_TASK_ID}":
            assert live_status is not None
            return httpx.Response(200, json=_task_payload(status=live_status))
        if request.method == "POST" and path == "/tasks":
            return httpx.Response(201, json={"id": "NEW-1"})
        if request.method == "POST" and path.endswith("/fail"):
            return httpx.Response(200, json={})
        return httpx.Response(404, json={"detail": f"no mock for {request.method} {path}"})

    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://testserver")


def _call(
    client: httpx.Client, *, retried: set[str] | None = None, snapshot_status: TaskStatus = TaskStatus.CLAIMED
) -> None:
    retry_or_fail_task(
        _TASK_ID,
        "Agent agent-1 reaped (heartbeat timeout)",
        client=client,
        server_url="http://testserver",
        max_task_retries=3,
        retried_task_ids=set() if retried is None else retried,
        tasks_snapshot={"active": [_snapshot_task(status=snapshot_status)]},
    )


# ---------------------------------------------------------------------------
# The race itself: completion landed after the snapshot was taken
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "live_status",
    ["done", "closed", "failed", "cancelled", "abandoned", "refused", "pending_approval"],
)
def test_terminal_live_status_skips_retry_and_fail(live_status: str, caplog: pytest.LogCaptureFixture) -> None:
    """A completion that landed inside the watcher's window must not be destroyed."""
    requests: list[tuple[str, str]] = []
    client = _client_for(live_status=live_status, requests=requests)
    retried: set[str] = set()

    with caplog.at_level(logging.WARNING):
        _call(client, retried=retried)

    assert ("POST", "/tasks") not in requests, "retry task must not be created for a resolved task"
    assert not any(method == "POST" and path.endswith("/fail") for method, path in requests), (
        "a terminal live status must never be overwritten by the watcher"
    )
    assert _TASK_ID not in retried, "skipped task must stay re-handleable, not marked retried"
    assert any("completion_race_guard" in record.message for record in caplog.records), (
        "the skip must be diagnosable from the log line alone"
    )


def test_in_flight_live_status_retries_normally() -> None:
    """No race: live status still mid-flight -> historical retry behaviour."""
    requests: list[tuple[str, str]] = []
    client = _client_for(live_status="claimed", requests=requests)

    _call(client)

    assert ("POST", "/tasks") in requests, "a genuinely stale task must still be retried"


# ---------------------------------------------------------------------------
# Guard boundaries
# ---------------------------------------------------------------------------


def test_terminal_snapshot_skips_the_live_probe() -> None:
    """Terminal snapshot states are their own verdict: no probe, legacy flow."""
    requests: list[tuple[str, str]] = []
    client = _client_for(live_status="claimed", requests=requests)  # GET would 200 if probed

    _call(client, snapshot_status=TaskStatus.FAILED)

    assert not any(method == "GET" for method, _ in requests), "terminal snapshots must not pay the live GET"
    assert ("POST", "/tasks") in requests


def test_probe_failure_falls_back_to_snapshot_behaviour() -> None:
    """An unreachable server must not lose the retry: proceed on the snapshot."""
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if requests is not None:
            requests.append((request.method, path))
        if request.method == "GET" and path == f"/tasks/{_TASK_ID}":
            raise httpx.ConnectError("server down mid-probe")
        if request.method == "POST" and path == "/tasks":
            return httpx.Response(201, json={"id": "NEW-1"})
        if request.method == "POST" and path.endswith("/fail"):
            return httpx.Response(200, json={})
        return httpx.Response(404, json={"detail": f"no mock for {request.method} {path}"})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://testserver")

    _call(client)

    assert ("GET", f"/tasks/{_TASK_ID}") in requests, "the live probe must have been attempted"
    assert ("POST", "/tasks") in requests, "a failed probe must not lose the retry (historical behaviour stands)"


def test_statusless_legacy_object_keeps_legacy_behavior() -> None:
    """Duck-typed task objects without ``status`` (legacy callers) are unguarded."""

    class _Scope:
        value = "medium"

    class _Complexity:
        value = "medium"

    class _TaskType:
        value = "feature"

    class _Task:  # mirrors the duck-typed doubles in test_checkpoint_retry_wiring.py
        def __init__(self, task_id: str) -> None:
            self.id = task_id
            self.title = "Test Task"
            self.description = "desc"
            self.role = "backend"
            self.priority = 1
            self.scope = _Scope()
            self.complexity = _Complexity()
            self.estimated_minutes = 10
            self.depends_on: list[str] = []
            self.owned_files: list[str] = []
            self.task_type = _TaskType()
            self.model = "sonnet"
            self.effort = "high"
            self.max_output_tokens = None
            self.max_turns = None
            self.meta_messages: list[str] = []
            self.completion_signals: list[Any] = []
            self.metadata: dict[str, Any] = {}
            self.retry_count = 0
            self.max_retries = 3
            self.retry_delay_s = 0.0

    from unittest.mock import MagicMock

    client = MagicMock(spec=httpx.Client)
    task = _Task(_TASK_ID)

    retry_or_fail_task(
        _TASK_ID,
        "agent died",
        client=client,
        server_url="http://test",
        max_task_retries=3,
        retried_task_ids=set(),
        tasks_snapshot={"active": [task]},
    )

    posted_urls = [call[0][0] for call in client.post.call_args_list]
    assert any(url.endswith("/tasks") for url in posted_urls), "legacy statusless callers keep the unguarded path"
