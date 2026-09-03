"""A claimed task with a dead agent must not let quiescence declare "complete".

Bug (2026-09-03, Outerloop multi-node proof): the tick-level quiescence check
in ``Orchestrator.tick`` (step 8b) only reads ``open_tasks``/``active_agents``
counts, never the "claimed" bucket. A task whose agent died - a crash, or the
stalled-manager/idle-log-age watchdog killing it mid-run - can sit at
"claimed" forever if ``_release_stale_claims`` has not yet run on its own
periodic ``_run_normal`` cadence. Observed twice in real end-to-end runs: the
orchestrator declared the run quiescent, wrote a run-summary with that task
excluded from both the done and failed counts, and notified "run.completed"
while the task never reached a terminal state.

The fix makes ``Orchestrator.tick`` call ``_release_stale_claims`` on the
full claimed bucket unconditionally, right before the quiescence decision, so
a dead-agent claim is always reconciled before "complete" can be declared.
This test pins the piece that fix depends on: ``_release_stale_claims``
itself must reclaim a claimed task whose agent is confirmed dead, immediately
and every time it is called - not just on the periodic cadence.

Each test below names the property it protects, and each names a way the
result could be wrong.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from bernstein.core.orchestration.orchestrator import Orchestrator


def _task(task_id: str, *, claimed_at: float = 0.0, created_at: float = 0.0) -> Any:
    class _Task:
        pass

    task = _Task()
    task.id = task_id
    task.title = f"task {task_id}"
    task.claimed_at = claimed_at
    task.created_at = created_at
    return task


def _stub_orchestrator(
    *,
    task_to_session: dict[str, str],
    agents: dict[str, Any],
    stale_claim_timeout_s: float = 900.0,
) -> Any:
    """A duck-typed stand-in carrying only what ``_release_stale_claims`` reads.

    Building a real ``Orchestrator`` needs a live task-server client, a
    workdir, and a full config -- this method only ever reads
    ``self._config``, ``self._task_to_session``, and ``self._agents``, and
    calls ``self._retry_or_fail_task``, so a duck-typed stub carrying exactly
    those is what the method under test actually depends on.
    """

    class _Config:
        pass

    config = _Config()
    config.stale_claim_timeout_s = stale_claim_timeout_s

    stub = MagicMock()
    stub._config = config
    stub._task_to_session = task_to_session
    stub._agents = agents
    return stub


def test_a_claimed_task_with_a_confirmed_dead_agent_is_reclaimed_immediately() -> None:
    """The correctness-critical branch: no timeout wait for a dead agent."""

    class _Agent:
        status = "dead"

    stub = _stub_orchestrator(
        task_to_session={"T-1": "agent-1"},
        agents={"agent-1": _Agent()},
    )

    released = Orchestrator._release_stale_claims(stub, [_task("T-1")])

    assert released == 1
    stub._retry_or_fail_task.assert_called_once()
    called_task_id = stub._retry_or_fail_task.call_args.args[0]
    assert called_task_id == "T-1"


def test_a_claimed_task_with_no_tracked_agent_at_all_is_also_reclaimed_immediately() -> None:
    """An agent entry that was never tracked is exactly as dead as one marked dead."""
    stub = _stub_orchestrator(task_to_session={}, agents={})

    released = Orchestrator._release_stale_claims(stub, [_task("T-2")])

    assert released == 1
    stub._retry_or_fail_task.assert_called_once()


def test_a_claimed_task_with_a_live_agent_under_the_timeout_is_left_alone() -> None:
    """A live agent still working must not be reclaimed out from under it."""
    import time

    class _Agent:
        status = "running"

    now = time.time()
    stub = _stub_orchestrator(
        task_to_session={"T-3": "agent-3"},
        agents={"agent-3": _Agent()},
        stale_claim_timeout_s=900.0,
    )

    released = Orchestrator._release_stale_claims(stub, [_task("T-3", claimed_at=now)])

    assert released == 0
    stub._retry_or_fail_task.assert_not_called()
