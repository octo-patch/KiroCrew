"""The ``crewmate.check_ins`` card and the nudge a crewmate gets when its goal is set."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from kiro_crew import change_card_catalog as catalog
from kiro_crew.dashboard import goal_setup
from kiro_crew.testing.wait import async_wait_until

SNAP = {
    "exists": True,
    "slug": "scout",
    "member_id": "m1",
    "has_goal": True,
    "schedules": {
        "daily": False,
        "weekly": False,
        "monthly": False,
        "weekday": 1,
        "month_day": 1,
        "time": "09:00",
    },
}


def _params(**over: Any) -> dict[str, Any]:
    return catalog.validate_params(catalog.KIND_CREWMATE_CHECK_INS, {"member": "scout", **over})


def test_the_card_turns_check_ins_on_and_undo_puts_them_back() -> None:
    p = _params(daily=True, weekly=True, weekday=3, time="08:30")
    preview = catalog.build_preview(catalog.KIND_CREWMATE_CHECK_INS, p, SNAP, {"label": "Scout"})
    assert preview["title"] == "Set the check-ins for “Scout”"
    rows = {r["field"]: (r["before"], r["after"]) for r in preview["changes"]}
    assert rows == {
        "daily": ("Off", "On"),
        "weekly": ("Off", "On"),
        "weekday": ("Monday", "Wednesday"),
        "time": ("09:00", "08:30"),
    }
    (step,) = preview["apply"]
    assert step["method"] == "PUT" and step["path"] == "/api/members/scout/goals/schedules"
    assert step["body"] == {
        "member": "scout",
        "daily": True,
        "weekly": True,
        "weekday": 3,
        "time": "08:30",
        "expected": {"daily": False, "weekly": False, "weekday": 1, "time": "09:00"},
    }
    plan, reason = catalog.build_undo(catalog.KIND_CREWMATE_CHECK_INS, p, SNAP, [{}], 1)
    assert reason is None
    (undo,) = plan
    assert undo["body"] == {
        "member": "scout",
        "daily": False,
        "weekly": False,
        "weekday": 1,
        "time": "09:00",
        "expected": {"daily": True, "weekly": True, "weekday": 3, "time": "08:30"},
    }


def test_the_card_refuses_what_changes_nothing_and_a_switch_with_no_goal() -> None:
    with pytest.raises(catalog.CardCatalogError) as exc:
        catalog.build_preview(catalog.KIND_CREWMATE_CHECK_INS, _params(daily=False), SNAP, {})
    assert exc.value.code == "no_change"
    with pytest.raises(catalog.CardCatalogError) as exc:
        catalog.build_preview(
            catalog.KIND_CREWMATE_CHECK_INS, _params(daily=True), {**SNAP, "has_goal": False}, {}
        )
    assert exc.value.code == "no_goal"


@pytest.mark.parametrize(
    "bad",
    [{}, {"time": "9:00"}, {"weekday": 7}, {"month_day": 29}, {"daily": "yes"}, {"other": 1}],
)
def test_the_card_validates_its_fields(bad: dict) -> None:
    with pytest.raises(catalog.CardCatalogError):
        _params(**bad)


def test_after_shows_the_write_compares_the_fields_the_card_set() -> None:
    from kiro_crew.dashboard import change_cards as cards

    p = _params(daily=True)
    on = {"schedules": {**SNAP["schedules"], "daily": True}}
    assert cards.after_shows_the_write(catalog.KIND_CREWMATE_CHECK_INS, p, on, [{}])
    assert not cards.after_shows_the_write(catalog.KIND_CREWMATE_CHECK_INS, p, SNAP, [{}])


@pytest.mark.parametrize(
    ("before", "after", "nudge"),
    [
        ("", "Keep the release notes current", True),
        ("Keep the release notes current", "Cut the build time in half", True),
        ("Keep the release notes current", "Keep the release notes curent", False),
        ("Keep the release notes current", "  keep the release   notes current ", False),
        ("Keep the release notes current", "", False),
    ],
    ids=["first-goal", "new-goal", "a-typo", "spacing-and-case", "cleared"],
)
def test_only_a_new_goal_is_worth_a_nudge(before: str, after: str, nudge: bool) -> None:
    assert goal_setup.changed_enough(before, after) is nudge


GOAL_A = {"id": "g00000001", "text": "Keep the release notes current"}
GOAL_B = {"id": "g00000002", "text": "Cut the build time in half"}


def test_the_kickoff_researches_in_the_background_then_proposes() -> None:
    rec = {
        "goals": [GOAL_A],
        "key_results": [],
        "schedules": {"daily": False, "weekly": False, "monthly": False},
    }
    text = goal_setup.kickoff_prompt(rec, GOAL_A["id"])
    assert text is not None
    assert "Keep the release notes current" in text
    assert "spawn_run" in text and "end your turn" in text
    assert "set_key_results" in text and "goal_ids" in text
    # Guidance toward a small set, not a cap.
    assert "easier to track" in text and "merging" in text
    # Check-ins wait for the milestones to be confirmed (milestones_prompt).
    assert "set_check_ins" not in text


def test_the_kickoff_names_the_goal_that_changed_and_lists_them_all() -> None:
    rec = {"goals": [GOAL_A, GOAL_B], "key_results": [], "schedules": {}}
    text = goal_setup.kickoff_prompt(rec, GOAL_B["id"])
    assert text is not None
    assert "set goal 2 (g00000002)" in text
    assert "1. Keep the release notes current" in text and "2. Cut the build time" in text
    # A goal removed before the turn runs sends nothing.
    assert goal_setup.kickoff_prompt({**rec, "goals": [GOAL_A]}, GOAL_B["id"]) is None


def test_the_milestones_nudge_offers_check_ins_only_when_none_is_on() -> None:
    rec = {
        "goals": [GOAL_A],
        "key_results": [
            {"id": "k1", "text": "notes out", "target": "yes", "goal_ids": [GOAL_A["id"]]}
        ],
        "schedules": {"daily": False, "weekly": False, "monthly": False},
    }
    off = goal_setup.milestones_prompt(rec)
    assert off is not None and "[Milestones confirmed]" in off
    assert "notes out" in off and "set_check_ins" in off
    assert "(serves goal 1)" in off and "1. Keep the release notes current" in off
    on = goal_setup.milestones_prompt({**rec, "schedules": {"weekly": True}})
    assert on is not None and "[Milestones confirmed]" in on and "set_check_ins" not in on
    assert goal_setup.milestones_prompt({**rec, "key_results": []}) is None


class _Slot:
    def __init__(self) -> None:
        from kiro_crew import members as members_mod

        self.key = "member-dm:scout"
        self.mode = members_mod.DM_SLOT_MODE
        self.agent = "scout"
        self.executor = ""
        self.running = True
        self._queue: list = []


class _State:
    def __init__(self, slot: _Slot) -> None:
        self.slot = slot

    def get_slot(self, key: str) -> _Slot | None:
        return self.slot if key == self.slot.key else None


def test_the_nudge_waits_for_a_busy_thread_then_sends_once(monkeypatch) -> None:
    from kiro_crew import member_goals
    from kiro_crew import members as members_mod
    from kiro_crew.dashboard import goal_reminder, mate_welcome

    slot = _Slot()
    state = _State(slot)
    sent: list[tuple] = []
    monkeypatch.setattr(goal_setup, "POLL_SECS", 0)
    busy_checks: list[str] = []
    real_idle_thread = goal_reminder._idle_thread

    def idle_thread(*args):
        found, skip = real_idle_thread(*args)
        if slot.running:
            assert found is None
            busy_checks.append(skip)
        return found, skip

    monkeypatch.setattr(goal_reminder, "_idle_thread", idle_thread)
    monkeypatch.setattr(
        members_mod, "read_dm_binding", lambda _slug: {"member": "scout", "slot_key": slot.key}
    )
    monkeypatch.setattr(
        member_goals,
        "read_goals",
        lambda _s, _m: {"goals": [GOAL_A, GOAL_B], "key_results": [], "schedules": {}},
    )
    monkeypatch.setattr(mate_welcome, "dispatch_hidden_turn", lambda *a, **k: sent.append((a, k)))

    async def go() -> None:
        goal_setup.schedule_nudge(state, "scout", "scout", GOAL_A["id"])
        first = state._goal_setup_tasks["scout"]
        await async_wait_until(lambda: bool(busy_checks))
        # A newer save replaces a nudge that has already examined the busy slot.
        goal_setup.schedule_nudge(state, "scout", "scout", GOAL_B["id"])
        second = state._goal_setup_tasks["scout"]
        await async_wait_until(first.done)
        assert first.cancelled()
        busy_checks.clear()
        await async_wait_until(lambda: bool(busy_checks))
        assert sent == []
        slot.running = False
        await asyncio.wait_for(second, timeout=30)
        assert state._goal_setup_tasks == {}

    asyncio.run(go())
    assert len(sent) == 1
    args, kwargs = sent[0]
    assert args[1] is slot and "[Goal set]" in args[2]
    assert "set goal 2" in args[2]
    assert kwargs == {"human_seen": True}
