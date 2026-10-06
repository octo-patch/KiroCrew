"""A project-bound cron run's text stays out of non-owner slot rows.

Notification and chat frames carrying a project-bound run's output are withheld
from non-owner sockets, but every slot-list row quotes its newest turn in
``last_message`` / ``prompt_preview``, and that list reaches every dashboard
user, SSE reader and app token. Only the owner boundary
(``include_check_status``) gets the quoted text for a ``cron-`` slot whose
newest turn is not explicitly unbound.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from kiro_crew.dashboard.state import DashboardState

SECRET = "project output nobody else should read"


@pytest.fixture(autouse=True)
def _loop():
    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield
    loop.close()
    asyncio.set_event_loop(None)


@pytest.fixture
def state(monkeypatch, tmp_path):
    monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
    st = DashboardState(
        sessions=MagicMock(count=0),
        crons=MagicMock(),
        lessons=MagicMock(),
        start_time=0.0,
    )
    monkeypatch.setattr(st, "is_yolo_active", lambda: False)
    monkeypatch.setattr(st, "_spawn_ws_send", lambda client, message: None)
    return st


def _cron_slot(state, key: str, *, project_bound: object, options: bool = False):
    slot = state.get_or_create_slot(key)
    meta = {} if project_bound is None else {"project_bound": project_bound}
    text = f"# Cron Job Result: job\n\n{SECRET}"
    if options:
        text += " [OPTIONS: ship it | hold]"
    slot.append("assistant", text, "msg msg-a", meta=meta)
    return slot


def _row(rows, key):
    return next(r for r in rows if r["key"] == key)


def _quotes(row) -> bool:
    return SECRET in row["last_message"] or SECRET in row["prompt_preview"]


def test_a_project_bound_row_is_blank_outside_the_owner_boundary(state):
    _cron_slot(state, "cron-a", project_bound=True, options=True)

    for rows in (
        state.serialize_slots(),
        state.serialize_slots(dashboard_user=True),
    ):
        row = _row(rows, "cron-a")
        assert not _quotes(row)
        assert row["options"] == [] and row["has_options"] is False

    owner = _row(state.serialize_slots(include_check_status=True), "cron-a")
    assert SECRET in owner["last_message"]
    assert owner["has_options"] is True and owner["options"]


def test_the_broadcast_views_split_the_same_way(state):
    _cron_slot(state, "cron-a", project_bound=True)
    bare, ws, owner = state.serialize_slot_views(owner=True)

    assert not _quotes(_row(bare, "cron-a"))
    assert not _quotes(_row(ws, "cron-a"))
    assert SECRET in _row(owner, "cron-a")["last_message"]
    # The derived owner view equals a full owner pass, field for field.
    assert (
        _row(owner, "cron-a")["last_message"]
        == _row(state.serialize_slots(include_check_status=True), "cron-a")["last_message"]
    )


def test_an_unstamped_cron_row_is_withheld(state):
    """Unknown provenance is not evidence of an unbound run."""
    _cron_slot(state, "cron-b", project_bound=None)
    assert not _quotes(_row(state.serialize_slots(dashboard_user=True), "cron-b"))


def test_an_explicitly_unbound_cron_row_is_unchanged(state):
    _cron_slot(state, "cron-c", project_bound=False)
    assert SECRET in _row(state.serialize_slots(dashboard_user=True), "cron-c")["last_message"]


def test_a_follow_up_in_a_project_bound_slot_is_withheld(state):
    slot = _cron_slot(state, "cron-d", project_bound=False)
    slot.append("user", f"why did it say {SECRET}?", "msg msg-u")
    assert not _quotes(_row(state.serialize_slots(dashboard_user=True), "cron-d"))


def test_an_ordinary_chat_slot_is_unchanged(state):
    slot = state.get_or_create_slot("chat-x")
    slot.append("assistant", SECRET, "msg")
    assert SECRET in _row(state.serialize_slots(), "chat-x")["last_message"]


# Source-link chips are parsed out of the same rows, so a bound run's reply that
# names a private issue surfaces it as a chip unless the chips follow the rows.
PRIVATE_ISSUE = "https://github.com/acme/private-repo/issues/42"
PUBLIC_PR = "https://github.com/acme/open-repo/pull/7"


def _linked_cron_slot(state, key: str, *, project_bound: object, url: str = PRIVATE_ISSUE):
    slot = state.get_or_create_slot(key)
    meta = {} if project_bound is None else {"project_bound": project_bound}
    slot.append("assistant", f"# Cron Job Result: job\n\nFiled {url}", "msg msg-a", meta=meta)
    return slot


def _urls(row) -> set[str]:
    return {link["url"] for link in row["source_links"]}


def test_a_project_bound_rows_links_stay_inside_the_owner_boundary(state):
    _linked_cron_slot(state, "cron-l", project_bound=True)

    for rows in (state.serialize_slots(), state.serialize_slots(dashboard_user=True)):
        row = _row(rows, "cron-l")
        assert PRIVATE_ISSUE not in _urls(row)
        assert row["source_links_total"] == 0, "the count disclosed the withheld link"

    owner = _row(state.serialize_slots(include_check_status=True), "cron-l")
    assert PRIVATE_ISSUE in _urls(owner)
    assert owner["source_links_total"] == 1


def test_the_broadcast_views_split_the_links_the_same_way(state):
    _linked_cron_slot(state, "cron-l", project_bound=True)
    bare, ws, owner = state.serialize_slot_views(owner=True)

    assert PRIVATE_ISSUE not in _urls(_row(bare, "cron-l"))
    assert PRIVATE_ISSUE not in _urls(_row(ws, "cron-l"))
    full = _row(state.serialize_slots(include_check_status=True), "cron-l")
    assert _row(owner, "cron-l")["source_links"] == full["source_links"]
    assert _row(owner, "cron-l")["source_links_total"] == full["source_links_total"]


def test_the_overflow_read_withholds_a_bound_rows_links(state):
    slot = _linked_cron_slot(state, "cron-l", project_bound=True)

    for kwargs in ({}, {"dashboard_user": True}):
        payload = slot.source_links_payload(**kwargs)
        assert payload == {"links": [], "total": 0}
    owner = slot.source_links_payload(include_check_status=True)
    assert [link["url"] for link in owner["links"]] == [PRIVATE_ISSUE]


def test_an_unstamped_rows_link_is_withheld(state):
    _linked_cron_slot(state, "cron-m", project_bound=None)
    assert PRIVATE_ISSUE not in _urls(_row(state.serialize_slots(dashboard_user=True), "cron-m"))


def test_a_later_unbound_run_does_not_vouch_for_an_earlier_bound_link(state):
    slot = _linked_cron_slot(state, "cron-n", project_bound=True)
    slot.append(
        "assistant",
        f"# Cron Job Result: job\n\nReviewed {PUBLIC_PR}",
        "msg msg-b",
        meta={"project_bound": False},
    )

    row = _row(state.serialize_slots(dashboard_user=True), "cron-n")
    assert _urls(row) == {PUBLIC_PR}
    owner = _row(state.serialize_slots(include_check_status=True), "cron-n")
    assert _urls(owner) == {PUBLIC_PR, PRIVATE_ISSUE}


def test_an_explicitly_unbound_rows_link_is_unchanged(state):
    _linked_cron_slot(state, "cron-o", project_bound=False)
    assert PRIVATE_ISSUE in _urls(_row(state.serialize_slots(dashboard_user=True), "cron-o"))


def test_an_ordinary_chat_slots_links_are_unchanged(state):
    slot = state.get_or_create_slot("chat-y")
    slot.append("assistant", f"See {PRIVATE_ISSUE}", "msg")
    assert PRIVATE_ISSUE in _urls(_row(state.serialize_slots(), "chat-y"))


# History resumes a cron transcript under whatever name the request carries:
# the pane sends the row's ``cron_<id>`` stem, and any caller may send another
# name over the job's ``cron:<id>`` key, which the slot keeps as its link. The
# guards judge the transcript, not the ``cron-`` display name.
_RESUMED_SPELLINGS = [("report", "cron:abcd1234"), ("cron_abcd1234", "")]


@pytest.mark.parametrize("key, linked", _RESUMED_SPELLINGS)
def test_a_resumed_cron_transcripts_previews_stay_inside_the_owner_boundary(state, key, linked):
    slot = _cron_slot(state, key, project_bound=True, options=True)
    slot.linked_session_key = linked
    row = _row(state.serialize_slots(dashboard_user=True), key)
    assert not _quotes(row) and row["options"] == []
    assert SECRET in _row(state.serialize_slots(include_check_status=True), key)["last_message"]


@pytest.mark.parametrize("key, linked", _RESUMED_SPELLINGS)
def test_a_resumed_cron_transcripts_links_stay_inside_the_owner_boundary(state, key, linked):
    slot = _linked_cron_slot(state, key, project_bound=True)
    slot.linked_session_key = linked
    assert PRIVATE_ISSUE not in _urls(_row(state.serialize_slots(dashboard_user=True), key))
    owner = _row(state.serialize_slots(include_check_status=True), key)
    assert PRIVATE_ISSUE in _urls(owner)
