"""The nudge a crewmate gets right after the user adds or changes one of its goals.

Saving a goal on the Goals tab, new or edited (or applying a ``crewmate.goal`` card, which
writes through the same route) should not leave the user to work out what
milestones are or where the check-ins live. So the gateway sends the crewmate
one hidden turn in its own thread (:func:`mate_welcome.dispatch_hidden_turn`):
acknowledge the goal in a sentence, research it in a background sub-agent so
the chat stays free, and propose milestones when the research comes back.
Applying that milestones card always sends a second hidden turn
(:func:`schedule_milestones_nudge`), which offers the check-ins when none is on. Each of those is a card the user
decides; the crewmate's own goal cards are admitted outside a turn the user sent
(``handlers.guide.REQ_OWN_GOAL_TURN``).

The nudge runs only when the goal really changed: removing it, an Undo putting
an older goal back, and a small edit (a typo) send nothing. It names the goal
that changed by its number and lists every goal. When the crewmate is
busy the nudge waits for its thread to go idle, for a bounded time, and one
crewmate holds at most one waiting nudge: a newer save replaces it.
"""

from __future__ import annotations

import asyncio
import difflib
import logging
from collections.abc import Callable
from typing import Any

from kiro_crew import member_goals
from kiro_crew import members as members_mod

logger = logging.getLogger(__name__)

#: Above this similarity a new goal reads as an edit of the old one, not a new goal.
SMALL_EDIT_RATIO = 0.9
#: How long a nudge waits for a busy thread before it is dropped.
WAIT_SECS = 600.0
POLL_SECS = 3.0


def changed_enough(before: str, after: str) -> bool:
    """Whether a save from *before* to *after* is a new goal worth a nudge."""
    new = " ".join(after.split()).casefold()
    if not new:
        return False
    old = " ".join(before.split()).casefold()
    if not old:
        return True
    return difflib.SequenceMatcher(None, old, new).ratio() < SMALL_EDIT_RATIO


def kickoff_prompt(rec: dict[str, Any], goal_id: str = "") -> str | None:
    """The hidden turn's text, built from the goal record as stored.

    Names the goal *goal_id* that was just added or changed (``None`` when it is
    gone by the time the turn would run) and shows the whole list.
    """
    numbers = member_goals.goal_numbers(rec)
    if goal_id not in numbers:
        return None
    goal = rec["goals"][numbers[goal_id] - 1]
    n = numbers[goal_id]
    has_krs = bool(rec.get("key_results"))
    listing = [
        f"{i}. {member_goals.scrub_text(g['text'])}" for i, g in enumerate(rec["goals"], start=1)
    ]
    lines = [
        f"[Goal set] The user just set goal {n} ({goal_id}):",
        "",
        member_goals.scrub_text(goal["text"]),
        "",
        "All your goals now:",
        *listing,
        "",
        "Do this, in order:",
        "1. Reply in one or two short sentences, in the user's language: say back what "
        f"you understand goal {n} to mean, and that you will look into the work before "
        "you propose milestones. Ask nothing else yet.",
        "2. Start ONE background sub-agent with spawn_run to research the goal: what "
        "the work looks like today (the repo, the queue, past releases or whatever you "
        "can read), what could be measured, what is already in place. Ask it for a few "
        "candidate milestones, each something you can check and report as a "
        "number or a plain yes/no, with its target and why. Then end your turn so the "
        "chat stays free for the user; do not wait for the result.",
        "3. When the result comes back, propose the milestones with set_key_results "
        "(a card, the whole set, each milestone linked with goal_ids to the goals it "
        "serves) and a one-sentence reason."
        + (
            " You already have milestones: keep the ones that still fit, and add or "
            "change only what this goal needs."
            if has_krs
            else ""
        )
        + " When the goal cannot be measured, say so and propose none for it.",
        "4. Do not bring up check-ins yet: once the user confirms the milestones you "
        "will be told, and that is the moment for them.",
    ]
    lines.append(
        "A few goals and a few milestones are easier to track than many: keep the "
        "set small, and when goals or milestones overlap, suggest merging them; the "
        "user decides."
    )
    lines.append("Ask the user about one thing at a time.")
    return "\n".join(lines)


def _tasks(state: Any) -> dict[str, asyncio.Task[Any]]:
    tasks = getattr(state, "_goal_setup_tasks", None)
    if not isinstance(tasks, dict):
        tasks = {}
        state._goal_setup_tasks = tasks
    return tasks


async def _wait_and_send(
    state: Any, slug: str, member: str, build: Callable[[dict[str, Any]], str | None]
) -> str:
    from kiro_crew.dashboard.goal_reminder import _idle_thread
    from kiro_crew.dashboard.mate_welcome import dispatch_hidden_turn

    loop = asyncio.get_running_loop()
    deadline = loop.time() + WAIT_SECS
    while True:
        binding = await asyncio.to_thread(members_mod.read_dm_binding, slug)
        slot_key = binding.get("slot_key", "") if binding else ""
        slot, skip = _idle_thread(state, slot_key, member)
        if slot is not None:
            try:
                rec = await asyncio.to_thread(member_goals.read_goals, slug, member)
            except member_goals.GoalsUnreadable:
                return "skipped: goal record unreadable"
            prompt = build(rec)
            if prompt is None:
                return "skipped: nothing left to talk about"
            # The read ran off the loop: a turn may have started meanwhile.
            again, skip = _idle_thread(state, slot_key, member)
            if again is slot:
                dispatch_hidden_turn(state, slot, prompt, human_seen=True)
                return "started"
        if skip == "skipped: the crewmate's thread is not open" or loop.time() >= deadline:
            return skip or "skipped: timed out"
        await asyncio.sleep(POLL_SECS)


def _goal_prompt(goal_id: str) -> Callable[[dict[str, Any]], str | None]:
    return lambda rec: kickoff_prompt(rec, goal_id)


def milestones_prompt(rec: dict[str, Any]) -> str | None:
    """The hidden turn after the user confirms the crewmate's milestones card."""
    krs = rec.get("key_results") or []
    if not krs:
        return None
    sched = rec.get("schedules") or {}
    any_on = any(sched.get(c) for c in member_goals.CADENCES)
    lines = ["[Milestones confirmed] The user just confirmed your milestones:", ""]
    for kr in krs:
        text = member_goals.scrub_text(str(kr.get("text") or ""))
        target = member_goals.scrub_text(str(kr.get("target") or ""))
        serves = member_goals.serves_note(rec, kr)
        lines.append(f"- {text} (aiming for: {target}){serves}" if target else f"- {text}{serves}")
    if rec.get("goals"):
        lines.extend(
            [
                "",
                *member_goals.goal_lines(
                    {
                        "goals": [
                            {**g, "text": member_goals.scrub_text(g["text"])} for g in rec["goals"]
                        ]
                    }
                ),
            ]
        )
    lines.append("")
    if any_on:
        lines.append(
            "Reply in one or two short sentences, in the user's language: acknowledge "
            "them and say what you will work on first toward the goal. The check-ins "
            "already running will record readings against these milestones."
        )
    else:
        lines.append(
            "Reply in the user's language: acknowledge them in a sentence, then offer "
            "check-ins with set_check_ins. Say in one plain sentence each what the Daily "
            "Goal Reminder, the Weekly report and the Monthly review would do for them, "
            "propose the ones that fit this goal, and let the card decide."
        )
    return "\n".join(lines)


def _schedule(
    state: Any,
    key: str,
    slug: str,
    member: str,
    build: Callable[[dict[str, Any]], str | None],
    what: str,
) -> None:
    if state is None:
        return
    tasks = _tasks(state)
    old = tasks.get(key)
    if old is not None and not old.done():
        old.cancel()

    async def run() -> None:
        try:
            outcome = await _wait_and_send(state, slug, member, build)
            logger.info("%s nudge for %s: %s", what, slug, outcome)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("%s nudge for %s failed", what, slug, exc_info=True)
        finally:
            if tasks.get(key) is task:
                tasks.pop(key, None)

    task = asyncio.get_running_loop().create_task(run())
    tasks[key] = task


def schedule_nudge(state: Any, slug: str, member: str, goal_id: str) -> None:
    """Send *member* the goal-setup nudge for *goal_id* once its thread is idle.

    One waiting nudge per crewmate: a newer save, of this goal or another,
    replaces it.
    """
    _schedule(state, slug, slug, member, _goal_prompt(goal_id), "goal setup")


def schedule_milestones_nudge(state: Any, slug: str, member: str) -> None:
    """Tell *member* the user confirmed its milestones, once its thread is idle."""
    _schedule(state, f"{slug}:milestones", slug, member, milestones_prompt, "milestones confirmed")
