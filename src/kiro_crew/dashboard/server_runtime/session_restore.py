"""Rebuilding the dashboard's chat slots from persisted sessions at startup."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kiro_crew.dashboard.server import (
        DashboardState,
        KiroCrewConfig,
        cautious_boot,
        channel_slots,
        chat,
        logger,
        slot_retention,
    )


async def _restore_dashboard_sessions(state: DashboardState, cfg: KiroCrewConfig) -> None:
    """Rebuild the dashboard's slots from the sessions it held, then surface channel ones.

    Restore exactly the tabs the user had open at last shutdown — these
    come back regardless of mtime, so long-running tabs don't silently
    fall off into History on every gateway restart. Closed tabs (meta.closed)
    are still excluded by the rehydrate guard. restore_open_slots() logs
    its own info line on success, so no caller-side log here.
    Awaited (not called bare) so the restore yields to the loop between tabs and
    the stall watchdog keeps getting its heartbeat — a user with many large tabs
    would otherwise block here long enough to trip the 25s watchdog and crash-loop the
    gateway before it finished starting.

    Both restores run inside suspend_slots_push() so the per-slot broadcasts
    coalesce into one at the end: get_or_create_slot() pushes the whole slot list
    on every call, which makes bulk restore O(N²) in serialization work for
    intermediate states no client renders. Reseeding happens inside the block too
    — it must complete before the single broadcast so clients never see slots
    under a counter that could still re-mint a colliding index.
    """
    # Session restores spawn a kiro-cli process per restored tab — the last
    # large group of the startup battery, so it too gets a cautious-boot window.
    await cautious_boot.pause_before("session restore")

    async def _rebuild_managers() -> None:
        """The manager rebuild, with its import inside the task that needs it.

        The import is HERE and not in the enclosing function because the enclosing
        function runs on the boot path, before readiness, and this one does not: it is
        scheduled as a task below and runs after the restore returns. Session control is
        an optional subsystem and importing it pulls its whole dependency set in, so at
        module or boot scope that cost is paid by every start -- including starts with
        session control switched off, which never reach this task at all.
        """
        from kiro_crew.dashboard import session_control as _session_control

        await _session_control.rebuild_managers_from_log(state)

    with state.suspend_slots_push():
        await chat.restore_open_slots_async(state)
        restored = await chat.restore_recent_sessions_async(
            state,
            cfg.dashboard.restore_window_minutes if cfg.dashboard.restore_sessions else 0,
            folders_only=not cfg.dashboard.restore_sessions,
        )
        if restored:
            logger.info("Restored %d session(s)", restored)

        # Both restore paths above rehydrate tabs under their original
        # "chat-<N>-<ts>" keys but leave _slot_counter at its boot value of 0.
        # Reseed it past the highest restored index so the next new chat can't
        # re-mint a colliding low index (which scrambles the tab -> session map).
        state.reseed_slot_counter()

        # A session handed to another session records its manager in its own crew log,
        # and the gateway publishes that manager only once the entry is on disk — so
        # the log is what survives a restart and a read has to bring it back.
        #
        # Only a slot the crew log's handed-over index names has anything to read: an
        # unmarked slot was decided at hydrate as held by its creator. A MARKED slot came
        # up unresolved, so session control already refuses its manager-dependent verbs
        # on its own -- no process-wide window is needed, because no slot that could be
        # guessed wrong is servable before its read lands.
        #
        # NOT AWAITED HERE. A marked slot's read is proportional to its surviving history
        # -- a unit is only PROVED to record no handoff by reading all of it -- so it runs
        # as its own task rather than in front of READY. A refusal also arms the read for
        # its own slot, so a caller that arrives first does not wait on this pass.
        _restore_managers = asyncio.create_task(_rebuild_managers())
        state._background_tasks.add(_restore_managers)
        _restore_managers.add_done_callback(state._background_tasks.discard)

    if state._dynamic_cards is not None:
        state._dynamic_cards.seed_open_sessions()

    # Archive idle tabs on a timer once the live count nears the slot cap, so a
    # dashboard nobody cleans up never reaches the point where new chats fail.
    if cfg.dashboard.idle_slot_sweep_days > 0:
        _sweep = asyncio.create_task(
            slot_retention.idle_slot_sweep_loop(state, cfg.dashboard.idle_slot_sweep_days)
        )
        state._background_tasks.add(_sweep)
        _sweep.add_done_callback(state._background_tasks.discard)

    # Surface conversations started on Slack/Discord/Teams (etc.) in the chat
    # list. These persist under channel-namespaced keys (``slack:<ts>``), which
    # neither restore path above builds slots for — without this they exist only
    # in the sidebar's collapsed History pane. Runs immediately, then on a timer
    # so a channel conversation started while the dashboard is open still shows
    # up without a restart.
    if cfg.dashboard.surface_channel_sessions:
        _chan_reconciler = asyncio.create_task(
            channel_slots.channel_slot_reconciler(state, cfg.dashboard.restore_window_minutes)
        )
        state._channel_slot_reconciler = _chan_reconciler  # prevent GC
