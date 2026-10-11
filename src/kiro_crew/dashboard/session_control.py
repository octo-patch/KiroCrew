"""Session control: letting one chat session observe and interrupt another.

Four operations — create a session, stop its turn, close (archive) it, and read
its transcript — plus the authorization that decides whether a caller may address
a target at all. The operations are deliberately thin: they reuse the same
creation, stop, close and history paths the dashboard itself uses, so a controlled
session behaves exactly like one a human is typing into.

**Two verbs here write into another session's conversation: ``session_send`` and
``session_broadcast``.**
Reading returns a transcript tail; stopping cancels an in-flight turn the way the
Stop button does; closing archives the session the way the tab ✕ does (the
conversation is saved to history and can be reopened — closing is not deletion);
creating opens an empty session in the user's sidebar; sending
delivers a message the target runs as its next turn, redacted through
``sanitize_outbound`` and prefixed with a ``[sent by session … via <verb>]``
envelope so it can never render as something the person typed — and the verb is
named in it, so a worker can tell a message addressed to it alone from one every
sibling also received. Broadcasting is not a second write path: it resolves an
audience and hands each target to the same delivery, so every bound below holds
per target unchanged. An IDLE target runs
it under the authorization that admitted it; a BUSY target queues it, and the
generic drain re-asserts the target-side containment before the entry becomes a
turn: producers stamp the constraints that held at admission
(:func:`containment_meta`), and ``chat_runner``'s drain drops — with a visible
notice and an SEL record — any entry for which a constraint holds at delivery
that did not hold at admission. A dropped CROSS-SESSION delivery is reported back
to its sender too, since the target's own notice is on a transcript the sender
does not read: the entry also carries the sending session, as a slot key plus
that slot's tab identity (:func:`send_origin_meta`), and the drop appends a
notice there (:func:`notify_send_origin_dropped`). The two deliberate discards
of a queued entry, a person cancelling its queue card and a hard stop clearing
the queue, report back the same way with their own wording
(:func:`notify_send_origin_discarded`). A human-typed queued message
shares the same window and the same re-check, and carries no sender to report to.

Authorization is deny-by-default and checked in one place
(:func:`authorize_target`) for the three operations that take a target — stop,
close and read — so a guard cannot be present on one verb and missing on another.
``session_create`` has no target; it checks the caller's own eligibility with the
same refusals.
"""

from __future__ import annotations

import asyncio
import json
import logging
import unicodedata
import uuid
import weakref
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, NamedTuple

from kiro_crew import model_registry
from kiro_crew.agent_sdk.backends import ACP_BACKENDS_MODEL_EFFORT_PAIR_IDS
from kiro_crew.agent_sdk.capabilities import MODEL_NAMESPACE_ACP, capabilities_for
from kiro_crew.agent_sdk.provider_identity import is_claude_code
from kiro_crew.config.loader import (
    KiroCrewConfig,
    _workspace_name_for_dir,
    default_project_dir,
    resolve_agent_bindings,
)
from kiro_crew.config.resolution import DEGRADED_WHOLE_CONFIG
from kiro_crew.crew_log import emit as crew_log_emit
from kiro_crew.crew_log.session_tree_projection import projection
from kiro_crew.dashboard.chat_delivery import sanitize_outbound
from kiro_crew.dashboard.chat_folders import (
    _folder_declared_project,
    _resolve_folder_project_dir,
    _slot_meta_txn_lock,
    _unhide_folder,
    note_folder_filed,
)
from kiro_crew.dashboard.chat_fork import (
    _FORK_DIRECTION_HEAD,
    ForkResult,
    ForkSource,
    fork_slot,
    resolve_fork_source,
)
from kiro_crew.dashboard.chat_persistence import _TRANSIENT_ROLES as _PERSISTENCE_TRANSIENT_ROLES
from kiro_crew.dashboard.chat_persistence import (
    _recent_session_slot_name,
    save_slot_off_loop,
)
from kiro_crew.dashboard.chat_title import (
    apply_manual_title,
    apply_regenerated_title,
    generate_manual_title,
)
from kiro_crew.dashboard.chat_utils import (
    _history_key_for,
    _normalize_model,
    drained_to_thread,
    effective_session_key,
    slot_history_key,
)
from kiro_crew.dashboard.create_rate_limit import (
    GENERATE_TITLE,
    SESSION_CREATE,
    WINDOW_SECS,
    allow_create,
    has_create_budget,
)
from kiro_crew.dashboard.handlers._shared import _scrub_text
from kiro_crew.dashboard.relay_archive import (
    RELAY_ARCHIVE_CODE,
    RELAY_ARCHIVE_ERROR,
    is_relay_archive,
)
from kiro_crew.dashboard.state import (
    MAX_LIVE_SLOTS,
    MAX_SLOTS_PER_CREATOR,
    SlotOrigin,
    _normalize_slot_key,
    _safe_folder_tree,
    _SLOT_KEY_MAX_CHARS,
    load_window_off_loop,
)
from kiro_crew.dashboard.stop_retry import allow_escalation
from kiro_crew.effort import EFFORT_LEVELS, is_valid_effort
from kiro_crew.execution_context import (
    ExecutionContext,
    MemoryStoreRef,
    bind_session_execution,
    read_session_execution,
    read_vouched_session_execution,
    refresh_vouched_session_execution,
    resolve_member_execution,
    revouch_at_verified_admission,
)
from kiro_crew.history import metadata_now_iso, transcript_stem
from kiro_crew.llm_helpers import slot_switch_session_lock
from kiro_crew.members import select_provider_backend
from kiro_crew.memory_stores import named_store_or_empty
from kiro_crew.messaging.link import CHAT_TYPE_DIRECT, ChannelLink, parse_session_key
from kiro_crew.messaging.transport import DM_TARGET_PREFIX, sole_direct_target
from kiro_crew.metrics.sessions import END_REASON_REMOVED
from kiro_crew.metrics.turns import turn_outcome
from kiro_crew.security import redact, redact_and_truncate
from kiro_crew.sel import sel
from kiro_crew.session_summary import derive_state
from kiro_crew.terminal_safe import _is_invisible  # one shared list of invisible code points
from kiro_crew.validation import (
    _MODEL_NAME_RE,
    BROADCAST_TARGET_ALLOWANCE_SECS,
    MAX_ACP_SESSION_ID_LEN,
    MAX_BROADCAST_TARGETS,
    MAX_LONG_STRING,
    MAX_SESSION_STATUS_ROWS,
    MAX_SESSION_STATUS_TITLE_CHARS,
    MAX_SHORT_STRING,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kiro_crew.dashboard.state import DashboardState, _ChatSlot

logger = logging.getLogger(__name__)

# Reads are cheap but not free — each one walks the target's in-memory window.
MAX_READ_MESSAGES = 100
DEFAULT_READ_MESSAGES = 20

# Per-message content cap for reads, so pulling a transcript tail cannot return
# a multi-megabyte tool payload verbatim.
MAX_READ_CONTENT_CHARS = 4000

# Cap on the tool title a read or status row names as waiting for approval.
MAX_PENDING_APPROVAL_TOOL_CHARS = 200

# Bounds on a ``read_summary`` response. The stored summary caps intent and note
# COUNTS at 50 but neither per-intent list lengths nor string lengths, so the
# response is cut here, once, and says how much it left out. Intents arrive
# most-recently-touched first, so the intent cut drops the oldest.
MAX_SUMMARY_INTENTS = 10
MAX_SUMMARY_ITEMS = 5
MAX_SUMMARY_NOTES = 10
MAX_SUMMARY_CHARS = 500

# A cron run's own slot (``cron-<job_id>``, minted by ``inject_cron_result_to_dashboard``).
CRON_SLOT_PREFIX = "cron-"

# A background workflow's result slot (``workflow-<run_id>``, minted by
# ``workflow_inject`` only when the originating tab is gone).
WORKFLOW_SLOT_PREFIX = "workflow-"

# Slot-key prefixes for sessions no human is watching. As a TARGET both are
# always refused: a message would start a fresh agent turn in a display-only slot
# nobody reads.
#
# As a SOURCE the two differ, and the difference is ownership. What must not
# happen is a scheduled job reaching the user's OWN conversations, which is a
# question about scope rather than about attendance -- an unattended job already
# starts turns in the session that owns it every time it delivers with
# ``send_message(session="origin")``. A cron slot can be held to that scope,
# because `authorize_target` fences it to slots carrying its own
# ``_created_by`` (see :func:`_cron_caller`), so it is admitted as a source. A
# workflow result slot cannot: it exists only when the originating tab is already
# gone, so there is no owning session to fence it to and nothing it would
# legitimately dispatch. It stays refused.
#
# Membership here is the fail direction for a prefix added later: a new
# unattended surface is refused as a source until it is given a fence of its own.
UNATTENDED_SLOT_PREFIXES = (CRON_SLOT_PREFIX, WORKFLOW_SLOT_PREFIX)

# The ``linked_session_key`` a cron slot carries (``cron:<job_id>``), written by
# ``inject_cron_result_to_dashboard`` to bind the run's transcript to the tab.
# It is NOT a channel link, and the caller-side linked refusals exist for channel
# links: they keep a session whose conversation is mirrored to Slack/Telegram from
# reading a peer's transcript into that thread. A cron link mirrors nothing and
# has no audience, so it is exempted where the caller is judged. The TARGET-side
# refusal is deliberately not exempted -- a cron tab is already unreachable as a
# target by prefix, and the exemption would otherwise widen to every linked slot.
CRON_LINK_PREFIX = "cron:"

# The ``created_by`` tag an app's own cron job carries (``app:<app_name>``,
# written by ``apps/cron_sdk.py``). Spelled here rather than imported: this
# module sits below the apps package, and the value is a persisted data format
# rather than something that package exports.
APP_CRON_OWNER_PREFIX = "app:"

# The channels whose 1:1 DM session can be recognised as the configured owner's
# own conversation by :func:`owner_dm_refusal`. Membership asserts three facts
# that were VERIFIED against the transport, and a surface is added only by
# verifying all of them again for it:
#
# * the dispatcher mints its DM key as ``{surface}:{agent}:direct:{peer}``
#   (``build_dm_session_key`` with ``chat_type=direct``), so the key names the one
#   human in the conversation and a thread, group, forum or unified key does not
#   parse as one;
# * ``configured_targets()`` advertises exactly that peer as ``user:{peer}`` and
#   draws from CONFIGURED state alone -- never from identities learned off inbound
#   traffic, which is the gap ``constants.CHANNEL_OWNER_DM_NAMESPACES`` names for
#   Weixin and WeCom and the reason this set is a subset of it; and
# * the transport attests, from its own state, which person a DM conversation it
#   opened or received a message from belongs to (``direct_peer_of``), so a
#   dashboard-born slot's persisted mirror row -- a conversation id, never a
#   principal -- can be placed as the owner's DM or refused. Discord answers from
#   the ``create_dm_channel`` / inbound-DM pairing its client records; Telegram's
#   private ``chat_id`` is the peer.
#
# Every other channel fails closed here and keeps its full containment.
OWNER_DM_CONDUCTOR_SURFACES: frozenset[str] = frozenset({"discord", "telegram"})


def _member_caller(state: "DashboardState", caller_key: str) -> bool:
    """Whether *caller_key* is a crew member acting through one of its slots.

    A crew member runs in TWO kinds of slot, and both are the member operating
    model rather than an optional capability, so the surface authorizes either
    WITHOUT the global ``agent.session_control`` opt-in. What bounds them
    instead is ownership: :func:`authorize_target` restricts a member caller to
    slots it created itself, so the automatic grant never reaches the user's
    own sessions.

    * (a) a pinned DM slot, keyed ``member-<slug>`` — recognised by the members
      module's own prefix constant (imported lazily, since ``members`` imports
      ``validation`` which sits below this module in the layering) rather than a
      restated literal, so the two cannot drift; and
    * (b) an ORDINARY dashboard chat slot (``chat-<n>-<ts>``) whose bound memory
      store is that member's private V2 store — the same store a DM slot would
      be bound to. A member's whole operating model (``session_create`` /
      ``session_send`` / ``session_read_message`` / ``session_stop`` /
      ``session_close``) also runs from such a chat slot, so refusing it there
      would leave the member chat-only in the surface it exists to drive. The
      store, not the key, is the member's identity here: it is what
      :func:`_store_is_member_owned` reads off the config record.

    Case (b) needs the caller's slot to read its bound store, hence *state*;
    ``member_dispatch`` gates the bypass either way (:func:`_member_bypass`).

    This predicate decides the switch BYPASS, re-read live at every gate so a
    member the operator un-assigns loses it at once — a change that can only
    tighten. It is not what keeps an admitted member creator-FENCED: the config
    record case (b) reads is mutable, so the HTTP gate carries its verified
    admission into :func:`authorize_target` (``precomputed_ownership_fenced``)
    rather than letting the fence re-derive it here a beat later.
    """
    from kiro_crew.members import DM_SLOT_KEY_PREFIX

    if caller_key.startswith(DM_SLOT_KEY_PREFIX):
        return True
    slot = state.get_slot(caller_key)
    if slot is None:
        return False
    return _store_is_member_owned(getattr(slot, "memory_store", "") or "")


def _store_is_member_owned(store: str) -> bool:
    """Whether *store* is a crew member's private V2 memory store, right now.

    The ONE predicate that recognises a member store, shared by the HTTP gate
    (``handlers/session_control.py``'s ``_private_caller_refusal``) and the inner
    switch bypass here (:func:`_member_caller` case (b)), so the two layers cannot
    disagree on what a member store is. It answers from the CONFIG RECORD, never
    the on-disk ownership manifest: it runs at :func:`authorize_target`'s
    synchronous gate, where a manifest ``stat`` / ``read`` would be blocking IO on
    the event loop. ``KiroCrewConfig.load()`` is the cached read the switch gate
    beside it already performs (warmed by :func:`prewarm_enabled_check`), and the
    fields it reads are in-memory attributes of the loaded record.

    ``True`` requires ALL of: a record for *store*; ``memory_version == 2``; a
    non-empty ``owner_member``; and that owner still an ACTIVE agent bound to
    exactly this store (the live-binding test ``active_member_memory_stores``
    applies). A crew can be deleted while a chat slot bound to its store is still
    live — the store record is retained with ``owner_member`` set but the agent is
    gone from ``cfg.agents`` — and a retired owner must not keep the switch bypass
    past a governance switch the operator turned off.

    Everything else is ``False``, and every ``False`` is FAIL-CLOSED for what this
    predicate decides — admission and the switch bypass: ``default``/empty, a
    missing record, a non-V2 store, an ownerless V2 store, a retired or re-bound
    owner, an unreadable config, or a degraded ``memory_stores`` section all
    withhold member status, and a caller then falls back under the global switch
    like any other. Withdrawing the case-(b) admission can never open the surface
    wider than it is.

    It is deliberately NOT the ownership FENCE's source of truth. The record is
    mutable — an operator's own config writer can un-assign the member, drop
    ``memory_version`` (the loader coerces a missing key to ``1``), or drop the
    entry outright — and any of those can land between the HTTP gate's admission
    and the inner authorization. The fence therefore does not re-derive member
    status from this record: the gate carries its VERIFIED admission into
    :func:`authorize_target` as ``precomputed_ownership_fenced`` (see
    ``handlers/session_control.py``), so a member admitted as one stays
    creator-fenced for the whole request whatever the record says a beat later.
    """
    if not store or store == "default":
        return False
    try:
        cfg = KiroCrewConfig.load()
    except Exception:
        logger.warning(
            "session_control: config read failed — store %r is not treated as a member store",
            store,
            exc_info=True,
        )
        return False
    if cfg.degraded_sections & {DEGRADED_WHOLE_CONFIG, "memory_stores"}:
        logger.warning(
            "session_control: memory_stores config section degraded — store %r is not "
            "treated as a member store",
            store,
        )
        return False
    record = cfg.memory_stores.get(store)
    if record is None or getattr(record, "memory_version", 1) != 2:
        return False
    owner = getattr(record, "owner_member", "")
    if not owner:
        return False
    owners_bound_here = [
        member
        for member, agent in cfg.agents.items()
        if getattr(agent, "memory_store", None) == store
    ]
    return owners_bound_here == [owner]


def _cron_caller(caller_key: str) -> bool:
    """Whether *caller_key* is a cron job's own slot.

    A cron caller is admitted to the surface DESPITE being unattended, and is
    bounded the same way a crew member is: :func:`authorize_target` refuses it on
    any slot it did not create itself, so its reach covers the sessions it
    dispatched and never the user's own conversations.

    It differs from a member caller in one way that matters. A member bypasses
    the global ``agent.session_control`` switch, because dispatching into workers
    is the member operating model rather than an opt-in. A cron does NOT: the
    switch is the user's statement that agents may open and drive sessions at
    all, and a job running while they are asleep is the last caller that should
    be exempt from it.

    Keyed on the slot-key prefix rather than on the slot's ``linked_session_key``
    or its ``SlotOrigin``, so the answer is available before the slot is resolved
    and cannot change under a caller: a slot key is immutable, while both of the
    others are fields a later write could alter.
    """
    return caller_key.startswith(CRON_SLOT_PREFIX)


def _caller_is_ownership_fenced(state: "DashboardState", caller_key: str) -> bool:
    """Whether *caller_key* may only reach slots it created itself.

    Four populations, one predicate, so the fence and the admissions that depend
    on it cannot drift apart:

    * a crew member's DM slot, which bypasses the config switch;
    * a cron job's own slot, which bypasses the unattended refusal;
    * a channel-born slot admitted as the owner's own DM (:func:`owner_dm_refusal`
      answering ``""``), which bypasses the channel-link refusals. Every non-cron
      link is fenced, not only the admitted ones: the only linked caller that gets
      past those refusals is an owner DM, and fencing on the link rather than on
      the admission keeps the fence readable without the transport roster or the
      session store. What it buys is the bound on a wrong audience inference: the
      DM reaches the workers it dispatched and never the person's own tabs, the
      same reach a crew member has. A DASHBOARD-born slot the same predicate
      admits on its outbound mirror is deliberately not a fifth population: the
      fence follows authority, and that slot's authority is the person's own
      dashboard session -- the words that drive it are typed into the dashboard by
      the authenticated owner, and the mirror the predicate admitted changes who
      READS them (the owner, alone) and not who authored them. Fencing it on the
      mirror would take the owner's own reach away for the price of linking a tab
      to their phone, which is the one thing the admission exists to allow. When
      an AGENT made that slot it is fenced anyway, by the mark below;
    * **anything any of them created**, which is the part a key prefix cannot
      see. A created child is minted with a plain ``chat-`` key and INHERITS the
      creator's agent, so a fenced caller running a session-control agent would
      otherwise get an unfenced deputy for free: create a child, seed it, and the
      child -- an ordinary caller by key -- reads any same-workspace session and
      reports back through the transcript its creator is allowed to read. The
      fence has to follow authority, not spelling.

    ``_created_by`` is the marker for that last population and needs no lineage
    walk: :func:`create_session` is its ONLY writer (a person's own tab and a fork
    reach ``get_or_create_slot`` directly and stay unattributed), so a non-empty
    value means "an agent made this session" at any depth. An agent-created
    session is fenced regardless of which population its creator belonged to: the
    predicate is read for every verb through :func:`authorize_target`, so it
    answers "whose authority is this session" from the slot's own stamp alone and
    never widens on a creator that happened to be unfenced. The owner-rooted
    private-member dispatch that :func:`create_session` must still permit is
    decided at that gate (see :func:`_delegation_lineage_fenced`), not here, so
    this predicate's containment of every agent-created session stays intact.

    There is deliberately NO attendance exemption. ``_ChatSlot._human_seen`` looks
    like the right hatch and is not: it records that a human has EVER driven the
    slot, is monotonic and persisted, and says nothing about who authored the turn
    running now. Releasing the fence on it would hand the creator its deputy back
    for the price of the user glancing at the tab once -- cron creates the child,
    the user types into it, and from then on every cron-authored turn in that child
    runs unfenced. The question this predicate can answer is "whose authority is
    this session", not "is a person at the keyboard", so a person working in an
    agent-created session keeps that session's reach rather than their own.
    """
    if _member_caller(state, caller_key) or _cron_caller(caller_key):
        return True
    slot = state.get_slot(caller_key)
    if slot is None:
        return False
    if _channel_link_of(slot):
        return True
    return bool(getattr(slot, "_created_by", ""))


#: How far either lineage walk climbs before the chain is treated as corrupt, for the
#: creation chain (:func:`_delegation_lineage_fenced`) and the manager line
#: (the fence's own read) alike. A real chain is at most as deep as the live
#: slots above a session, so it is short; the bound is here so a cycle written by a
#: corrupted record cannot spin, and a chain this long fails closed as a gap. ONE
#: spelling, because the two walks bound the same thing for the same reason and a
#: second number would drift from this one the first time either moved.
LINEAGE_WALK_MAX_DEPTH = 64


def _delegation_lineage_fenced(state: "DashboardState", caller_key: str) -> bool:
    """Whether *caller_key* may NOT dispatch a private-member worker.

    The private-member delegation gate in :func:`create_session` needs the one
    thing :func:`_caller_is_ownership_fenced` deliberately does not give it: a
    conductor the owner started in their own tab must be allowed to mint private
    workers, while a conductor rooted in a cron, channel link or crew member must
    not. The shared predicate answers "every agent-created session is fenced" for
    the ownership boundary read on every verb, and that answer must stay intact;
    this walk is read ONLY here, at the once-per-create delegation gate, so it
    never widens :func:`authorize_target`.

    It climbs the ``_created_by`` chain LIVE at each hop -- a creator that has
    since become a crew member, acquired a channel link, or is a cron tab fences
    the whole chain the moment it does, so a mid-chain takeover cannot leave a
    stale "unfenced" behind (there is no frozen verdict to go stale). The root of
    an owner-rooted chain is a person's own unattributed tab, which reaches
    ``get_or_create_slot`` directly and carries no ``_created_by`` and no fence
    source, so the walk ends unfenced. It fails CLOSED on any gap: a hop whose
    creator slot is gone, or a chain longer than the depth bound, is fenced, so a
    chain whose middle slot was closed loses dispatch rather than widening.
    """
    seen: set[str] = set()
    key = caller_key
    # The chain is at most as deep as live slots, but a bound keeps a corrupted
    # ``_created_by`` cycle from spinning; any chain this long is treated as a
    # gap and fails closed.
    for _ in range(LINEAGE_WALK_MAX_DEPTH):
        if key in seen:
            return True
        seen.add(key)
        if _member_caller(state, key) or _cron_caller(key):
            return True
        slot = state.get_slot(key)
        if slot is None:
            # Mid-chain creator gone: cannot prove the root is the owner's, so
            # fail closed rather than treat an unreadable ancestor as unfenced.
            return key != caller_key
        if _channel_link_of(slot):
            return True
        parent = getattr(slot, "_created_by", "")
        if not parent:
            # Unattributed root -- a person's own tab or a fork. Owner-rooted:
            # the one chain the delegation gate exists to permit.
            return False
        key = parent
    return True


def _channel_link_of(slot: Any) -> str:
    """*slot*'s channel link, or ``""`` for an unlinked slot and for a cron tab.

    The one reading of "this slot is channel-born" the caller-side gates share: a
    ``cron:<job_id>`` link names the job's own run transcript and republishes to
    nobody, so it is not a channel link (see ``CRON_LINK_PREFIX``).
    """
    link = str(getattr(slot, "linked_session_key", "") or "")
    if not link or link.startswith(CRON_LINK_PREFIX):
        return ""
    return link


def _created_by_other(slot: Any, caller_key: str) -> bool:
    """Whether *slot* was not created by *caller_key*.

    Fail-closed on an unowned slot, which is what an ownerless rehydrate looks like:
    a blank ``_created_by`` matches no caller, so a fenced caller does not reach it.
    """
    return getattr(slot, "_created_by", "") != caller_key


def manager_of(slot: Any) -> str:
    """The slot key that holds *slot* now: its ``_manager``, else its creator.

    ONE reading of "who manages this session", shared by the fence and by the handoff
    verb's own caller rule, so the two cannot drift. An empty ``_manager`` is not a
    missing answer: it means nobody has handed this session over, and the session is
    held by whoever created it. That is what makes the field free to add -- on a
    gateway where no handoff has happened every caller of this function gets
    ``_created_by``, which is the value the fence read before the field existed.

    An attribute read and nothing else, with no I/O and no suspension, because the
    fence that calls it runs inside ``close_target``'s no-suspension window.

    Fail-closed at the edges for the reason :func:`_created_by_other` is: a slot with
    neither field answers ``""``, and ``""`` is not a slot key any caller can hold.

    THE RAW READ, and it answers even when the answer is not known yet -- an unresolved
    slot's empty ``_manager`` reads through to the creator here. Only the REBUILD may
    use it that way, because the rebuild is what makes the value known. Every reader
    that AUTHORIZES against it calls :func:`require_manager_of`, which refuses instead
    of guessing, and ``test_session_handoff.py`` pins that split from source.
    """
    return str(getattr(slot, "_manager", "") or "") or str(getattr(slot, "_created_by", "") or "")


#: How much of a session's title a refusal will quote. A title is author-written and
#: unbounded; a refusal is one line a person reads. Long ones are cut with an ellipsis
#: rather than wrapped, because the part that identifies the session is the front.
_NAME_FOR_PERSON_MAX = 48


def name_for_person(state: "DashboardState", slot_or_key: Any) -> str:
    """A session named the way a person recognises it: its title, else its key.

    Every refusal this module hands back is read by somebody -- in a tab, in an agent's
    transcript, in a notice. ``chat-1760244`` identifies a session to the gateway and to
    nobody else, so a refusal that names only the key makes the reader go and look up
    which session it meant before they can act on what they were told.

    Takes either a slot or a bare key, because both are what the call sites hold: the
    handoff verb has the slot it resolved, and it has the receiver and the replaced
    manager as keys alone. A key that names no live slot answers the key -- there is
    nothing better to say about a session this gateway is not holding, and inventing a
    placeholder would read as a title.

    The key is the FALLBACK and never dropped, so a session whose title is empty, blank
    or still being generated is still identified rather than named "". Quoting is the
    caller's business: these go into sentences that already quote.
    """
    slot = state.get_slot(slot_or_key) if isinstance(slot_or_key, str) else slot_or_key
    key = str(getattr(slot, "key", "") or "") or (
        slot_or_key if isinstance(slot_or_key, str) else ""
    )
    title = str(getattr(slot, "title", "") or "").strip()
    if not title:
        return key
    if len(title) > _NAME_FOR_PERSON_MAX:
        title = title[: _NAME_FOR_PERSON_MAX - 1].rstrip() + "\u2026"
    return title


def _tab_identity(state: "DashboardState", slot_key: str) -> str:
    """The ``_tab_id`` of the session open under *slot_key*, or ``""``.

    One reader for the occupancy a manager record carries, so the publish and the
    check cannot disagree about where it comes from. Empty when no session is open
    under the key or it carries no tab, which both read as UNKNOWN occupancy and
    refuse rather than matching any occupant.
    """
    live = state.get_slot(slot_key)
    return str(getattr(live, "_tab_id", "") or "") if live is not None else ""


def manager_revision(slot: Any) -> int:
    """How many times who manages *slot* has been STATED in this process.

    Read by a caller that is about to suspend before it publishes, and given back to
    :func:`publish_manager` as ``expect_revision``. Monotone and process-local: the
    crew log is the authority across restarts, and a counter on a line would claim an
    order between two processes that never compared their reads.
    """
    return int(getattr(slot, "_manager_revision", 0) or 0)


def publish_manager(
    slot: Any,
    *,
    manager: str,
    manager_tab: str,
    unresolved: bool = False,
    inadmissible: bool = False,
    expect_revision: int | None = None,
) -> bool:
    """THE ONLY WRITER of who manages *slot*. Four attributes, one statement.

    ``_manager``, ``_manager_tab``, ``_manager_unresolved`` and
    ``_manager_grant_inadmissible`` are not four pieces of state. They are one fact --
    what this process can say about who drives this session -- held in four attributes
    because an authorization answer needs the key, the occupancy that qualifies it, and
    the two distinct ways the key may not be answerable. Anything that writes some of
    them leaves the slot describing a state that never existed, and a reader arriving
    in between gets an answer assembled from halves of two different facts.

    That is not hypothetical and it is not a matter of care at the call site. Spread
    across the resolve, the verb, the compensation and the hydrate, the four were
    writable from twenty-one places, and the SAME ordering defect was found and fixed
    at four of them in turn: flags cleared, a read awaited, the key written after --
    which publishes "resolved" with an empty key, and an empty key on a marked slot
    reads as "the creator holds it", handing a revoked giver its reach back. Each fix
    was correct and each left the other sites, because the invariant cannot be stated
    at a site. Here it can be, once: every attribute is assigned, from arguments, with
    NO suspension between them.

    So this function contains no ``await`` and must never gain one, which a contract
    test pins along with its sole ownership of the four names. A caller that needs an
    awaited value -- the occupancy read off the log, say -- awaits it BEFORE calling
    and passes the result, so the suspension happens while the slot still holds its
    previous whole statement rather than a partial new one.

    ``unresolved`` and ``inadmissible`` are the two ways the key cannot authorize, and
    both default to False because the ordinary publish is a decided, usable answer.
    They are never both true: one says the value was never read, the other that it was
    read and may not be acted on.

    EVERY CALLER HANDS IN A RECORDED VALUE. No caller may mint one here. ``manager_tab``
    is the occupancy the ``session/manager_set`` entry carries, and the only code
    entitled to read the live occupant of a key is the handoff, because that is where
    the record is MADE; everything else APPLIES a record and has to use the identity
    that record holds. The two are not interchangeable: a refusal path that stamped the
    current occupant would write a stranger's identity into the slot, a later
    compensation would copy it onto the log, and the log would then name a session
    nobody handed anything -- the mirror image of putting identity on the record in the
    first place. A path that refuses must never rewrite what it refuses.

    ATOMIC IS NOT ORDERED, which is why there is a REVISION. Writing the four together
    stops a reader seeing half of two statements; it does nothing about a statement
    that was already old when it arrived. A resolve reads the crew log off the loop and
    then publishes, and a handoff can commit inside that window: the entry lands, the
    marker lands, the new manager is published -- and then the older read publishes the
    manager it saw, which the handoff revoked. Four atomic writes of a stale fact are
    still a lost update.

    So a caller that suspends before publishing reads :func:`manager_revision` first
    and gives it back as ``expect_revision``. A publication whose revision has moved is
    DISCARDED and answers False: something else stated the fact meanwhile, and the
    stale answer has nothing to add. An authoritative writer -- the handoff publishing
    the grant it just made durable, a refusal forcing the slot unresolved -- passes
    nothing and always wins, which is what makes the discard land on the right side.
    Note the cost of having waited for the funnel: this is ONE check. Before the four
    writes became one function it was twenty-one.
    """
    if expect_revision is not None and manager_revision(slot) != expect_revision:
        return False
    slot._manager = manager
    slot._manager_tab = manager_tab
    slot._manager_unresolved = unresolved
    slot._manager_grant_inadmissible = inadmissible
    slot._manager_revision = manager_revision(slot) + 1
    return True


def manager_record_names(slot: Any, identity: str) -> bool:
    """Whether *slot*'s published manager record names the session *identity*.

    THE ONE PLACE a tab identity is compared against the manager record, and the reason
    it is a function rather than two comparisons is that this exact defect -- a key
    treated as an identity -- has come back fourteen times on this line. Two askers need
    the answer and both get it here:

    * the RECORD's own staleness check, which asks whether the session answering to the
      manager's key is still the one the record names;
    * a CALLER claiming to be the manager, which asks whether the session asking is that
      one. A queued delivery's stamped sender is that caller, re-asked at the drain.

    False when either side is empty, because an unknown identity is not a match: a
    record written without one, and an asker that cannot say which session it is, are
    both unprovable and neither may be read as "yes".
    """
    recorded = str(getattr(slot, "_manager_tab", "") or "")
    return bool(recorded and identity and recorded == identity)


def _manager_record_refusal(state: "DashboardState", slot: Any) -> "tuple[str, str] | None":
    """The refusal owed when *slot*'s published manager MAY NOT drive it, re-asked now.

    ONE place asks whether the record still names a session allowed to hold this one,
    for the same reason :func:`_manager_unread_refusal` is one place: every condition
    split out of here becomes a condition each reader has to remember, and the readers
    are added later than the condition. Both questions it asks were established when
    the record was PUBLISHED and can stop being true while it stands, so both are
    re-asked on the read rather than trusted from the resolve:

    * IDENTITY -- is the session answering to the manager's key the one the record
      names? An edge that carries only a key is ambiguous once that key's occupant is
      a different SESSION. Key re-occupancy is not the test and would be the wrong
      one: a close followed by a resume puts a new slot object under the same name,
      and refusing that would break a plain resume of the very session that was handed
      this one -- lost reach for no ambiguity at all.
    * STORE EQUALITY -- do the two slots still agree on their memory store? That is
      the positive principal test: a handed session and the session driving it run
      under one principal. It is checked when the grant is admitted and when a record
      is applied, and a store can be rebound by the manager's own routes afterwards,
      under the manager's slot lock and not this one's. Without it, a grant admitted
      while the stores agreed keeps answering after they stop agreeing, and the next
      restart is what notices -- the rebuild re-validates on exactly this comparison
      and drops the grant, so the reach is one the process holds and the log does not.

    Same session under the same store resolves and proceeds; anything else, including
    undeterminable, refuses.

    The identity is ``_tab_id`` and it needs no new storage: it is PERSISTED
    (``metadata_codec`` writes ``tab_id``, and ``_read_tab_id`` keeps the one it finds
    rather than minting), so a resume carries it across while a genuinely new occupant
    of the name mints its own. The handoff records it on its own
    ``session/manager_set`` entry, which is where the durable half lives. It is also
    the mechanism this module already uses for the same question on the send-origin
    stamp.

    Compared only when a manager is published: an empty ``_manager`` is the creator
    reading, which compares ``_created_by`` the same key-to-key way and has the same
    (pre-existing) question, and this does not reach it.

    UNDETERMINABLE refuses with the different-session case: no live slot under the key,
    a slot carrying no identity, or a record written without one. None of those is a
    statement that the occupant is the same session, and the compare cannot be made
    from a value nobody has.
    """
    manager_key = str(getattr(slot, "_manager", "") or "")
    if not manager_key:
        return None
    if not manager_record_names(slot, _tab_identity(state, manager_key)):
        return (
            "the session recorded as managing the one you named is not the session "
            "answering to that name now, so who manages it cannot be identified -- "
            "hand it over again to say who does",
            "manager_session_changed",
        )
    live = state.get_slot(manager_key)
    if str(getattr(live, "memory_store", "") or "") != str(getattr(slot, "memory_store", "") or ""):
        # The SAME comparison the admission made and the rebuild re-makes, with the
        # same code: a record whose two sides disagree on a store is a grant that
        # is not admissible now, and the way out is the same fresh handoff. Reached
        # only with the identity settled above, so this is the recorded session's own
        # binding and not some later occupant's.
        return (
            "the session managing the one you named is bound to a different memory "
            "store now, so it runs under another principal and cannot hold it -- "
            "hand it over again to say who manages it",
            "manager_grant_inadmissible",
        )
    return None


#: The one refusal code about a manager that a caller may WAIT on.
#: :func:`_arm_manager_reresolve` fires from this refusal and from no other, so this is
#: the only one that ends without anybody doing anything. Named rather than spelled at
#: each comparison because the difference decides whether a queued message is held or
#: destroyed, and a second copy of that decision is how the two got treated alike.
MANAGER_READ_PENDING = "manager_unresolved"


def _manager_unread_refusal(
    state: "DashboardState", slot: Any, *, for_write: bool = False
) -> "tuple[str, str] | None":
    """``(message, code)`` owed by a read of who manages *slot*, or ``None`` when known.

    ONE place decides "the manager is not usable", for every way it can fail to be,
    because the alternative is a condition at each reader that lists which operations
    consult the value -- and a list like that is wrong the moment a reader is added
    without it, which is how the same hole was closed four times on this branch. The
    conditions that can go stale while a record STANDS live in
    :func:`_manager_record_refusal`, which this calls, so there is one function to add
    a staleness test to and one function every reader passes through.

    EVERY BODY HERE IS GENERIC, AND THAT IS A CONTAINMENT RULE RATHER THAN A STYLE. These
    refusals are raised from the READ, which runs BEFORE the ownership fence has decided
    the caller may reach this slot at all -- that is the whole point of putting the guard
    on the read, and it is also what makes the wording a disclosure surface. A refusal
    that interpolated the target's title would answer "what is this stranger's session
    called" to any fenced caller willing to name a key, and one that named the manager's
    key would answer "who is driving it": a reporting oracle reached by being refused.
    So no body carries the target's title, the manager's key, or any slot key. The CODE
    stays specific, because the caller has to know whether to retry, and a code is a
    fixed vocabulary that says nothing about whose session it is. Same rule as the
    tracebacks this module drops on its fail-closed paths: a path that refuses must not
    become the way to read what it is protecting.

    All of them are PER SLOT. A slot the handed-over index names comes up unresolved
    and is refused here until its own log is read; one whose log could not be read
    stays refused; one whose recorded manager cannot hold it, or whose key a different
    session now holds, is refused with its own reason. An unindexed slot never reaches
    any of these: it was never handed over, so its creator holds it and that is decided
    from a cached set. An empty ``_manager`` on an INDEXED slot is not evidence of
    "unmanaged": it is the decided claim "the creator holds it", and a read that has not
    happened cannot make that claim.
    """
    if not for_write and getattr(slot, "_manager_grant_inadmissible", False):
        # Read, decided, and NOT publishable: the recorded manager is closed or rebound
        # to another principal. Its own code and its own wording, because the caller's
        # position is different -- there is nothing to retry, and the way out is a fresh
        # handoff recording who holds the session rather than a moment's wait.
        return (
            "the crew log records a session managing the one you named that cannot "
            "hold it now, and no transfer back was ever recorded -- hand it over "
            "again to say who manages it",
            "manager_grant_inadmissible",
        )
    if not for_write and (stale := _manager_record_refusal(state, slot)) is not None:
        # Readable for a WRITE, like the inadmissible grant and for the same reason: the
        # refusal tells the caller to hand the session over again, and that verb has to
        # be able to name what it is replacing. Refusing it here would make the way out
        # unreachable and the denial permanent.
        #
        # AFTER the two above, because both are sharper statements about the same
        # refusal: "the recorded manager cannot hold this session" and "the log has not
        # been read" each say more than "whoever answers to that name is not it".
        return stale
    if getattr(slot, "_manager_unresolved", False):
        # The wording tells the caller WHICH kind of wait it is in, because the two
        # have different answers. A read already in flight will land on its own; one
        # that has merely been scheduled is about to. Either way the refusal is
        # self-clearing, which is what makes "retry" a true instruction and not a
        # promise with nothing behind it.
        reading = str(getattr(slot, "key", "") or "") in _manager_reresolve_inflight
        return (
            "which session manages the one you named has not been read from the crew "
            "log yet, so this cannot be authorized against it -- "
            + ("that read is running now" if reading else "it is being read now")
            + ", retry in a moment",
            MANAGER_READ_PENDING,
        )
    return None


#: Slot keys whose re-resolution is already scheduled, so a slot refused once per
#: verb does not queue one read per refusal. Discarded when the read finishes,
#: whatever it decided -- a read that failed again leaves the slot unresolved and the
#: next refusal is free to schedule another, which is what makes a transient store
#: fault recoverable without a restart.
_manager_reresolve_inflight: set[str] = set()


def _arm_manager_reresolve(state: "DashboardState", slot: Any) -> None:
    """Schedule the read the refusal tells the caller to retry.

    Without this the refusal is a lie. ``_manager_unresolved`` is set by a read that
    did not decide, and the only callers of :func:`resolve_slot_manager` are BUILD
    paths -- the boot pass, a revive, a resume from history. None of them runs again
    for a slot already open, so a slot that went unresolved stayed that way until the
    process restarted, while every refusal said "retry in a moment". Retrying could
    never work.

    One bad unit anywhere in the store is enough to get there: the listing
    :func:`~kiro_crew.crew_log.session_tree.slot_manager` makes is store-wide, and a
    ``CrewLogError`` from it is not an ``OSError``, so it escapes the three handlers
    there and lands on the catch-all -- which marks the slot unresolved. A fault that
    clears has to be re-read, or it is permanent for the life of the gateway.

    Armed from the REFUSAL, so there is no path that denies for an unread manager
    without scheduling the read that clears it. Fire-and-forget rather than awaited,
    because the refusal is raised from :func:`require_manager_of`, which is sync and
    runs inside ``close_target``'s no-suspension window: scheduling a task is not a
    suspension, and the caller is being refused either way. Held on the state's
    background set like every other detached task here, so a shutdown can see it.
    """
    key = str(getattr(slot, "key", "") or "")
    if not key or key in _manager_reresolve_inflight:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No loop: a synchronous caller outside the gateway's runtime. The slot stays
        # unresolved and the next refusal inside the loop arms the read.
        return

    async def _reread() -> None:
        try:
            await resolve_slot_manager(state, slot)
        except Exception:
            logger.debug("manager re-resolve: %s could not be read", key, exc_info=True)
        finally:
            _manager_reresolve_inflight.discard(key)

    _manager_reresolve_inflight.add(key)
    task = loop.create_task(_reread())
    background = getattr(state, "_background_tasks", None)
    if isinstance(background, set):
        background.add(task)
        task.add_done_callback(background.discard)


def manager_read_is_pending(state: "DashboardState", slot: Any) -> bool:
    """Whether a refusal about *slot*'s manager will clear ITSELF.

    Three refusals say "the manager cannot be used", and they are one word apart in the
    snapshot (``manager_unknown``) and one exception class apart in code -- but they
    have opposite answers to "should a caller wait". A read that has not landed lands:
    the refusal arms it. A record naming a session that cannot hold the slot, and a key
    whose occupant is a different session, are SETTLED answers about a record that
    stands, and the only thing that moves either is a person handing the session over
    again -- which may never happen.

    Asked by any caller that holds data back rather than refusing an action. Waiting on
    a settled refusal is not a conservative choice, it is an unbounded one: the queue
    head that waits for it waits for the life of the process, and everything behind it
    waits too, including the prompts a person typed into the session themselves.

    ``False`` when the manager reads fine, because then there is nothing to wait for.
    """
    refusal = _manager_unread_refusal(state, slot)
    return refusal is not None and refusal[1] == MANAGER_READ_PENDING


def require_manager_of(state: "DashboardState", slot: Any, *, for_write: bool = False) -> str:
    """:func:`manager_of`, REFUSED when the value has not been read yet.

    The chokepoint every authorization read goes through, and the whole reason the
    refusal is not a condition in :func:`authorize_target` instead: a gate that asks
    "does this operation consult the manager?" is an entry predicate approximating the
    read, and it admitted a guessed value at three sites that reach the value without
    passing it -- the handoff verb's own ``already_manager`` check, its commit's
    ``previous_manager``, and :func:`broadcast_audience`. Put on the READ, the guard
    cannot be reached around: a reader added later gets it by calling this, and a
    reader that wants the raw value has to say so by calling :func:`manager_of`.

    An unfenced caller whose operation never reads the manager is never refused by
    this, because it never calls it -- which is the availability property the refusal
    had to keep without widening into verbs that do not depend on the value.

    A refusal for an UNREAD value also ARMS the read that clears it
    (:func:`_arm_manager_reresolve`), so "retry in a moment" is a true instruction.
    The inadmissible grant does not arm it, because the read already succeeded and
    would return the same answer for the life of the process. Arming there would spend
    one store listing per refusal to learn nothing.

    ``for_write`` is for the one caller that is not authorizing a delivery but WRITING
    the statement that replaces the one it reads -- the handoff verb's gate and commit,
    which need the manager they are superseding in order to record it. An inadmissible
    grant is readable for that purpose and refused for every other, which is what keeps
    the denial from being a dead end: the refusal tells the caller to hand the session
    over again, and that verb has to be able to run. An UNREAD value is still refused
    even for a write, because a statement about what is being replaced cannot be made
    from a value nobody read. Kept as a parameter here rather than a second accessor so
    the rule stays in one function.
    """
    if (refusal := _manager_unread_refusal(state, slot, for_write=for_write)) is not None:
        if refusal[1] == MANAGER_READ_PENDING:
            _arm_manager_reresolve(state, slot)
        raise SessionControlError(refusal[0], code=refusal[1], status=409)
    return manager_of(slot)


def _managed_by_other(
    state: "DashboardState", slot: Any, caller_key: str, caller_tab: str, *, for_write: bool = False
) -> bool:
    """Whether *slot* is not held by *caller_key*, by creation or by handoff.

    Raises :class:`SessionControlError` when who manages *slot* is not known, because
    it reads the value through :func:`require_manager_of`: "not held by the caller" is
    an authorization answer and there is none to give from an unread field.

    *for_write* is that read's own parameter, passed through rather than re-derived: a
    caller asking in order to REPLACE the value needs to be able to name a record that
    is readable and stale, or the refusal's way out -- hand the session over again --
    is itself refused. Passing it through is what lets the handoff's own gate ask this
    question here instead of comparing keys beside it.

    *caller_tab* is the asker's own IDENTITY and has no default, because the one
    defect this parameter exists to close is a site that forgot it. Matching the key
    alone says a session answering to that NAME holds the slot, which a REPLACEMENT
    under the same name also does -- so a key-only answer hands a replacement the
    grant its predecessor was given and then lost. Required rather than optional so a
    new asking site cannot inherit that answer by omission: the signature refuses it
    before any test does.

    Compared through :func:`manager_record_names`, the one place an identity meets the
    record, because a second comparison written at the asking site IS this defect --
    it has come back fourteen times, and every recurrence was a new copy of the
    compare rather than a new idea about it.

    A live session-control caller passes its own identity too, where the compare is
    redundant by construction: the asker is the slot asking in its own process, so the
    identity it offers is the occupant of its own key. Passed anyway because an
    exemption is a list, a list is what a later caller is added without, and a
    redundant dictionary read costs nothing against that.

    An EMPTY identity reads key-only. It is reachable from one place -- a queued
    entry's provenance written before this field existed, restored after an upgrade --
    and dropping those would delete messages their senders were authorized to send and
    still are. It grants nothing a forged provenance could not already claim by
    stamping the recorded identity outright, so the fail-closed reading buys no
    containment and costs real text.
    """
    if require_manager_of(state, slot, for_write=for_write) != caller_key:
        return True
    if not str(getattr(slot, "_manager", "") or ""):
        # The key that matched is ``_created_by``, which no record carries an identity
        # for -- the same guard :func:`_manager_record_refusal` opens with, for the
        # same reason. There is no record here, so there is nothing about one that
        # could have gone stale, and comparing anyway refuses every creator of a
        # session that was never handed over: an absent record names nobody, so it
        # names nobody who is asking.
        return False
    if not caller_tab:
        return False
    return not manager_record_names(slot, caller_tab)


class _LineRead(NamedTuple):
    """What one walk of the manager line found, and if it stopped, whether that clears.

    *found* is whether the wanted key is on the line. *unread* is whether the walk could
    not be finished, which the two callers read with opposite polarity. *waits* is
    whether the reason it could not be finished RESOLVES ITSELF -- a hop whose manager
    has been decided and not yet published -- as against a cycle, a line past the depth
    bound, or a record the bound evicted, none of which a retry would answer differently.

    Three fields rather than a bool plus a second traversal, because the thing that
    went wrong four times in this feature is a check that is correct where it runs and
    is not on the path that reaches the decision. One walk cannot disagree with itself.
    """

    found: bool
    unread: bool
    #: WHY the walk could not be finished, for the refusal to name. ``""`` when it
    #: finished. ``"pending"`` a hop whose manager is decided and not yet published;
    #: ``"reused"`` a hop whose key is held by a session its edges are not about;
    #: ``"inadmissible"`` a hop whose recorded manager may not be published at all;
    #: ``"forgotten"`` a hop whose record the cap dropped; ``"cycle"`` and ``"depth"``
    #: the two shapes that were always unread. Separate from :attr:`waits` because they
    #: answer different questions: whether to retry, and what to tell someone reading
    #: the refusal.
    cause: str = ""

    @property
    def waits(self) -> bool:
        """Whether the reason this line could not be read RESOLVES ITSELF.

        Only a hop mid-publication does. A key held by another session, a grant that is
        not admissible, a dropped record, a cycle and a line past the depth bound all
        answer the same way however long anyone waits.
        """
        return self.cause == "pending"


#: How a refusal from :func:`require_manager_of` reads as a walk cause. Only the two
#: self-clearing refusals wait; the rest are decided answers about this hop's manager.
#: An unmapped code is treated as decided, because a cause that promises a retry is
#: worse than one that does not when nobody knows whether waiting helps.
_WALK_CAUSE = {
    "managers_not_restored": "pending",
    "tree_not_ready": "pending",
    "manager_unresolved": "pending",
    "manager_session_changed": "reused",
    "manager_grant_inadmissible": "inadmissible",
}


def _walk_manager_line(
    state: "DashboardState",
    start_key: str,
    wanted_key: str,
    *,
    seed: Sequence[str] = (),
    expected: "str | None" = None,
    claimed: str = "",
) -> _LineRead:
    """Walk the manager line above *start_key* once, judging every hop in one place.

    The ONE traversal. The fence reads ``found`` from it and the loop guard reads
    ``found`` and ``unread`` with the opposite polarity, and both read ``cause`` from
    the same result -- so every check below happens exactly once, on the only path
    either of them can reach a decision through.

    Dict and attribute reads only, with no I/O and no suspension, because the fence that
    calls it runs inside ``close_target``'s no-suspension window.

    ORDERING INVARIANT, and it is the whole lesson of this function's history. Ten
    findings on this walk had one shape: a guard that was correct where it ran, on a
    path that reached a decision without running it. Each time the guard was added to
    one more branch; each time a branch nobody had enumerated got there first. So the
    rule is positional, and it is stated here because the only way to hold it is to
    enumerate the branches rather than wait for the eleventh finding:

        A check about a hop's IDENTITY runs before every branch that USES that identity.
        A check about a hop's MANAGER runs before every branch that READS that manager.

    Both halves matter and they are not the same rule. ``_key_is_ambiguous`` asks whether
    the session holding this key is the one the edge that NAMED it was written for -- an
    identity comparison against what that edge recorded, so a session that closed and
    reopened its own tab passes it -- so it must precede the identity match
    and the visited test -- the match IS an identity claim, which is how it came to be
    the tenth finding. ``_hop_handoff_unsettled`` and the ``_manager_unresolved`` read ask
    whether the hop's manager is published, so they must precede every read of that
    manager, and they may sit AFTER the identity match: matching a key does not read its
    manager, so refusing there would deny a caller whose own identity was never in doubt.
    That asymmetry is deliberate and ``test_a_hop_mid_handoff_is_still_matched_on`` pins
    it.

    *claimed* is the IDENTITY THE ASKER OFFERS for *wanted_key*, and it is read at the
    match and nowhere else. ``key == wanted_key`` is a comparison of NAMES, and a name
    outlives the session that answered to it: a lead's key, retaken and then handed its
    old descendants again by its new occupant, is a key whose edges all agree with each
    other and name a different session from the one a waiting delivery was stamped by.
    Every hop ABOVE the asker is judged against the edge that named it, so this is the
    one hop that was being taken on a string. It reads key-only when empty, exactly as
    the one-level :func:`_managed_by_other` reads an empty identity, because this walk
    re-asserts that predicate and may never refuse what it allowed.

    The branches, in order, so the next change has a list to check itself against: the
    empty hop; the identity check; the identity match together with the asker's own
    claim; the visited test; the manager check; the live-versus-departed split; and
    within the departed side the occupancy lookup, the eviction test, the tombstone, the
    pending entry and the entry naming nobody. The empty hop is the one early return
    ahead of the identity check, and it is ahead of it because there is no hop to ask
    about -- both checks take a key.
    """
    if not wanted_key:
        # Not a key any session can hold, so it is on no line at all, read to the end.
        return _LineRead(False, False)
    seen = {key for key in seed if key}
    key = start_key
    for _ in range(LINEAGE_WALK_MAX_DEPTH):
        if not key:
            # No hop at all, so there is nothing for the per-hop checks to be about:
            # both of them take a key, and the identity check answers False for "".
            return _LineRead(False, False)
        if _key_is_ambiguous(state, key):
            # KEY-RELATIVE, and BEFORE the match below and anything that branches on
            # whether this hop's slot is live, because a key held by a session its own
            # record cannot identify is ambiguous in every one of those positions. The
            # match is the position that matters most: the key the caller is being
            # recognised BY is itself a reusable name, so a new occupant of a deleted
            # lead's key would otherwise be accepted as that lead and inherit every
            # session below it -- the walk would hand over the whole line on the
            # strength of a string.
            #
            # Only a key whose occupant is NOT the session its own record names is
            # refused, so this costs nothing to a key that never changed hands and
            # nothing to a session that closed and resumed under its own id.
            # Deliberately narrower than moving the whole hop-state block above the
            # match: a hop mid-handoff is still matched on, because its identity is not
            # in doubt -- only its manager is, and that question belongs below.
            return _LineRead(False, True, "reused")
        if _edge_is_ambiguous(state, key, expected):
            # EDGE-RELATIVE, a SEPARATE step and not weakened: the occupant answering
            # this key now is not the session the edge that NAMED it was published to.
            # The first hop is seeded with the target's own edge identity, which only
            # this comparison reads, so it has to stay distinct from the key-relative
            # check above -- that one asks about the key's own record, this one about the
            # edge pointing AT the key.
            return _LineRead(False, True, "reused")
        if key == wanted_key:
            if _claim_is_ambiguous(state, key, expected, claimed):
                # THE ASKER'S OWN CLAIM, asked INSIDE the branch that matched the name
                # so the comparison of keys does not stand on its own. The hops above
                # are each judged against the edge that named them; without this the
                # last hop -- the asker itself -- is the one place the line is read off
                # a string, and a new occupant of a deleted lead's key that re-publishes
                # that lead's old edges presents a chain every other check agrees with.
                return _LineRead(False, True, "reused")
            # The MATCH comes before the pending check: a hop that IS the wanted key is
            # recognised whatever is in flight ABOVE it. An unsettled handoff on this
            # key is not an identity problem -- its identity is not in doubt, only its
            # own manager is, and that question belongs to the hop ABOVE it, not here.
            return _LineRead(True, False)
        if _hop_handoff_unsettled(key):
            # Asked of every hop the walk CLIMBS THROUGH -- after the match, before the
            # cycle test and before anything branches on whether the slot is still in
            # the table. This hop's own next manager is already claimed and not yet
            # published: the queued entry names the manager it is moving to, and what
            # the walk can read -- the live ``_manager``, or a committed departure
            # record -- names the one it is moving from. A departed hop is if anything
            # the worse of the two, since its record looks settled and is read with no
            # sign that a different manager is on its way.
            #
            # Placed ahead of ``key in seen`` on purpose: the loop a handoff would close
            # runs THROUGH this hop, so the walk reaches the visited test on the
            # strength of a manager edge that is in the act of moving. Deciding "cycle"
            # there reads the move as settled; what is true is that the line cannot be
            # read yet. So the hop counts as unread, both polarities refuse, and it
            # CLEARS because that append settles either way -- which is what ``waits``
            # carries to the caller as ``manager_line_pending`` rather than the
            # permanent ``would_cycle``.
            #
            # Reading either is how a cycle gets written. The endpoint fences refuse a
            # move whose OWN target or recipient has an unsettled append, but they say
            # nothing about the hops BETWEEN them, and the loop a handoff closes can run
            # entirely through those: with L managing A and D, A managing C and D
            # managing B, a timed-out A -> B leaves A reading as L's whether A is live or
            # closed, so D -> C passes every endpoint check and the two deferred commits
            # land A -> B -> D -> C -> A, cutting L off with no recovery path.
            return _LineRead(False, True, "pending")
        if key in seen:
            # A cycle in a record that should not have one. Unread, and waiting does not
            # make it readable.
            return _LineRead(False, True, "cycle")
        seen.add(key)
        above = state.get_slot(key)
        if above is None:
            # A DEPARTED hop, resolved here rather than by anybody having rewritten a
            # field when it left: the departure recorded its own manager, and the walk
            # reads through it. A hop it cannot resolve STOPS the walk -- it is not
            # skipped, and nothing above it is admitted -- so an unresolvable line
            # denies reach, which is what the gateway answers without this feature.
            #
            # WHICH occupant's record is read is decided by the key's occupancy and by
            # nothing else, and the match on it is EXACT. A slot key is reusable, so
            # "what did this key report to" is a question the key cannot answer: it has
            # had several occupants. Resolving the occupancy first and then requiring
            # the pair to match means an entry filed under any other generation is not
            # something the walk can reach for -- not as a fallback, not when the exact
            # one is missing, not ever. That is the property, and it does not depend on
            # every writer having been fenced correctly.
            held = _key_occupancy.get(key)
            if held is None:
                # Nothing on record about this key at all -- a key from before a
                # restart, or one the bound has evicted. Read to the END, which is what
                # a gap meant before any record existed: the line is ALREADY unreadable
                # to every walk, a reader is denied reach by it, and a person's own
                # closed tab is exactly this shape, so counting it as unread would
                # refuse every handoff below it. No entry is reachable here to leak.
                #
                # One of those two causes is worth saying out loud. A key the CAP
                # dropped is a line that worked a moment ago and stopped working for a
                # reason that has nothing to do with the caller, and the refusal it
                # produces is the same one an ordinary ownership failure produces. So
                # the eviction is named here, once per walk that meets one.
                if key in _recently_evicted_keys:
                    logger.info(
                        "manager line: %s is on a line through %s, whose departure "
                        "record the %d-key cap dropped, so reach across it is refused "
                        "-- this is a retention limit and not an ownership decision",
                        wanted_key,
                        key,
                        _KEY_OCCUPANCY_MAX,
                    )
                if key in _recently_evicted_keys or _key_occupancy_evictions[0]:
                    # UNREAD, not read-to-the-end, and the two polarities part here.
                    # This key HAD a line a moment ago; the cap dropped it. A reader is
                    # denied either way, but a WRITER must not conclude "nothing sits
                    # above this hop" from a record that was discarded -- that is how
                    # the loop guard writes an edge which becomes a cycle when the
                    # evicted session reopens under its old creator.
                    #
                    # The CONDITION is the counter, not the shadow table. That table is
                    # itself evictable, so a key can age out of it too -- and if this
                    # answer depended on finding the key there, the boundary would move
                    # when the shadow forgot, which is a boundary resting on an
                    # evictable cache. The counter is never evicted: once this process
                    # has dropped anything, it cannot prove THIS key is not one of the
                    # ones it dropped, so it refuses. The shadow only decides whether
                    # the log line above can name the key.
                    #
                    # Before the first eviction the counter is zero and a gap keeps the
                    # answer below -- which is the case that must not refuse: a key
                    # never seen in this process, such as a tab closed before a
                    # restart, where refusing would freeze every handoff under it.
                    return _LineRead(False, True, "forgotten")
                if not _departures_rebuilt[0]:
                    # THE BOOT WINDOW. A gap means "no record of a line", and that reads
                    # as a line that ENDED -- right once the pass has run, wrong while it
                    # is still running: the record is not absent, it is not read yet. A
                    # reader loses nothing either way, but the loop guard concludes
                    # "nothing sits above this hop" and writes an edge that becomes a
                    # PERSISTED cycle the moment the pass restores that hop's real
                    # manager, and one cycle takes the line away from every session in it.
                    #
                    # It CLEARS, so the refusal says retry: the pass settles either way.
                    return _LineRead(False, True, "pending")
                return _LineRead(False, False)
            departure = _departed_managers.get((key, held))
            # NO second identity test here, and that absence is deliberate. This branch
            # used to re-ask whether the key had changed hands, reading the same entry
            # the per-hop check at the top of the loop reads -- two copies of one
            # question, which is the shape every finding on this walk has had. The check
            # above answers it for a departed hop and a live one alike, so a hop that
            # reaches this line has already been judged on identity and the only
            # questions left are about the line itself.
            if departure is None or departure.pending:
                # A hop whose line is NOT YET KNOWN, which is a different thing from a
                # hop with no line. The occupant holding this key is inside its own
                # close -- popped, not yet archived -- so what it reports to has not
                # been published; or its entry was evicted, which leaves the same
                # ignorance. Either way the walk could not read this line, so it
                # answers ``unread_is_present`` and BOTH polarities refuse: a reader is
                # denied reach, and a writer refuses to add an edge below a hop whose
                # own manager is about to arrive.
                #
                # That second half is the whole point of separating this from the case
                # below. Read as a line that ENDED, a pending vacancy tells the handoff
                # loop guard that nothing sits above this hop, so the guard permits an
                # edge that closes a cycle once the close commits and the real manager
                # lands -- T -> R -> M -> T, with every manager above M cut off and the
                # cycle persisted to the manager log. The vacancy is not an absence of
                # a manager; it is a manager not yet said out loud.
                #
                # ``waits`` is True for a PENDING entry, whose close publishes or rolls
                # back, and False for a MISSING one, which the bound dropped and no
                # amount of waiting brings back.
                return _LineRead(False, True, "pending" if departure is not None else "forgotten")
            if not departure.manager:
                # Departed, COMMITTED, and reporting to nobody. This one really is a
                # line read to its end: there is no hop above it, now or later, so
                # nothing above is admitted and a handoff below it is safe to write.
                return _LineRead(False, False)
            # The next hop's expected identity comes from THIS departure's own edge,
            # exactly as a live hop's comes from its slot. A closed hop does not stop
            # the line being judged on identity; it only moves where the answer is read.
            if departure.manager_tab is not None:
                # A DEPARTED HANDED-OVER EDGE, and the principal check belongs here for
                # the same reason the identity one does: this edge grants exactly the
                # reach the live edge grants, and the live edge is refused once the two
                # sides stop agreeing on a store. The hop is closed, so its own store
                # comes from the record; the manager's comes from its live slot, or
                # from its own record when it is closed too.
                above_store = _principal_of(state, departure.manager)
                if above_store is None or above_store != departure.store:
                    return _LineRead(False, True, "inadmissible")
            expected = departure.manager_tab
            key = departure.manager
            continue
        expected = _edge_identity(above)
        try:
            # THE ONE CHOKEPOINT, for a hop exactly as for a target. Everything this
            # walk used to ask itself about a live hop -- was the manager ever read, is
            # the recorded grant admissible, is the session answering to that key the
            # one the record names, do the two still agree on a memory store -- is what
            # this function asks, and it is the function every verb already asks. Four
            # checks here were four copies to keep in step with it, and a bare edge read
            # RAW beside them was a path that authorized off a value nobody checked.
            #
            # The verbs ask it about the slot they were called on. Nothing asked it about
            # a slot that is merely a STEP in somebody else's line, and that gap is the
            # whole of what this call closes.
            key = require_manager_of(state, above)
        except SessionControlError as refused:
            # Its refusal, carried out as this walk's own cause so the message a person
            # reads names what actually happened. Only the two self-clearing ones WAIT:
            # a manager not yet read from the log, and a tree mid-publication. Every
            # other refusal is a decided statement that this hop's manager may not drive
            # it, which is what ``inadmissible`` says -- so an unmapped code lands there
            # rather than promising a retry that cannot help.
            return _LineRead(False, True, _WALK_CAUSE.get(refused.code, "inadmissible"))
    # Out of depth, with every hop so far live and readable. Unread, and an ancestor
    # above the bound is ordinary rather than something a retry resolves.
    return _LineRead(False, True, "depth")


def _edge_identity(slot: Any) -> "str | None":
    """WHICH occupant of its manager's key *slot*'s edge was written for, or ``None``.

    The identity an edge carries, read off the slot the edge belongs to. The base PR
    records it when a handoff publishes: the ``_tab_id`` of the session that was handed
    this one, which is persisted and so survives a restart.

    ``None`` is a CREATION edge -- no handoff was published, so ``manager_of`` answers
    ``_created_by`` and there is no recorded occupant to compare. That case is left
    exactly as the gateway answers it without this feature: the base PR states it is the
    same key-to-key question ``_created_by`` has always had, and widening it here would
    refuse the ordinary creation chain. ``""`` is different and is NOT the same as
    ``None``: the edge IS a handoff and names no occupant, which cannot be matched by
    anybody and refuses.
    """
    handed = str(getattr(slot, "_manager", "") or "")
    if not handed:
        return None
    # ``""`` and NOT ``None`` when the edge is a handoff whose occupancy is missing --
    # a restored slot whose tab read returned empty or raised. ``None`` means "creation
    # edge, no occupant to compare", and collapsing the two made that hop skip the
    # identity check AND the store check the moment the slot closed: refused while it
    # was live, resolvable once it was gone, so the close was what opened it. ``""``
    # matches nobody, which is what an unverifiable occupancy has to answer.
    return str(getattr(slot, "_manager_tab", "") or "")


def _principal_of(state: "DashboardState", hop_key: str) -> "str | None":
    """The memory store *hop_key*'s occupant runs under, or ``None`` when unknown.

    Live slot first, then that occupant's own departure record, because a line may pass
    through a hop whose session has closed -- which is the case this whole feature is
    for. ``None`` is UNKNOWN and refuses: a principal nobody can state is not a
    principal that agrees.
    """
    live = state.get_slot(hop_key)
    if live is not None:
        return str(getattr(live, "memory_store", "") or "")
    held = _key_occupancy.get(hop_key)
    if held is None:
        return None
    entry = _departed_managers.get((hop_key, held))
    return entry.store if entry is not None else None


def _occupant_identity(state: "DashboardState", hop_key: str) -> str:
    """WHO holds *hop_key* now, as a ``_tab_id``, or ``""`` when that cannot be said.

    Live occupant first, through the base PR's own reader so the publish and every check
    agree about where the value comes from. For a DEPARTED occupant the departure record
    carries the tab it had while the slot was still in hand -- the walk's whole purpose
    is to read through a hop whose session is gone, and a closed hop has no live slot to
    read an identity off.

    ``""`` for a key with no live occupant and no record, for an occupant that carried no
    tab, and for a record the cap dropped. None of those says the occupant is the session
    an edge meant, so none of them matches.
    """
    live = state.get_slot(hop_key)
    if live is not None:
        return _tab_identity(state, hop_key)
    return _departed_identity(hop_key)


def _took_the_key_from_somebody(state: "DashboardState", hop_key: str) -> bool:
    """Whether the session holding *hop_key* took that name from a DIFFERENT one.

    The question an edge naming only a key cannot answer for itself. Read off the live
    occupant, which is where the mark lives: a capped table cannot hold it, because a
    mark the cap drops reads as "this key is pristine" and hands a deleted session's
    descendants to whoever opened a tab under its name.

    A session that took the key from NOBODY answers False, which is the ordinary case
    and costs nothing. One that took it from an occupant it can be shown to BE -- a
    resume of its own tab -- answers False too. Anything else, including a take from an
    occupant whose identity was never recorded, answers True.
    """
    live = state.get_slot(hop_key)
    if live is None:
        return False
    displaced = getattr(live, "_displaced_tab", None)
    if not isinstance(displaced, str):
        return False
    return not (displaced and displaced == _tab_identity(state, hop_key))


def _departed_identity(hop_key: str) -> str:
    """The tab a DEPARTED occupant of *hop_key* left on record, or ``""``."""
    held = _key_occupancy.get(hop_key)
    if held is None:
        return ""
    entry = _departed_managers.get((hop_key, held))
    return entry.tab if entry is not None else ""


_UNSET: Any = object()


def _key_is_ambiguous(
    state: "DashboardState", hop_key: str, expected: "str | None | Any" = _UNSET
) -> bool:
    """Whether the session holding *hop_key* cannot be shown to be the one meant.

    Two readings live here, chosen by whether *expected* is given, because the walk
    needs both and the resume tests call each directly.

    KEY-RELATIVE (two arguments). "Is whoever holds this key now the session its OWN
    departure record names?" True when the key has been retaken by a session that cannot
    be identified as that one -- an impostor on a departed key, or a reuse tombstone that
    names nobody -- and False for a key nobody retook and for a session resuming its own
    key under the same id. This is the check the walk runs at the top of every hop,
    ahead of the identity match, because the key a caller is recognised BY is itself a
    reusable name. A LIVE occupant is compared on its ``_tab_id`` against the id the
    key's record carries (:attr:`_Departure.occupant`); a DEPARTED occupant is ambiguous
    exactly when its record was tombstoned (:attr:`_Departure.tab` cleared).

    EDGE-RELATIVE (three arguments). *expected* is what a specific edge recorded about
    the occupant it was published to, from :func:`_edge_identity`: ``None`` for a
    creation edge, which carries no occupant and is not judged, and otherwise the
    ``_tab_id`` of the session the handoff named. True when the occupant answering the
    key now is not that session. Kept as a SEPARATE walk step from the key-relative one
    and not folded into it: the first hop is seeded with the target's own edge identity,
    which only this comparison reads.

    A slot key is a NAME, and a name can change hands while an edge keeps naming it, so
    both readings compare on IDENTITY and not on how often the name was taken: a session
    that closes and reopens its own tab keeps its ``_tab_id`` and still resolves. The
    identity is the base PR's PERSISTED one, so the answer outlives the process rather
    than resolving every edge confidently again after the next restart.

    Dict and attribute reads, with no I/O and no suspension, for the reason
    :func:`manager_of` has none.
    """
    if expected is not _UNSET:
        return _edge_is_ambiguous(state, hop_key, expected)
    if not hop_key:
        return False
    entry = _departed_managers.get((hop_key, _key_occupancy.get(hop_key)))
    if entry is None:
        # No record under the current occupancy: a key nobody retook. Nothing to be
        # ambiguous about, and the ordinary live hop costs nothing here.
        return False
    if state.get_slot(hop_key) is not None:
        # A LIVE occupant with a record means the key was retaken while a mark stood --
        # a resume or an impostor. Resolve it by identity: the record carries the id the
        # key was known by, and the occupant answering now either is it or is not.
        occupant_now = _tab_identity(state, hop_key)
        return not (entry.occupant and occupant_now and occupant_now == entry.occupant)
    # A DEPARTED occupant. Ambiguous exactly when the record was tombstoned -- the reuse
    # mark clears :attr:`tab`, so no edge above can be shown to have meant this one.
    return not entry.tab


def _edge_is_ambiguous(state: "DashboardState", hop_key: str, expected: "str | None") -> bool:
    """EDGE-RELATIVE: is the occupant answering *hop_key* not the one *expected* names?

    *expected* is what the edge pointing AT this key recorded about its occupant, from
    :func:`_edge_identity`: ``None`` when the edge named no occupant -- a creation edge,
    or a bare manager edge no published handoff ever wrote a tab for -- in which case it
    is NOT judged on identity and this returns ``False``. Otherwise it is the ``_tab_id``
    of the session the handoff was published to, and this is ``True`` when the key's
    occupant now cannot be shown to be that session.

    A SEPARATE function from :func:`_key_is_ambiguous` so the walk can hold both checks
    without the two colliding when a test patches the key-relative one, and so the
    resume tests can exercise each on its own.
    """
    if not hop_key:
        return False
    if expected is None:
        # A CREATION edge: it names no occupant, so there is nothing to compare it
        # against -- and the base PR states that is the same key-to-key question
        # ``_created_by`` has always had, which this feature does not widen. What it
        # DOES answer is the one thing the chain adds: this walk follows such an edge
        # upward, so a key whose name has changed hands since the edge was written must
        # not be resolved through whoever holds it now.
        return _took_the_key_from_somebody(state, hop_key)
    occupant = _occupant_identity(state, hop_key)
    return not (expected and occupant and occupant == expected)


def _claim_is_ambiguous(
    state: "DashboardState", hop_key: str, expected: "str | None", claimed: str
) -> bool:
    """ASKER-RELATIVE: is the session asking not the one that holds *hop_key*?

    The third of the three identity readings, and the only one about the CALLER rather
    than about a hop the walk climbs through. *claimed* is what the asker says it is --
    for a queued delivery, the ``tab`` its admission stamped, which is the identity of
    the session that was granted the reach and not of whoever answers to its name now.

    ``False`` for an EMPTY claim, which is key-only and deliberate: a stamp written
    before that field existed carries none, and the one-level :func:`_managed_by_other`
    reads such a claim key-only for the same reason -- refusing here would delete
    messages whose senders were authorized to send them and still are, for a field the
    sender never had the chance to write.

    ``False`` for a CREATION edge (*expected* is ``None``). No record carries an identity
    for one, the one-level read does not compare one either, and this walk re-asserts
    that predicate: a re-check may refuse what the admission refused and may never
    refuse what the admission allowed.

    Otherwise the asker is compared against the occupant of the key it matched, live or
    departed, through :func:`_occupant_identity`. An occupant nobody can name does not
    match, so an unidentifiable hop refuses an asker that claims to be it.
    """
    if not claimed:
        return False
    if expected is None:
        return False
    occupant = _occupant_identity(state, hop_key)
    return not (occupant and occupant == claimed)


def _hop_handoff_unsettled(hop_key: str) -> bool:
    """Whether *hop_key* has a reparent append still in flight, so its manager is moving.

    A handoff claims its edge before it publishes it: the entry is queued, the commit is
    deferred until the writer's verdict, and ``_manager`` keeps naming the previous
    manager across that whole window. So a walk that reads only the committed field
    reads this hop as belonging to a manager it is in the act of leaving.

    Read from :data:`_tree_pending_slots`, which is where both reparenting verbs
    register the append and which clears itself when the append settles. A dict lookup
    and a ``done()`` call, with no I/O and no suspension, for the reason
    :func:`manager_of` has none: the fence that reaches this runs inside
    ``close_target``'s no-suspension window.

    Not the same question as the ENDPOINT fence :func:`_refuse_if_append_pending` asks.
    That one refuses a move whose own target or recipient is mid-write, which is the
    cheap early refusal and stays. This one is asked of every hop the line passes
    THROUGH, because a loop can close entirely through hops that are neither endpoint.
    """
    if not hop_key:
        return False
    pending = _tree_pending_slots.get(hop_key)
    return pending is not None and not pending.done()


def _outside_manager_line(
    state: "DashboardState", slot: Any, caller_key: str, caller_tab: str
) -> "tuple[bool, str]":
    """Whether *caller_key* is neither *slot*'s manager nor anyone above that manager.

    The skip-level reading of the ownership fence, and a widening of nothing: a caller
    on the line above a session's manager already controls that manager, so it can
    already send it the words that drive the session. What the walk removes is the
    detour -- a lead reaches the worker its conductor runs, instead of reaching the
    worker only by telling the conductor to.

    The DIRECT manager is answered first, by :func:`_managed_by_other`, which is an
    attribute read and nothing else. So the one-level answer costs exactly what it cost
    before the line existed, and the walk is reached only when that answer refuses.

    *caller_tab* is the asker's own IDENTITY, required for the same reason
    :func:`_managed_by_other` requires it: a key is a name, and the session answering to
    that name now may not be the one the grant was given to. It has no default here
    either, because a walk that judges identity at every hop above the caller and takes
    the caller itself on a key alone is the key-only answer with more steps.

    Reached ONLY when ``agent.session_manager_chain`` is on. With the line off the fence
    calls :func:`_managed_by_other` instead and never enters here, so there is no
    "walk disabled" path through this function to get wrong and no second spelling of
    the one-level answer. The switch is read by the fence, not here, because the fence
    runs inside ``close_target``'s no-suspension window where a config load on the loop
    is the one thing it must not do.

    An unreadable line is left to deny reach, which is the fail-closed direction for a
    READER: no ancestor is proven, so the caller is outside and the verb is refused. The
    target's own key is seeded as visited, so a line that comes back to the target is
    read as the cycle it is rather than walked a second time.

    Answers ``(outside, cause)``. The cause is the walk's own reason for not finishing,
    carried out so the REFUSAL can name it: a caller that was authorized a moment ago
    and is now refused because a cap dropped a record has hit a retention limit, not an
    ownership decision, and those must not read alike to whoever is debugging it. Empty
    when the caller is simply not on the line, which is an ownership answer.
    """
    # NO exception for a bare ``_manager`` edge. One stood here, reading such an edge
    # raw on the grounds that it names no published occupant for the chokepoint to
    # validate -- and it meant a slot whose manager key was restored while its occupancy
    # read failed was authorized without the identity or admissibility check every other
    # target gets. A missing occupancy is exactly when those checks matter, so the
    # answer is the refusal, not the raw read.
    if not _managed_by_other(state, slot, caller_key, caller_tab):
        return False, ""
    manager_key = require_manager_of(state, slot)
    line = _walk_manager_line(
        state,
        manager_key,
        caller_key,
        seed=(str(getattr(slot, "key", "") or ""),),
        # The FIRST hop is named by the target's own edge, so the identity that edge
        # recorded has to come in with it. Without this the walk judges every hop above
        # the first and takes the first on the strength of a key -- which is the one hop
        # a handoff just moved, and the one most likely to name a session that has since
        # been replaced.
        expected=_edge_identity(slot),
        # The ASKER'S own identity, forwarded rather than compared here: a comparison
        # written beside the walk is a second answer the walk can disagree with, and
        # the walk is the only path that reaches this decision.
        claimed=caller_tab,
    )
    return not line.found, line.cause


class _Departure(NamedTuple):
    """One occupant's departure: what it reported to, whether that is SETTLED, and
    whether the key it is filed under has been REUSED.

    *manager* is what the walk reads through -- the departed session's own manager, or
    ``""`` for one that reported to nobody.

    *pending* is the distinction the walk cannot do without. A close takes the slot out
    of the live table and then keeps awaiting, so between those points the key has an
    occupant whose manager is not published yet. That is not the same fact as "this
    session reported to nobody", even though both leave no manager to read: one is a
    line not yet knowable, the other is a line that has ended. Collapsing them lets the
    handoff loop guard read a closing manager as the top of a line and write an edge
    under it that becomes a CYCLE when the close commits.

    *tab* and *manager_tab* are the TOMBSTONE, and they are the base PR's PERSISTED tab
    identities rather than a flag because the question is about IDENTITY. A manager edge
    names a slot KEY, and a key can be held by one session after another, so an edge
    naming a key is ambiguous when the session holding that key now is a DIFFERENT
    session from the one the edge was written for. Re-occupancy of the key is not that
    fact -- it is a proxy for it, and the proxy is wrong in the direction a person meets
    most often: a session that closes and reopens its own tab keeps its ``_tab_id``, and
    refusing that refuses every handoff to a reopened tab.

    So *tab* is this occupant's own identity, captured from the slot at its departure
    while the slot is still in hand, and *manager_tab* is the identity ITS edge named,
    so the hop above a closed hop is judged by the same rule a live edge is judged by.

    The identity is ``_tab_id`` and not this module's own invention, for a reason beyond
    not having two mechanisms for one fact: ``_tab_id`` is persisted, so the answer
    outlives the process. A process-local identity would resolve every edge confidently
    again after the next restart, which is the same ambiguity arriving one restart later.
    which costs reach across that hop and grants nothing.
    """

    manager: str
    pending: bool
    #: WHICH occupant of :attr:`manager`'s key this departure's own edge named, from
    #: :func:`_edge_identity`: ``None`` for a creation edge, which carries no occupant,
    #: and otherwise that session's ``_tab_id``. Carried so the walk can judge the hop
    #: ABOVE a closed one by the same rule it judges a live edge by.
    manager_tab: "str | None" = None
    #: This occupant's OWN ``_tab_id``, captured while the slot was still in hand, so an
    #: edge naming this key can be told whether it meant this occupant. ``""`` when the
    #: slot carried none, which matches nobody. This is what :func:`_unidentifiable`
    #: and the edge-relative hop check read, and a reuse TOMBSTONE sets it to ``""``.
    tab: str = ""
    #: This occupant's own SESSION ID, which is what the KEY-RELATIVE check
    #: (:func:`_key_is_ambiguous` called with two arguments) compares a LIVE occupant
    #: against. Captured from the departing slot's ``_tab_id`` on the live paths, and
    #: derived from the key's single ``session_units_for_slot`` entry by the boot
    #: rebuild. A reuse tombstone carries the PRIOR occupant's id here -- so a session
    #: resuming its OWN key is recognised as itself while an impostor is refused -- even
    #: though :attr:`tab` is cleared so no edge above can resolve through the key.
    occupant: str = ""
    #: The PRINCIPAL this occupant ran under, as its memory store. Carried because a
    #: live edge is refused once the two sides stop agreeing on a store, and a departed
    #: edge grants the same reach: without it a handed-over hop that closes lets a
    #: manager since rebound to another principal keep driving everything below it,
    #: which the equivalent live edge refuses. ``""`` is the default store and agrees
    #: with another ``""`` exactly as the live comparison does.
    store: str = ""


_key_occupancy: "OrderedDict[str, int]" = OrderedDict()
"""Which occupant holds each slot key NOW: slot key -> occupancy generation.

    What makes a REUSABLE key safe to record departures against. A slot key can be held
    by one session after another, so "what did this key report to" has as many answers
    as the key has had occupants, and a lookup taking the key ALONE can hand back a
    previous occupant's answer -- to a walk that would then give that occupant's manager
    reach over the sessions the current one opened. So the occupancy is resolved from
    here first, and the record is read under that occupant and no other.

    Written where a key is TAKEN, by ``_ChatSlot.__init__`` through
    :func:`note_key_taken`: that is the moment the answer changes, and it is the one
    place every path that puts a session at a key already goes through -- a tab open, a
    resume, a transfer import, a restore at startup. Deliberately NOT written by the
    departure paths, which are several and each of which would have to remember.

    This table carries the one size bound, and :data:`_departed_managers` lives inside
    it: at most one generation per key is resolvable, so a key that is retaken or
    evicted takes its entry with it. Process-local and never persisted, for the reason
    the generation itself is not.
    """


#: The manager each DEPARTED session had, so the walk can resolve a hop whose slot is
#: gone instead of a departure having to rewrite anybody's field.
#:
#: This is what replaces a repair writer. Writing the new manager onto each report at
#: departure meant a loop over live slots, a staged field, a save per report and a
#: publish -- and every one of those was a place for the departure and a concurrent
#: handoff to disagree about one slot, or for a promoted manager to stop being valid
#: between the stage and the publish. None of that exists here: a departure records ONE
#: value about ITSELF, no report is touched, nothing is saved, and the resolution happens
#: when the line is read.
#:
#: Held in memory rather than read back from the departed session's persisted metadata,
#: which is the other way to resolve the hop. That read is disk I/O, and the walk is
#: called from the ownership fence, which runs inside ``close_target``'s no-suspension
#: window -- the window ``skip_enabled_check`` exists to keep one config read out of.
#: A metadata read there would be a strictly worse instance of the blocking-call-on-the-
#: loop class, so the value is captured while the slot is still in hand.
#:
#: KEYED ON THE PAIR (slot key, occupancy generation), never on the slot key alone, and
#: read by EXACT match on that pair with no fallback of any kind. A record stored under
#: a different generation for the same key can therefore never answer for the occupant
#: holding it, which is the property a key-level lookup cannot have however the writers
#: are fenced: a key-level entry that survives its occupant is an answer waiting to be
#: given to the wrong question.
#:
#: An entry is written, naming nobody, for a departure that reported to nobody, and the
#: walk reads that as the end of a line. An entry written at the POP is marked PENDING
#: instead, and the walk reads that as a line it could not finish -- see
#: :class:`_Departure`, where the difference between the two is the thing that keeps the
#: handoff loop guard from writing a cycle under a closing manager.
#:
#: BOUNDED, and an evicted entry is not a silent skip: a hop with no exact record stops
#: the walk and the fence refuses, which is the reach a gateway has without this feature
#: at all -- so eviction costs reach and never a boundary. The bound is
#: :data:`_key_occupancy`'s, because at most one generation per key is resolvable:
#: retaking or evicting a key drops its entry with it, and there is no second eviction
#: policy to disagree with the first.
_departed_managers: "OrderedDict[tuple[str, int], _Departure]" = OrderedDict()

#: How many KEYS are remembered, and so how many departures can be. A SIZE bound,
#: deliberately its own number rather than :data:`LINEAGE_WALK_MAX_DEPTH`: that one
#: limits how far ONE line is followed, while this limits how many independent lines can
#: have a departed hop at once, and collapsing two different bounds onto one number is
#: how a later change breaks one of them while satisfying the other.
#:
#: Bounded by COUNT rather than by liveness, and that is not laziness. "Drop an entry no
#: live session's manager names" looks tighter and is wrong: in a line whose middle AND
#: upper hops have both departed, no live session names the upper one -- only the
#: departed middle one did -- so that rule deletes the hop the walk needs and the top
#: manager loses reach to the bottom session. ``test_two_departed_hops_in_a_row_are_both
#: _read_through`` is that case, and it fails under the liveness rule.
#:
#: How many keys the cap has dropped, in a one-element list so the counter can be read
#: and bumped without a module-level ``global``. Counted because this bound is reached by
#: ORDINARY use -- every slot construction takes a key -- and because what it costs is
#: silent: the walk refuses a line through a dropped key and the fence answers the same
#: ``not_creator`` an ownership failure answers, so without a count a lead that has lost
#: reach over a closed conductor's workers is indistinguishable from one that never had
#: it. A bound that is reached has to be observable, not merely safe.
_key_occupancy_evictions: "list[int]" = [0]

#: The keys the cap has dropped most recently, so a refusal can NAME this hop's record
#: as evicted rather than never written. Bounded by the same number as the table it
#: shadows, and it holds keys only -- which are already length-bounded by
#: ``_normalize_slot_key`` -- so it retains nothing the occupancy table did not.
#:
#: COSMETIC ONLY, and it has to stay that way. This table is itself evictable, so a key
#: can age out of it as well as out of the occupancy table; if the walk's answer for a
#: gap depended on finding the key HERE, that answer would change when this table
#: forgot, which is a boundary resting on an evictable cache. So the walk asks
#: :data:`_key_occupancy_evictions` -- a counter nothing evicts -- whether this process
#: has forgotten anything at all, and uses this table only to say WHICH key when it
#: still knows.
_recently_evicted_keys: "OrderedDict[str, None]" = OrderedDict()

#: What eviction costs, which is REACH and not a boundary -- it is worth saying plainly
#: because an earlier version of this comment documented the opposite as an accepted
#: residual, and a reader could re-derive the hole from it. Evicting a key removes the
#: occupancy the walk resolves against, so every departed hop under that key stops the
#: walk and the fence refuses. Nothing is admitted by an eviction.
#:
#: The shape that comment used to admit was a late close for an older occupant writing
#: its stale manager back once the current occupant's entry had been evicted, and the
#: walk then handing that manager the replacement's sessions. Two things now refuse it
#: independently: the read matches the pair EXACTLY, so an older occupant's entry is not
#: a thing the walk can reach for; and the write refuses outright when the writer's
#: generation is not the key's occupancy at write time, so the stale entry is never
#: created. Raising this number would only narrow the window it used to need -- it is
#: not what closed it, and it is not what keeps it closed.
_KEY_OCCUPANCY_MAX = 512


def _occupancy_generation_of(slot: Any) -> int:
    """Which occupant of its key *slot* is, or ``0`` for an object that cannot say.

    ``0`` is below every real generation (the counter starts at 1), so an object with
    no such field is never PROVEN to be an obsolete occupant and its departure records
    normally -- which is the direction that keeps a test double, or any future slot-like
    object, behaving as it did before the fence existed. Every ``_ChatSlot`` carries the
    field unconditionally from its constructor, so the fence is never inert for a real
    slot.

    An attribute read and nothing else, for the reason :func:`manager_of` is.
    """
    generation = getattr(slot, "_occupancy_generation", 0)
    return generation if isinstance(generation, int) else 0


#: The slot object holding each key, weakly. Liveness, answered without the live table:
#: this module is called from ``_ChatSlot.__init__`` and has no state to ask, and the
#: paths that END an occupancy are several -- some of them a bare ``_slots.pop`` that
#: tells this module nothing. A weak value answers both: while the slot is alive the key
#: reads as held, and when the last reference goes the entry disappears on its own, so a
#: pop nobody told us about costs nothing and leaks nothing.
_live_occupants: "weakref.WeakValueDictionary[str, Any]" = weakref.WeakValueDictionary()


#: Whether this process has READ closed hops' lines from the log yet, in a one-element
#: list so it can be set without a module-level ``global``. False until the boot pass
#: finishes, and that window is the difference between "no record of a line" and "a line
#: that ended": the first is what a gap means before the pass, the second after it. The
#: reader loses nothing by waiting; the loop guard, told a hop has nothing above it,
#: writes an edge that becomes a persisted cycle once the pass restores the real manager.
_departures_rebuilt: "list[bool]" = [False]


#: The manager-line switch as the last ASYNC reader saw it, in a one-element list so a
#: sync caller can read it without a config load. ``None`` until something has read it.
#:
#: The bookkeeping this module does -- an occupancy per slot construction, a record per
#: close, a disk read per closed hop at boot -- is gated on it, because with the switch
#: off nothing reads what those record and work nobody reads should not run. The gate
#: cannot be a config load: the writers run from ``_ChatSlot.__init__`` and from close
#: paths on the event loop, and a load there is the blocking call this feature has
#: already been pulled up on twice. So the ASYNC readers publish what they read and the
#: writers consult that.
#:
#: Default ``None`` means OFF, which is the shipped default and the fail-quiet
#: direction: nothing is recorded, every line through a closed hop refuses, and that is
#: exactly what the gateway answers without this feature. Turning the switch on starts
#: the writes from the next async read; lines through sessions that closed while it was
#: off refuse, which costs reach and moves no boundary.
_chain_switch_seen: "list[bool | None]" = [None]


def note_chain_switch(value: bool) -> bool:
    """Publish what an async reader just read, for the sync writers to consult."""
    _chain_switch_seen[0] = bool(value)
    return bool(value)


def _chain_recording() -> bool:
    """Whether this process should be recording the manager line at all."""
    return _chain_switch_seen[0] is True


def _evictable(key: str, entry: "_Departure | None") -> bool:
    """Whether the cap may drop *key*: only a DEPARTED hop's own settled line.

    Three kinds are kept whatever the cap says, and all three for one reason -- what
    they hold is not reach across a closed hop, which a cap may cost, but a fact whose
    absence reads as a safe answer.

    A key whose occupant is still ALIVE. That session could not then record its own
    departure, because the writer checks its generation against the key's and finds
    another, so the line would go unreadable the moment it closed.

    An entry with no ``tab``: a reuse TOMBSTONE, the one record saying this key has been
    held by somebody else. Dropped, the next occupant of the name reads as pristine and
    every edge still naming the key resolves through it, which hands a deleted session's
    descendants to whoever opened a tab under its name. One tombstone per key however
    often the name is taken, and two short strings each.

    A PENDING entry, which is a close still in flight.

    Everything else goes, INCLUDING a key with no entry and no live occupant -- a bare
    pop that told this module nothing, or a key from before a restart. Those hold no
    answer to lose, and refusing to reclaim them is how the table grows without bound:
    one key per request on a surface that mints a slot per call.

    A reuse mark is NOT in the kept list, and that is the whole reason it does not live
    here. Keeping one per reused key made the cap stop bounding the table: close, resume
    and close enough distinct sessions and the entries outlive every cap. So the mark
    sits on the LIVE OCCUPANT instead (``_displaced_tab``), where it is bounded by the
    number of sessions that can be open at once and cannot be evicted at all, and this
    table is left holding only lines a cap may reclaim.
    """
    # THE RECORD DECIDES FIRST, and the order matters. A settled entry is proof the
    # occupant departed, whatever still holds a reference to the object -- and the
    # reference is a proxy, not the fact. Asked the other way round, a closed session
    # somebody still holds becomes unevictable and the cap stops reclaiming the very
    # lines it exists to reclaim.
    if key in _keys_a_live_edge_names():
        # STILL REFERENCED. An edge that names only a key cannot say which occupant it
        # meant, and the record under that key is the only thing that can -- so while
        # some open session's own edge still names it, dropping the record turns the
        # next occupant of the name into the one every such edge resolves through.
        #
        # What bounds this is the live table: at most two keys per open session, and a
        # key stops being referenced when the session naming it closes.
        return False
    if entry is not None:
        return not entry.pending
    # No entry at all: live, a bare pop that told this module nothing, or a key from
    # before a restart. Only the first must be kept, and the weak reference is what
    # tells them apart -- it goes when the slot does, so the other two cost nothing.
    return key not in _live_occupants


def _keys_a_live_edge_names() -> "set[str]":
    """Every key a line from an OPEN session can still reach, to the end of the line.

    Read off the weak occupant map, which is this module's own view of what is open --
    no state to ask and nothing to keep in step, and an entry disappears when its slot
    does. Both edges count: a published handoff names a key, and so does a creation
    chain, and the walk follows either.

    AND THE KEYS THOSE RECORDS THEMSELVES NAME, climbing while entries answer. Stopping
    at the keys a live slot names DIRECTLY was a grant, not a retention limit. A creates
    B, B creates W; A is deleted and B closes, and B's record is kept because the open W
    names it -- while A, named only by B's record, is not. Ordinary churn then drops A
    from the occupancy table and from the eviction shadow after it, and a key absent
    from both reads as a name nobody has ever held: the next session to open under A's
    name is marked as having displaced nobody, every creation edge above B resolves
    through it, and a replacement inherits the sessions the original was running. The
    chain the walk can follow and the chain this cap protects have to be the same chain.

    Bounded by the WALK, not by the table: a kept key is one some open session's line
    reaches, so the set is at most the open sessions times the depth bound, both bounded
    elsewhere. A chain stops at the first hop with no entry -- already evicted, or never
    recorded -- and a cycle stops on the keys already collected.
    """
    named: "set[str]" = set()
    for key, slot in list(_live_occupants.items()):
        # A settled record for this occupant's OWN key means it departed, whatever
        # still holds a reference to the object -- the same order :func:`_evictable`
        # uses, and for the same reason: the reference is a proxy and the record is the
        # fact. A departed session's edges are not what keeps another key's record
        # alive; its own record is, and the cap may take that.
        own = _departed_managers.get((key, _key_occupancy.get(key)))
        if own is not None and not own.pending:
            continue
        for attribute in ("_manager", "_created_by"):
            value = getattr(slot, attribute, "")
            if isinstance(value, str) and value:
                named.add(value)
    # UPWARD through the records that are kept, because that is what the walk does. A
    # departed hop resolves to its own recorded manager and the walk climbs to it, so
    # every key on that chain is one a live session can still reach -- and the one the
    # cap drops is the one an edge then resolves through whoever opens the name next.
    frontier = list(named)
    while frontier:
        hop = frontier.pop()
        entry = _departed_managers.get((hop, _key_occupancy.get(hop)))
        if entry is None or not entry.manager or entry.manager in named:
            # No record for this hop (live, evicted, or never written), a line that
            # ended, or a key already collected -- including the second visit to a
            # key in a cycle, which is what bounds this against a persisted loop.
            continue
        named.add(entry.manager)
        frontier.append(entry.manager)
    return named


def note_key_taken(taken_key: str, generation: int, occupant: Any = None) -> None:
    """Record that *generation* is the occupant holding *taken_key* from now on.

    Called from ``_ChatSlot.__init__``, where a key is TAKEN. Recording the occupancy at
    the take rather than on the departure paths is what makes a reusable key safe
    structurally: a departure path that forgets something can only fail to record ITS
    OWN entry, which costs reach, while a key whose occupancy went unrecorded would let
    the walk read a previous occupant's entry, which grants it.

    Drops the entry of the occupant being replaced, so a record never outlives the
    occupancy it describes, and evicts oldest-first to hold the one size bound. Neither
    is what makes the read safe -- the exact pair match is -- but together they keep the
    table free of entries nothing can reach.

    Occupancy only ever moves FORWARD. The counter is monotonic and a key's occupants
    are constructed in order, so an equal or lower number is either the same object
    announcing twice or an ordering that cannot happen; keeping the later occupant is
    the direction that never lets an earlier one take a key back.

    NOT GATED ON THE SWITCH, and this is the one writer that must not be. A take is the
    only moment the occupancy is knowable, and the moment keys are taken in BULK is the
    restore at startup -- which runs before any async reader has published the switch,
    so a gate here reads "off" for every restored session and files nothing. Each of
    them then closes with no occupancy to key a departure on, the write refuses because
    the key's occupancy is not its own, and the walk reads that hop as the end of a line:
    after every restart a lead loses reach over a closed restored conductor's workers,
    which is the case this feature exists for. The departure writers ARE gated, because
    a departure is a statement about the line and its path can read the switch; this is
    a fact about a take, in a bounded in-memory table, that nothing reads with the line
    off and nothing can reconstruct later.
    """
    if not taken_key:
        return
    if occupant is not None and taken_key not in _key_occupancy:
        if taken_key in _recently_evicted_keys:
            # A KEY THE CAP DROPPED, taken again. The occupancy that would have said
            # "this name has been held before" is exactly what the cap reclaimed, so
            # finding nothing here is not the same fact as the key having been free --
            # and read as free, every edge still naming it resolves through this new
            # occupant, which hands a deleted session's descendants to whoever opened a
            # tab under its name.
            #
            # The shadow is what remembers, and it is consulted ONLY for a key with no
            # occupancy: a key the cap never touched is unaffected, so this costs
            # nothing to the ordinary case. What it cannot cover is a key dropped so
            # long ago that the shadow has forgotten it too, which is the residual this
            # mechanism has and a persisted creator identity would not.
            occupant._displaced_tab = ""
    if occupant is not None:
        # WEAKLY, so the paths that end an occupancy without telling this module -- a
        # bare ``_slots.pop`` on a per-request surface -- cost nothing: the entry goes
        # when the slot does.
        try:
            _live_occupants[taken_key] = occupant
        except TypeError:  # pragma: no cover - an object that cannot be referenced
            pass
    held = _key_occupancy.get(taken_key)
    carried_occupant = ""
    reuse = False
    if held is not None:
        if held >= generation:
            return
        # The earlier occupant's entry goes, but its OCCUPANT identity is carried into a
        # reuse tombstone written below: a session resuming its own key under the same
        # id has to be told from a different session taking it, and that comparison is
        # against the id the key was last known by. Nothing else about the key's history
        # is kept -- an edge naming this key carries the identity it was written for, so
        # a new occupant is told apart by comparison, not by remembering the count of
        # takes. That is what makes the capped table safe: dropping an entry can cost
        # reach across a closed hop and cannot turn into a grant.
        prior = _departed_managers.pop((taken_key, held), None)
        carried_occupant = prior.occupant if prior is not None else ""
        reuse = True
        if occupant is not None:
            # ON THE OCCUPANT, because this is the one fact whose absence grants reach
            # and it must not sit in a capped table: an edge that names only a key --
            # a creation edge carries no occupant to compare -- is ambiguous once the
            # key has been held by somebody else, and a mark the cap dropped reads as
            # "this key is pristine". Here it lasts exactly as long as the occupant
            # holds the key, with no cap to outlive.
            #
            # The value is WHO the key was taken from, so the one session that may
            # still resolve through it is the one resuming its own key.
            occupant._displaced_tab = carried_occupant
    _key_occupancy[taken_key] = generation
    _key_occupancy.move_to_end(taken_key)
    while len(_key_occupancy) > _KEY_OCCUPANCY_MAX:
        # OLDEST FIRST, but only among keys whose occupant has DEPARTED. Evicting in
        # take order alone drops the occupancy of sessions that are still open, and a
        # live session whose occupancy is gone cannot record its own departure at all --
        # the writer checks its generation against the key's and finds another. So a
        # conductor that stays open while 512 other tabs are built silently stops being
        # readable the moment it closes, and the lead's reach over its workers is
        # refused as a retention limit that never applied to it.
        #
        # A key with no departure entry under its current occupancy is either live or
        # from before this process. Neither is safe to forget, so the scan skips both
        # and the cap yields: what bounds the table then is the number of sessions that
        # can be open at once, which is bounded elsewhere, and each entry is two short
        # strings. A cap is allowed to cost reach across a CLOSED hop; it is not allowed
        # to break the record of a session that is still running.
        evictable = next(
            (
                key
                for key, gen in _key_occupancy.items()
                if _evictable(key, _departed_managers.get((key, gen)))
            ),
            None,
        )
        if evictable is None:
            logger.info(
                "manager line: the %d-key cap is full of occupancies whose sessions are "
                "still open, so nothing is dropped -- the table holds %d keys until one "
                "of them closes",
                _KEY_OCCUPANCY_MAX,
                len(_key_occupancy),
            )
            break
        evicted_generation = _key_occupancy.pop(evictable)
        evicted_key = evictable
        _departed_managers.pop((evicted_key, evicted_generation), None)
        # COUNTED AND NAMED, because this bound is reached by ordinary use: every slot
        # construction takes a key, so a long-running gateway passes the cap without
        # anything unusual happening. What an eviction costs is reach across that key's
        # departed hop, and the refusal it produces is the same ``not_creator`` an
        # ordinary ownership failure produces -- so with nothing recorded here, a lead
        # that has quietly lost reach over a closed conductor's workers looks exactly
        # like a caller that never had it. The count is what tells those apart.
        _key_occupancy_evictions[0] += 1
        _recently_evicted_keys[evicted_key] = None
        _recently_evicted_keys.move_to_end(evicted_key)
        while len(_recently_evicted_keys) > _KEY_OCCUPANCY_MAX:
            # The shadow is bounded too, so it can forget WHICH keys were dropped. That
            # costs a REFUSAL ITS WORDING and nothing else: an edge carries the identity
            # it was written for, so a key with no record of any kind is refused by the
            # gap it leaves, not resolved through whoever holds it now. The identity
            # question never rests on this table.
            _recently_evicted_keys.popitem(last=False)
        logger.info(
            "manager line: the departure record for %s was dropped by the %d-key cap "
            "(%d evicted so far); a line through it now refuses, so reach across that "
            "closed session is gone until it is reopened",
            evicted_key,
            _KEY_OCCUPANCY_MAX,
            _key_occupancy_evictions[0],
        )
    if reuse:
        # A reuse TOMBSTONE under the NEW occupancy. ``_entry`` returns it, so a line
        # through this departed-and-reused key refuses rather than resolving an edge
        # that no longer says which occupant it meant; ``_unidentifiable`` is True
        # because ``tab`` is cleared, so no edge above can match it; and ``_recorded``
        # is ``""`` because it names no manager. The PRIOR occupant's id rides along in
        # ``occupant`` so a session resuming its own key is still recognised as itself
        # the moment it answers the key live. Written after the eviction sweep so this
        # mark is the freshest entry rather than a candidate for the drop it triggered.
        _record_departure(
            taken_key, generation, "", pending=False, tab="", occupant=carried_occupant
        )


def claim_occupancy_for_rebuilt_key(rebuilt_key: str) -> int:
    """Give *rebuilt_key* an occupancy generation, for a key with no slot to carry one.

    The boot rebuild files departures for keys whose session has CLOSED, so there is no
    ``_ChatSlot`` to read a generation off. The record is keyed on the pair, so the key
    still needs one, and it comes from the same monotonic counter every live slot draws
    from -- so a rebuilt key and a slot minted afterwards can never collide, and a later
    occupant of the key is still a LATER generation than the rebuilt one.
    """
    from kiro_crew.dashboard.state import _SLOT_OCCUPANCY_GENERATIONS

    generation = next(_SLOT_OCCUPANCY_GENERATIONS)
    note_key_taken(rebuilt_key, generation)
    return generation


def note_hop_left_unreadable(state: "DashboardState", hop_key: str) -> bool:
    """Record that this gateway could NOT decide *hop_key*'s line. Answers whether it did.

    The boot pass writes an entry for a hop it resolved, and a DECIDED empty answer --
    "this session reported to nobody" -- is an entry too, naming nobody. A hop it could
    not read was left with neither, and a key with no entry at all is what the walk
    reads as a line that ENDED once the pass has finished. So the two came out the same:
    a hop whose metadata would not open, whose identity the line does not name, or whose
    recorded manager is not admissible, read as a session with nothing above it.

    For a reader that costs nothing -- both answers refuse reach. For the handoff's LOOP
    GUARD it is the defect this whole distinction exists for: told that nothing sits
    above the hop, the guard permits an edge, and the hop's real manager -- which the log
    does hold and this process could not read -- closes a cycle that is then persisted.

    So the occupancy is claimed and NO entry is filed, which is the one combination the
    walk already answers correctly: it finds an occupancy, finds no record under it, and
    answers "this gateway has no usable record of that line" -- unread for both
    polarities, not a wait, and answered one level by the queue drain so an accepted
    message is never destroyed over it. No new table, no new cause, no new wording.

    Refuses to touch a key with a LIVE occupant or an occupancy already on record: the
    pass runs as a background task on a serving gateway, and a resume landing inside the
    read window holds the key by now. Claiming it would move the occupancy off that
    session and disable every departure it later records.
    """
    if not hop_key or state.get_slot(hop_key) is not None or hop_key in _key_occupancy:
        return False
    claim_occupancy_for_rebuilt_key(hop_key)
    return True


#: "Keep whatever the prior entry said", for the two identity fields of
#: :class:`_Departure`. A distinct object rather than ``None``, because ``None`` is a
#: MEANINGFUL value of ``manager_tab`` -- it is "a creation edge" -- and a
#: sentinel that collides with a real value is how a sticky field quietly un-sticks.
_KEEP: Any = object()


def _short_enough_to_keep(value: "str | None | Any") -> "str | None":
    """*value* if a bounded table may retain it, ``""`` when it is too long.

    ``None`` passes through: that is a STATEMENT -- a creation edge names no occupant --
    and not a string being retained. Everything else is bounded by the slot-key limit,
    which is the right one because every value this guards is a short gateway-minted
    name, and nothing on the way in bounds it.
    """
    if value is None:
        return None
    text = str(value)
    if len(text) > _SLOT_KEY_MAX_CHARS:
        logger.warning(
            "manager line: a retained identity of %d characters is past the %d bound, "
            "so it is kept as no identity at all and lines through that key refuse",
            len(text),
            _SLOT_KEY_MAX_CHARS,
        )
        return ""
    return text


def _record_departure(
    key: str,
    generation: int,
    manager: str,
    *,
    pending: bool,
    manager_tab: "str | None | Any" = _KEEP,
    tab: "str | Any" = _KEEP,
    occupant: "str | Any" = _KEEP,
    store: "str | Any" = _KEEP,
) -> bool:
    """Write the entry for *generation* of *key*, if that occupant still holds the key.

    *pending* says whether what this occupant reports to is settled. The pop writes a
    PENDING entry and the committed side overwrites it with a settled one, so the walk
    can tell a manager not yet published from a session that reported to nobody.

    The one writer of :data:`_departed_managers`, and the write-side half of the fence:
    an occupant whose generation is no longer the key's occupancy AT WRITE TIME records
    nothing at all, so a close finishing late for a session the key has moved on from
    does not leave a stale entry behind to be found. The read does not depend on this --
    it matches the pair exactly and would not reach such an entry either way -- which is
    the point of having both. Returns whether it wrote.
    """
    if _key_occupancy.get(key) != generation:
        return False
    if len(manager) > _SLOT_KEY_MAX_CHARS:
        # A COUNT bound does not bound the memory a table holds: 512 entries each
        # carrying an arbitrarily long string is 512 arbitrarily long strings, and this
        # value arrives from restored history metadata, where ``created_by`` is read
        # back with no length check of its own. The slot-key bound is the right one to
        # apply, because that is what this field is -- a slot key naming a manager.
        #
        # Rejected rather than truncated, and nothing is written at all. Truncating an
        # authorization key would invent a DIFFERENT key, which either names nobody or,
        # worse, names some other session; and a value no real slot key can equal is not
        # a manager the walk should resolve to. With no entry the hop is unresolved, so
        # both polarities refuse -- the loss is reach across one departed hop, never a
        # boundary, which is the direction everything else here fails in.
        logger.warning(
            "manager line: a departure for %s names a manager of %d characters, past "
            "the slot-key bound, so the hop is left unresolved",
            key,
            len(manager),
        )
        return False
    # The tombstone is STICKY. Who this occupant took the key from is a fact about the
    # take, so this occupant recording its own manager must not clear it -- the edge
    # that names this key may well have been written for the occupant before it, and
    # nothing in the edge says otherwise.
    prior = _departed_managers.get((key, generation))
    if manager_tab is _KEEP:
        manager_tab = prior.manager_tab if prior is not None else None
    if tab is _KEEP:
        tab = prior.tab if prior is not None else ""
    if occupant is _KEEP:
        occupant = prior.occupant if prior is not None else ""
    if store is _KEEP:
        store = prior.store if prior is not None else ""
    # BOUNDED WHERE THEY ARE RETAINED, for the reason the manager key above is: a count
    # cap bounds how many entries this table holds and says nothing about how long each
    # one's strings are. These arrive from a persisted line and from a harness, and
    # over-long becomes ``""`` rather than a truncation -- a shortened identity is a
    # DIFFERENT identity and might be another session's, while ``""`` refuses.
    _departed_managers[(key, generation)] = _Departure(
        manager,
        pending,
        _short_enough_to_keep(manager_tab),
        _short_enough_to_keep(tab) or "",
        _short_enough_to_keep(occupant) or "",
        _short_enough_to_keep(store) or "",
    )
    _departed_managers.move_to_end((key, generation))
    return True


def note_key_vacated(vacated_key: str, vacated_slot: Any) -> None:
    """Invalidate *vacated_key*'s departure edge the moment the key leaves the table.

    Called at the POP, by each path that has awaits between taking the slot out and
    committing the departure. Between those two points the key is absent from the live
    table, so the walk stops reading :func:`manager_of` off a live slot and starts
    reading the record -- and the record still describes whichever occupant held the
    key BEFORE this one, because the manager of this one is published only once its
    close commits. A key that has been reused therefore hands its previous occupant's
    manager reach over the sessions the CURRENT occupant opened, for as long as the
    close takes.

    So the edge is cleared here, under the CURRENT occupancy: the entry names nobody,
    and the walk refuses at that hop exactly as it refuses a hop with no exact entry.
    What the window costs is reach, for the length of one close, and it costs it to the
    session that is closing anyway. The previous occupant's entry is not involved --
    it went when this occupant took the key, and the exact match would not reach it in
    any case.

    Nothing is published about this slot's own manager. That stays with
    :func:`note_manager_departed` on the committed side, so a close that fails and puts
    the tab back has still handed nothing up -- the entry it leaves resolves to nobody,
    and the slot is live again and answered from the table.
    """
    if not vacated_key or not _chain_recording():
        return
    # PENDING, not "nobody". The difference is what stops the handoff loop guard from
    # treating this closing manager as the top of a line and writing an edge under it
    # that closes a cycle once the close commits. See :class:`_Departure`.
    #
    # The occupant's own SESSION ID is captured here, at the pop, because this is the
    # last point the slot is in hand -- and it is what the next occupant of this key is
    # compared against. A resume finds its own id here and resolves; a different
    # session finds another and is refused.
    #
    # When this occupant itself took the key from an earlier one, its construction left
    # a reuse TOMBSTONE (empty ``tab``) under this occupancy, and that mark has to
    # survive the pop: the key is reused whatever this occupant goes on to do, so the
    # cleared tab is carried forward rather than overwritten with this slot's own.
    generation = _occupancy_generation_of(vacated_slot)
    identity = str(getattr(vacated_slot, "_tab_id", "") or "")
    prior = _departed_managers.get((vacated_key, generation))
    reused = prior is not None and not prior.tab
    _record_departure(
        vacated_key,
        generation,
        "",
        pending=True,
        tab="" if reused else identity,
        occupant=prior.occupant if reused else identity,
        # THE PRINCIPAL, captured here because here is where the slot is still in hand.
        # The walk refuses a departed handed edge whose two sides disagree on a store,
        # and a record that never carried one compares ``""`` against ``""`` and passes:
        # the field existed, the writers did not fill it, and an unwritten fact read as
        # a safe answer is the defect this whole line keeps producing. The reverse broke
        # too -- two sides on the same non-default store compared ``"X"`` with ``""``
        # and refused every departed hop.
        store=str(getattr(vacated_slot, "memory_store", "") or ""),
    )


def _forget_unsettlable_entry(key: str, slot: Any) -> None:
    """Drop the PENDING entry the pop wrote, for a close that will never settle it.

    A pop writes a pending entry so the window between it and the commit refuses on an
    entry of its own rather than on the absence of one, and the commit overwrites it.
    Two of those commits record nothing at all -- a manager that was never read, and a
    grant that is not admissible -- and the pending entry they leave behind is then
    permanent: the walk reads it as a line about to be published, every handoff under
    it is refused as ``manager_line_pending``, and the message tells the caller to retry
    something that cannot land. The hop must still refuse. It must stop CLAIMING it is
    about to clear.

    So the entry goes, which leaves the gap a key with no record leaves, and the walk
    refuses that without promising anything -- the same answer an evicted record gets,
    for the same reason: nothing here is known.
    """
    generation = _occupancy_generation_of(slot)
    if _key_occupancy.get(key) != generation:
        # Not this occupant's key any more, so the entry under that pair is not this
        # close's to remove -- the same fence every writer here applies.
        return
    _departed_managers.pop((key, generation), None)


def note_manager_departed(state: "DashboardState", departed_key: str, departed_slot: Any) -> str:
    """Record what *departed_slot* reported to, so the walk can read through it later.

    Called by each path that takes a live slot OUT of the table -- a tab close, the
    idle-archive sweep, a history delete -- on that path's committed side, with the slot
    still in hand. It is the whole of what a departure does to the reporting line.

    Synchronous, one mapping write, no save and no loop over other slots. That is the
    point: there is no second writer to exclude, no report whose manager could change
    between a check and a stage, and no promoted manager whose identity could go stale
    across a save, because none of those steps happen.

    Records the departed session's OWN manager, read with :func:`manager_of` while the
    slot object is still available. A session that reported to nobody records NOBODY --
    there is no hop above it to resolve to, and the walk stopping there is correct.

    RECORDED AGAINST ITS OWN OCCUPANCY, because a close is not instantaneous and a slot
    key is reusable. The paths that call this take the slot out of the live table FIRST
    and then keep awaiting -- a task cancel with its own timeout, then a save -- so the
    key is free for a whole window before the departure is recorded. A replacement
    minted in that window can open sessions of its own and depart in turn, and the
    original close then arrives LAST with a manager that has nothing to do with the
    key's current line. Recorded under the KEY, that close's manager becomes the answer
    for the replacement's sessions -- reach over sessions it never managed, with nothing
    to take it back.

    Recorded under the PAIR, it is not an answer to any question the walk asks, because
    the walk resolves the occupancy first and matches the pair exactly. And this writer
    refuses outright when its generation is not the key's occupancy at write time, so
    the entry is not created either. Those are two independent refusals on purpose: the
    exact match is what holds even if a writer is ever added that forgets to check, and
    the write check is what keeps the table free of entries nothing can reach -- neither
    is load-bearing alone, and the ordering the closes arrive in stops mattering.

    So an entry carries the generation of the occupant it is about (see
    :func:`_record_departure`). The entry for a departure with nobody above it is KEPT,
    naming nobody, rather than dropped, because an ABSENT entry means something else:
    an occupant still inside its own close, which the walk refuses rather than reading
    to the end.

    The OTHER half of that window is :func:`note_key_vacated`, called at the pop, which
    publishes nobody under the current occupancy so the window refuses on its own entry
    rather than on the absence of one.

    Returns the value recorded, or ``""`` when nothing was -- including when the key's
    occupancy has moved on from this slot.
    """
    if not departed_key or not _chain_recording():
        return ""
    # ``is True`` and not truthiness, for the reason :func:`_occupancy_generation_of`
    # checks ``isinstance``: a slot-like double answers every attribute with a truthy
    # object, and reading that as "unresolved" would silently stop recording departures
    # for every such object. A real slot sets this to a bool.
    if getattr(departed_slot, "_manager_grant_inadmissible", False) is True:
        # Its manager was READ and may not be published -- closed, or under another
        # principal's store. The live walk refuses such a hop; recording its manager
        # here would publish the same refused grant as a SETTLED edge that outlives the
        # slot carrying the refusal, so the walk would follow after the close what it
        # declined to follow before it. Nothing is recorded and the hop stays
        # unresolved, which costs reach across one departed session.
        logger.info(
            "manager line: %s departed with a manager that is not admissible, so no "
            "departure is recorded and a line through it refuses",
            departed_key,
        )
        _forget_unsettlable_entry(departed_key, departed_slot)
        return ""
    if getattr(departed_slot, "_manager_unresolved", False) is True:
        # The base PR's default-deny, carried into the record rather than only into the
        # verbs. ``_manager_unresolved`` says no decided manager has been read from the
        # log for this slot yet, so ``manager_of`` would fall back to the CREATOR -- and
        # an empty ``_manager`` means "the creator holds it" only once that read has
        # happened. Recording the creator here would turn "not known yet" into a
        # decided answer that outlives the refusal session control makes for the verbs,
        # and the walk would then resolve through a manager nothing established.
        #
        # So nothing is recorded and the hop stays unresolved, which costs reach across
        # a session that departed before its manager was read and grants nobody
        # anything.
        logger.info(
            "manager line: %s departed before its manager was read from the crew log, "
            "so no departure is recorded and a line through it refuses",
            departed_key,
        )
        _forget_unsettlable_entry(departed_key, departed_slot)
        return ""
    generation = _occupancy_generation_of(departed_slot)
    if generation == 0 and departed_key not in _key_occupancy:
        # An object that cannot say which occupant it is -- a test double, or any
        # future slot-like object. Nothing can prove it obsolete and no other entry can
        # be filed under its key, so it anchors its own occupancy and records, which
        # keeps such an object behaving as it did before the pair key existed. Guarded
        # on the key being UNKNOWN: a key that has a real occupancy must not be taken
        # over by an object that cannot name one.
        note_key_taken(departed_key, generation)
    # The COMMITTED manager, through :func:`manager_of`, which reads ``_manager`` and
    # then ``_created_by`` and no third field -- the slot carries no pending-manager
    # field at all, and that is what makes this structural rather than careful. Were
    # one reintroduced and read here, the record could name a manager the disk does
    # not agree with, and the walk would admit reach on the strength of a write that
    # may yet be withdrawn. A session whose handoff was in flight when it departed is
    # recorded under the manager it actually had.
    above = manager_of(departed_slot)
    if above == departed_key:
        # Names ITSELF, so there is no hop above it to resolve to. Recorded the same
        # way a departure with an empty manager is: as nobody.
        above = ""
    # Written under THIS occupant's pair, or not at all. An earlier session that held
    # the same key is neither consulted nor overwritten: its entry went when this
    # occupant took the key, and had it survived it would still not be a pair the walk
    # looks up. A departure with nobody above it says exactly that, as an entry, since
    # an absent entry means an occupant still inside its close.
    #
    # Its own session id goes in with it, for the same reason the pop records one: the
    # next occupant of this key is told apart from this one by comparing the two, and a
    # close that commits without the slot in hand could not read it afterwards.
    #
    # If this occupant took the key from an earlier one, its construction left a reuse
    # TOMBSTONE (empty ``tab``) under this occupancy. The departed MANAGER is still
    # recorded -- the record being right is useful -- but the cleared tab is carried
    # forward, because the key is reused whatever manager this occupant had, and an edge
    # above naming the key still cannot say which occupant it meant.
    identity = str(getattr(departed_slot, "_tab_id", "") or "")
    prior = _departed_managers.get((departed_key, generation))
    reused = prior is not None and not prior.tab
    if not _record_departure(
        departed_key,
        generation,
        above,
        pending=False,
        tab="" if reused else identity,
        occupant=prior.occupant if reused else identity,
        manager_tab=_edge_identity(departed_slot),
        # Its PRINCIPAL too, read off the slot while it is still in hand. The walk
        # compares it against the next hop's, and a record that never carried one made
        # that comparison answer ``""`` against ``""`` -- which admits a manager since
        # rebound to another store over everything below a closed hop.
        store=str(getattr(departed_slot, "memory_store", "") or ""),
    ):
        return ""
    return above


#: The operations a handoff widens, and the whole of what it widens. A session handed
#: to a new manager can be driven and put down by that manager -- which is the point,
#: since a conductor that cannot send to a worker cannot run it -- and nothing else
#: moves with it.
#:
#: What is NOT here is the part that makes the field safe to add. ``create`` and
#: ``fork`` mint a session, and the minted session INHERITS the caller's memory
#: binding, so admitting a manager there would hand it the giver's memory store
#: through a session it was given rather than one it made. The memory delegation gate
#: (:func:`_delegation_lineage_fenced`) reads ``_created_by`` alone for the same
#: reason, and ``_created_by`` is never written again, so a handoff cannot reach it.
#: ``rename``, ``set_model``, ``retry`` and ``end_wait`` are left out on a narrower
#: ground: nothing a manager needs in order to RUN its reports requires them, and a
#: verb is widened here when a report cannot be managed without it.
HANDOFF_WIDENED_OPERATIONS: frozenset[str] = frozenset(
    {
        "send",
        "read",
        # Authorized as a transcript read is authorized, so it moves with ``read``: a
        # manager that may read a report's whole transcript is not withheld the cached
        # one-line summary of it, and the giver that handed the session away keeps no
        # window onto it either.
        "summary",
        "stop",
        "close",
        "adopt",
        "release",
        # The transfer itself, because the design puts the transfer in the hands of
        # the CURRENT manager: a session handed on once has to be handable on again,
        # or the first handoff would be the last. The receiver is authorized
        # separately and on its own terms, so this widens WHO may give a session away
        # and never WHICH sessions can be given.
        "handoff",
    }
)


def _app_owned_cron_refusal(state: "DashboardState", caller_key: str) -> tuple[str, str] | None:
    """``(message, code)`` when *caller_key* is an APP's cron, else ``None``.

    An app-scoped SESSION is refused by the ``_app`` check that sits beside every
    call site of this one, but a cron tab does not carry that tag:
    ``inject_cron_result_to_dashboard`` mints it without ``app=``, so an app's own
    scheduled job reaches this surface with ``_app == ""`` and would pass. Left
    unchecked, an app could create a persistent, sidebar-visible session that is
    NOT app-scoped -- precisely the confinement escape the ``_app`` refusal exists
    to prevent, reached through the app's scheduled job instead of its session.

    App ownership therefore has to be read from the JOB, and it has TWO spellings
    there because two writers record it differently: the app cron SDK tags
    ``created_by = "app:{app_name}"``, while ``mcp_cron``'s own ``cron_add``
    records the calling session in ``session_key`` and never writes ``created_by``
    at all -- so an app-scoped session's job carries its authority only in the
    second. Both are checked, and the second delegates to ``_app`` on the owning
    slot rather than re-deriving app-ness, so there is one definition of "is this
    an app" and not a third. **A new field on the job that can name a principal is
    a hole here until it is added to this function.**

    Fail-CLOSED when ``session_key`` names a closed session: its
    ``_app`` cannot be read, and "could not verify the owner is not an app" must
    not read as "has no owner". ``mcp_cron``'s ``cron_add`` records an app's
    authority ONLY in ``session_key`` -- so once that slot is gone, allowing the
    job would let an app escape confinement through a cron it authored and then
    abandoned by closing its session. The cost is a genuinely user-created
    dispatching cron whose authoring tab has closed is refused too; that caller
    can reopen a tab, whereas an app session minted outside its confinement cannot
    be undone. This is the same fail-closed direction the missing-job case below
    takes, and what still bounds the app case beyond it is that a LIVE app session
    has its jobs refused directly by the ``_app`` read.

    Fail-CLOSED on a job that cannot be found, or a registry that cannot answer:
    "could not verify the owner" must not read as "has no owner", the same
    direction ``agent_unverifiable`` takes on its own unreadable input. Nothing
    legitimate is refused by it, because a cron whose job is gone is not running.

    Returns rather than raises so each call site keeps its own idiom -- the create
    path raises ``SessionControlError`` directly, while ``authorize_target`` must
    go through its ``deny`` closure to get the audit record and the 403.
    """
    if not _cron_caller(caller_key):
        return None
    job_id = caller_key[len(CRON_SLOT_PREFIX) :]
    found: Any = None
    try:
        for job in state.crons.list_jobs(include_disabled=True):
            if str(getattr(job, "id", "")) == job_id:
                found = job
                break
    except Exception:
        logger.warning(
            "session_control: cron owner lookup failed for %s -- refusing",
            caller_key,
            exc_info=True,
        )
        found = None
    if found is None:
        return (
            "the scheduled job behind this session could not be found, so its "
            "ownership cannot be verified",
            "cron_owner_unverifiable",
        )
    refusal = (
        "app-owned scheduled jobs cannot create or control sessions",
        "app_owned_cron_caller",
    )
    if str(getattr(found, "created_by", "") or "").startswith(APP_CRON_OWNER_PREFIX):
        return refusal
    owning_key = str(getattr(found, "session_key", "") or "")
    if owning_key:
        # Resolved through this module's own :func:`caller_slot_key` rather than a
        # ``removeprefix("dashboard:")``: chat_utils documents that the naive strip
        # is wrong for every non-dashboard session key, and the resolver already
        # matches on the identity each slot actually writes.
        owning_slot = state.get_slot(caller_slot_key(state, owning_key))
        if owning_slot is None:
            # The job names an owning session, but no live slot carries that key
            # any more -- the authoring tab was closed or evicted. Its ``_app``
            # tag lived only on that slot (``mcp_cron``'s ``cron_add`` records the
            # caller in ``session_key`` and never writes ``created_by``), so the
            # one place app-ness could be read is gone. That is precisely the
            # confinement escape: an app creates a cron through ``cron_add``,
            # closes its session, and its scheduled job then dispatches a
            # persistent, non-app, sidebar-visible session this gate can no longer
            # recognise as the app's.
            #
            # "Could not verify the owner is not an app" therefore fails CLOSED,
            # the same direction the missing-job and unreadable-registry cases
            # above take. This narrows the docstring's former "known residual":
            # the residual was an accepted fail-OPEN, and a fail-open on an
            # unresolvable owner is a security-gate defect (anchor
            # backend-security-controls). The cost is that a genuinely
            # user-created dispatching cron whose authoring tab has closed is
            # refused too -- but that caller can reopen a tab, whereas nothing can
            # undo an app session minted outside its confinement.
            return (
                "the session that authored this scheduled job is no longer open, "
                "so its ownership cannot be verified",
                "cron_owner_unverifiable",
            )
        if str(getattr(owning_slot, "_app", None) or ""):
            return refusal
    return None


# Roles a read must not count, taken from the persistence layer's own list rather
# than restated here: those are exactly the rows rehydration DROPS, so any cursor
# that counted them would name a different position after a restart than before
# it. ``chunk`` runs are deleted when a segment flushes and ``done`` markers never
# persist at all, so counting either inflates ``total``, the list shrinks back
# under it, and the next ``since=next_since`` read skips the finished reply for good.
TRANSIENT_ROLES = _PERSISTENCE_TRANSIENT_ROLES


class SessionControlError(Exception):
    """A refusal carrying the HTTP status AND the machine-readable reason.

    ``code`` is the contract the dashboard and the MCP tools match on; ``message``
    is advisory English prose (RFC 9457 3.1.3). Prose alone would be
    untranslatable by construction, since callers render it verbatim.
    """

    def __init__(
        self, message: str, status: int = 400, code: str = "session_control_error"
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code


def session_control_enabled() -> bool:
    """Whether the session-control surface is switched on in config.

    A config read that RAISES resolves to disabled, not to the field's default.
    ``load()`` can fail on a malformed section that has nothing to do with this
    feature, and treating that as "enabled" would let unrelated corruption
    silently undo an explicit ``session_control: false`` — the one switch
    standing between two of the user's sessions. Failing closed costs a
    refusal the user can diagnose from the log line; failing open costs the
    opt-out.
    """
    try:
        return bool(KiroCrewConfig.load().agent.session_control)
    except Exception:
        logger.warning(
            "session_control: config read failed — refusing until config loads", exc_info=True
        )
        return False


def manager_chain_enabled() -> bool:
    """Whether reach follows the whole manager line, and a departure hands sessions up.

    OFF by default, and off is not a degraded mode: with it off the fence answers the
    DIRECT manager and nothing else, and a departing session hands nothing up, which is
    exactly what the gateway answers without this feature at all. So the shipped default
    changes no behaviour, and an operator opts in to the line.

    A config read that RAISES resolves to disabled, for the reason
    :func:`session_control_enabled`'s does: unrelated corruption in another section must
    not quietly widen who can drive a session. Here that direction is also the field's
    own default, so the two agree and there is nothing for a malformed value to ride
    back to.
    """
    try:
        return bool(KiroCrewConfig.load().agent.session_manager_chain)
    except Exception:
        logger.warning(
            "session_manager_chain: config read failed -- keeping the line off",
            exc_info=True,
        )
        return False


def member_dispatch_enabled() -> bool:
    """Whether a crew member's DM session may bypass the ``session_control`` switch.

    The operator ceiling on the zero-configuration member grant. Default true
    reproduces today's behaviour exactly: a member caller bypasses
    ``agent.session_control`` and dispatches into workers it created. Set
    ``agent.member_dispatch`` to false and a member caller stops bypassing —
    it falls back under ``session_control_enabled()`` like any ordinary caller,
    so an operator who turned session control off keeps member DM threads
    chat-only without disabling the member itself.

    Fails CLOSED in BOTH ways the ceiling can lose the operator's value, the
    same direction :func:`session_control_enabled` does:

    * a config read that RAISES resolves to false; and
    * a config that LOADS but discarded the ``agent`` section (or the whole
      file) resolves to false too. ``load()`` does not raise on a malformed
      section -- it coerces it away, falls back to the field default (which is
      ``member_dispatch=True``, permissive), and records the loss in
      ``degraded_sections``. Without this second check a degraded ``agent``
      overlay carrying ``member_dispatch: false`` would silently revert to the
      bypass the operator meant to withdraw -- a governance-ceiling fail-open.
      This is the same "could not read it" vs "was never set" distinction
      :func:`tailnet_identity_unknown` and the publish gate already draw from
      ``degraded_sections``. The bypass never fails open.
    """
    try:
        cfg = KiroCrewConfig.load()
    except Exception:
        logger.warning(
            "member_dispatch: config read failed — withdrawing member bypass until config loads",
            exc_info=True,
        )
        return False
    if cfg.degraded_sections & {DEGRADED_WHOLE_CONFIG, "agent"}:
        # The agent section (or the whole file) was discarded, so a stored
        # `member_dispatch: false` was replaced by the permissive default.
        # Withdraw the bypass rather than trust that default.
        logger.warning("member_dispatch: agent config section degraded — withdrawing member bypass")
        return False
    return bool(cfg.agent.member_dispatch)


def member_admitted_to_scoped_surface(session_key: str, store: str) -> bool:
    """Whether a scoped caller is a crew member reaching a member-open surface.

    The ONE predicate the two surface gates that admit member callers share --
    the session-control HTTP gate (``handlers/session_control.py``'s
    ``_private_caller_refusal``) and the chat folder/tag gate
    (``handlers/_shared.py``'s ``private_chat_route_refusal``) -- so the two
    cannot drift on who a member is or when the surface is reachable for it.

    A caller is admitted when BOTH hold:

    * it is a crew member -- either an ``member-*`` DM slot key
      (:func:`is_member_session_key`) OR an ordinary chat slot bound to a crew
      member's private V2 store (:func:`_store_is_member_owned`, reading the
      VERIFIED scope the gate already resolved); AND
    * the surface is reachable for a member -- its own ``agent.member_dispatch``
      bypass OR the global ``agent.session_control`` switch it otherwise falls
      back under, since ``member_dispatch`` is a bypass ON TOP of the switch.

    This is the surface-level, caller-independent reachability only. It never
    decides which folders or sessions the admitted member may touch -- that
    ownership fence is per-resource and lives with each route
    (``member_owns_slot`` for filing/tagging, ``owner_app``/``folder_principal``
    for the tree). All reads run off the loop and fail closed on an unreadable
    config, so this gate can never open wider than the two switches behind it.
    """
    from kiro_crew.members import is_member_session_key

    is_member = is_member_session_key(session_key) or _store_is_member_owned(store)
    return is_member and (member_dispatch_enabled() or session_control_enabled())


def member_owns_slot(state: "DashboardState", slot: Any, caller_key: str) -> bool:
    """Whether *caller_key* (a member) may FILE or TAG *slot*.

    The member analogue of the app path's ``_app`` / ``app_owns_transcript``
    ownership check, and the SAME fence :func:`_caller_is_ownership_fenced`
    draws for session-control targets, so a member reaches through the chat
    folder/tag routes exactly the sessions it reaches through session-control
    and no others:

    * (a) its OWN session -- the slot whose slot key is the caller's own; and
    * (b) a session it CREATED -- ``_created_by == <caller's slot key>``, the
      value :func:`create_session` stamps on every child a member mints.

    ``caller_key`` is the VERIFIED ``X-Session-Key`` the gate authorized on --
    a SESSION key (``effective_session_key``, e.g. ``dashboard:chat-20-...``),
    never a body value. :func:`create_session` stamps ``_created_by`` with a
    SLOT key (``caller_slot_key(state, ...)``, e.g. ``chat-20-...``), so the two
    live in DIFFERENT key spaces: a raw ``_created_by == caller_key`` compare
    never matches a created child, because it compares a slot key to a session
    key. Resolve the caller to its slot key ONCE through :func:`caller_slot_key`
    (the same map :func:`create_session` writes the field through) and compare in
    that one space: ``_created_by`` and the slot's own ``key`` are both slot
    keys.

    The ``effective_session_key(slot) == caller_key`` arm stays as a
    session-space fallback: a channel-born slot whose key the live slot map
    cannot resolve (``caller_slot_key`` returns ``""``) is still owned when its
    session key IS the caller. Everything else is refused: the person's own
    sessions, an app's sessions, and another member's sessions all fail every
    arm.
    """
    if not caller_key or slot is None:
        return False
    resolved_slot_key = caller_slot_key(state, caller_key)
    if resolved_slot_key:
        created_by = getattr(slot, "_created_by", "")
        if created_by and created_by == resolved_slot_key:
            return True
        if getattr(slot, "key", "") == resolved_slot_key:
            return True
    return effective_session_key(slot) == caller_key


def _member_bypass(state: "DashboardState", caller_key: str) -> bool:
    """Whether *caller_key* may skip the ``session_control`` switch as a member.

    The single expression both switch gates key on, extracted rather than
    copy-pasted so the member-bypass condition cannot drift between
    :func:`create_session` and :func:`authorize_target`. A member caller
    bypasses only while the operator ceiling ``agent.member_dispatch`` is on;
    turned off, the member is no longer exempt and the switch gate applies to
    it like any other caller.

    Keyed on :func:`_member_caller` (a ``member-`` DM slot OR a chat slot bound
    to a member's V2 store — hence *state*) AND the config ceiling: the two
    together decide the bypass, and neither is a proxy for it.
    ``member_dispatch_enabled`` is read at the gate, synchronously, right before
    the act, exactly as ``session_control_enabled`` is beside it.
    """
    return _member_caller(state, caller_key) and member_dispatch_enabled()


async def prewarm_enabled_check() -> None:
    """Warm the config cache in a thread so the sync gate reads the cached path.

    :func:`session_control_enabled` cannot await -- ``authorize_target`` is
    synchronous, and ``read_messages`` is synchronous with it -- but its
    ``KiroCrewConfig.load()`` re-reads and validates the file on the FIRST call
    after a config edit, and doing that inline blocks the loop for every other
    session.

    Must be called with NOTHING that suspends between it and the gate. An
    ``await`` in that gap reopens exactly the hole this closes: a config edit
    landing in the window changes the fingerprint, so the gate's own read misses
    the cache and does the synchronous read anyway. That is why this is not done
    once at the top of each handler -- reading a request body suspends, and so
    does the SEL prewarm inside ``stop_target``.

    Lives here rather than in the handlers so the three call sites share one
    implementation, and so ``stop_target`` can warm it after its own prewarm
    without the handler layer reaching back into it.

    Best-effort: a failure is the gate's business, and the gate fails closed on
    its own.
    """
    try:
        await asyncio.to_thread(session_control_enabled)
    except Exception:  # pragma: no cover - the gate re-reads and decides
        logger.debug("session-control config prewarm failed; the gate will read it inline")


def caller_slot_key(state: "DashboardState", session_key: str) -> str:
    """Map a caller's session key to its slot key, or ``""`` when unknown.

    The MCP process authenticates as a session key (the history key), while
    every operation here is slot-keyed. Resolution walks the live slots and
    matches on the key each slot actually writes, which is the same identity
    ``list_sessions`` reports — so "who am I" cannot disagree between the two.

    An unresolvable caller is not fatal: it only means the self-target guard has
    nothing to compare against, which :func:`authorize_target` treats as a
    refusal rather than a pass.
    """
    if not session_key:
        return ""
    for slot in list(state._slots.values()):
        try:
            history_key = slot_history_key(slot)
            if session_key in (history_key, slot.key, transcript_stem(history_key)):
                return slot.key
        except Exception:
            continue
    return ""


# Joins the rooms of a :func:`_probe_channel_mirror` identity. Never part of a
# channel or thread id on any surface the probe composes from (Slack, Discord and
# Telegram ids are alphanumeric, a Slack thread is a decimal timestamp), so the
# split in :func:`mirror_audience` is the exact inverse of the join.
MIRROR_IDENTITY_SEPARATOR = "|"


def _probe_channel_mirror(state: "DashboardState", slot: "_ChatSlot") -> str | None:
    """The identity of *slot*'s outbound channel mirror, ``""`` when the
    conversation is not mirrored, or ``None`` when the session store could not
    answer.

    The tri-state exists because the two consumers need OPPOSITE fail-closed
    treatments of an unreadable store, and a collapsed boolean forces one of
    them to lie: the refusal paths must treat unknown as mirrored (refuse rather
    than open the boundary), while the queue-drain notice must not claim "the
    session gained a mirror" for a state change that is merely unverifiable.

    The identity (channel type + channel + thread) rather than a bare boolean,
    because a mirror can be RETARGETED while a queue waits: rebinding session
    mirror A to channel B keeps the boolean true from admission to drain while
    substituting the audience — exactly the republication change the drain
    re-check exists to catch.

    Read on the EFFECTIVE session key, because that is the key the mirror is
    registered under -- the slot key would miss a mirror on a session whose turns
    run under a different identity.

    Composed from BOTH store accessors the delivery legs read -- ``get_mirror_link``
    and ``get_slack_link`` -- because the first shadows the second: it returns the
    explicit ``mirror`` row whenever one exists and never looks at the Slack fields
    beside it, while the dashboard's slack-link binds its thread onto a channel-born
    slot's own session key (``DashboardState.link_slack``) without touching that
    row. Read through the mirror alone, a thread bound while an entry waited would
    leave this identity unchanged and the drain would deliver into a room the
    admission never saw. The thread is appended only when one is named: a
    threadless Slack row is bookkeeping, and the store itself never reads it as a
    mirror. Same two reads as :func:`owner_dm_refusal`, for the same reason.

    The identity is therefore a SET of rooms -- one ``type:channel:thread`` part
    per accessor that names one, joined by :data:`MIRROR_IDENTITY_SEPARATOR` --
    and the drain compares it as that set (:func:`mirror_audience`), not as one
    opaque string: a room the admission never saw is a retarget or a widening
    and drops, while a room that has since gone away is a narrowing and admits.
    Compared whole, a Slack thread UNLINKED while an entry waited would read as
    "the mirror changed" and drop a delivery whose audience only shrank.
    """
    if getattr(getattr(state, "sessions", None), "get_mirror_link", None) is None:
        # No store to ask: answer "not mirrored" WITHOUT touching the slot, the
        # same order the pre-split probe had. Callers that stamp containment
        # on a duck-typed slot (the spec-builder queue path) rely on it.
        return ""
    try:
        key = slot_history_key(slot)
    except Exception:
        logger.debug("mirror-link probe failed", exc_info=True)
        return None
    return _probe_channel_mirror_for_key(state, key)


def _probe_channel_mirror_for_key(state: "DashboardState", session_key: str) -> str | None:
    """:func:`_probe_channel_mirror` for a session that has no live slot.

    The mirror link lives in the session store keyed by the effective session
    key, so an ARCHIVED session can carry one just as a live slot can; the
    revive path reads it through this form because it has only the history key.
    Same tri-state, same fail-closed reading by the refusal paths, and the same
    two store reads (mirror row plus Slack binding) as the live form.
    """
    sessions = getattr(state, "sessions", None)
    if getattr(sessions, "get_mirror_link", None) is None:
        return ""
    binding = _read_mirror_binding(sessions, session_key)
    return None if binding is None else _mirror_identity_of(binding)


def _read_mirror_binding(sessions: Any, session_key: str) -> tuple[Any, str, str] | None:
    """ONE read of *session_key*'s outbound binding -- ``(mirror row, Slack thread,
    Slack channel)`` -- or ``None`` when the store cannot answer.

    The single store read a caller-side admission rests on. :func:`judge_owner_dm`
    judges its clauses on this tuple AND builds the audience it records from the
    same tuple (:func:`_mirror_identity_of`), so the row a gate validates and the
    row it records are one object: a retarget that lands after this read is a
    change the publisher sees against the record, never a room the record adopts
    unjudged. The probe (:func:`_probe_channel_mirror_for_key`) is this read
    reduced to its identity, so the drain and the publisher compare the same two
    accessors the delivery legs read: ``get_mirror_link`` and, through
    :func:`_slack_thread_of`, ``get_slack_link``.

    *sessions* must carry ``get_mirror_link``; the callers check that first, since
    a store without the accessor is "no mirror" (a duck-typed slot on the
    spec-builder queue path) rather than an unreadable one. A store whose accessor
    RAISES is unreadable and reads ``None`` -- the tri-state the two consumers need,
    see :func:`_probe_channel_mirror`.
    """
    try:
        link = sessions.get_mirror_link(session_key)
        slack_thread, slack_channel = _slack_thread_of(sessions, session_key)
    except Exception:
        logger.debug("mirror-link probe failed", exc_info=True)
        return None
    return link, slack_thread, slack_channel


def _mirror_identity_of(binding: tuple[Any, str, str]) -> str:
    """The probe identity of one :func:`_read_mirror_binding` -- its rooms, joined.

    One ``type:channel:thread`` part per accessor that names a room, joined by
    :data:`MIRROR_IDENTITY_SEPARATOR`; ``""`` when neither does. The Slack thread is
    appended only when one is named: a threadless Slack row is bookkeeping the store
    itself never reads as a mirror.
    """
    link, slack_thread, slack_channel = binding
    parts: list[str] = []
    if link:
        parts.append(
            f"{getattr(link, 'channel_type', '')}"
            f":{getattr(link, 'channel_id', '') or ''}"
            f":{getattr(link, 'thread_id', '') or ''}"
        )
    slack_identity = f"slack:{slack_channel}:{slack_thread}" if slack_thread else ""
    if slack_identity and slack_identity not in parts:
        parts.append(slack_identity)
    return MIRROR_IDENTITY_SEPARATOR.join(parts)


def mirror_audience(identity: Any) -> frozenset[str]:
    """The rooms a probe identity names, as the set the drain compares.

    Each member is one ``type:channel:thread`` part, compared WHOLE -- the colons
    inside a part are never split, so a surface whose ids contain colons still
    compares as one room. ``""`` (no mirror) and any non-string are the empty
    set. Order-insensitive by construction, which the probe's fixed part order
    does not need but a comparison must not depend on.
    """
    if not isinstance(identity, str) or not identity:
        return frozenset()
    return frozenset(part for part in identity.split(MIRROR_IDENTITY_SEPARATOR) if part)


def _slack_thread_of(sessions: Any, key: str) -> tuple[str, str]:
    """``(thread_ts, channel_id)`` of *key*'s Slack binding, ``("", "")`` when none.

    The probe's second read. Only the store's own answer shape counts -- a pair
    whose first member names a thread -- so a store without the accessor, or a
    stand-in that answers something else, reads as "no thread" rather than as an
    unverifiable probe: the mirror read alone was the whole probe before this read
    existed, and a second read must not turn every store that lacks it into a
    fail-closed drop. A store whose accessor RAISES still fails the probe, exactly
    as a raising mirror read does -- the caller's ``except`` is what decides that.
    """
    thread_getter = getattr(sessions, "get_slack_link", None)
    if not callable(thread_getter):
        return "", ""
    pair = thread_getter(key)
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        return "", ""
    thread, channel = pair
    if not isinstance(thread, str) or not thread:
        return "", ""
    return thread, str(channel or "")


def _has_channel_mirror(
    state: "DashboardState", slot: "_ChatSlot", *, on_probe_failure: bool = True
) -> bool:
    """Boolean view of :func:`_probe_channel_mirror` for the refusal paths.

    `linked_session_key` catches a channel-BORN slot. It does not catch a
    dashboard-born slot that was later given an OUTBOUND mirror link, which
    reaches a channel just as surely: the link lives in the session store, not
    on the slot, so a slot with an empty `linked_session_key` can still be
    republishing every turn to Slack or Telegram.

    Best-effort by design: a store that cannot answer returns *on_probe_failure*,
    and the default (``True``) keeps the refusal paths failing closed -- an
    unreadable link is treated as mirrored rather than opening the boundary.
    The enqueue-time containment snapshot passes ``False`` because ITS fail-closed
    direction is inverted: recording "not mirrored" for an unreadable link is the
    least-authorized admission state, so the drain-side re-check re-validates the
    entry instead of waving it through (see :func:`containment_snapshot`).
    """
    probed = _probe_channel_mirror(state, slot)
    return on_probe_failure if probed is None else bool(probed)


ORIGIN_NOT_ON_RECORD = (
    "this conversation's origin is not on record -- the channel dispatcher records "
    "it on each inbound message and it is not kept across a gateway restart, so "
    "send a message from the DM and retry"
)
"""The refusal an owner DM meets between a gateway restart and its next inbound turn.

The origin (``SessionManager.get_origin_link``) is held in memory only, while the
slot and its mirror are persisted and re-surfaced at boot -- so a monitor-loop cycle
or a dashboard-tab turn that runs before the owner's next channel message finds
every other clause satisfied and this one not. Naming it keeps a caller from
hunting for a link it cannot clear; the exemption itself stays withheld, because a
mirror without the recorded origin cannot be told from a retarget.
"""

MIRROR_PEER_NOT_ON_RECORD = (
    "the channel does not place the mirror's conversation as anyone's 1:1 direct "
    "message -- it is not a DM this channel opened or received a message from (a "
    "room, a thread, a group, a chat outside its roster), or that record was lost "
    "with a gateway restart; if it is the owner's DM, send a message from it (or "
    "link it again) and retry"
)
"""The mirrored shape's counterpart of :data:`ORIGIN_NOT_ON_RECORD`.

A mirror row persists a conversation id and no principal, and the transport's
answer to "whose DM is this" (``direct_peer_of``) is a peer or nothing -- so this
one refusal covers the two things "nothing" can mean, which the predicate cannot
tell apart at this seam and does not pretend to: a conversation that is not a DM
the transport would place at all (a guild room or thread, a Telegram group or
forum, a chat outside the roster the transport opens DMs for), and a Discord DM
whose in-process pairing a gateway restart dropped, until the DM is re-opened or
its peer writes into it. The remedy is therefore stated for the DM it may be and
for nothing else. A DM the transport DOES place with someone other than the
roster's sole owner never reads this; it reads :data:`NOT_THE_SOLE_OWNER`.
"""

# The two clause reasons both shapes of :func:`owner_dm_refusal` share, spelled once
# so the channel-born and the mirrored dashboard-born slot are refused in the same
# words for the same missing fact.
NOT_THE_SOLE_OWNER = "the channel's roster does not name this conversation's peer as its sole owner"
SLACK_THREAD_BESIDE = "the session also mirrors to a Slack thread"


def _sole_owner(state: "DashboardState", surface: str) -> tuple[Any, str, str]:
    """``(transport, owner, "")`` -- *surface*'s LIVE transport and the one
    ``user:<id>`` target it names as owner (``""`` when it names none) -- or
    ``(None, "", why)`` when the clause cannot be established at all.

    The roster clause both shapes of :func:`owner_dm_refusal` walk, in one place:
    the transport is the roster in force NOW (reloaded live, the very set that
    admits the peer's turns), read in memory so the predicate stays callable from
    ``close_target``'s no-suspension re-check; an absent transport means the
    channel is not running and refuses; an unreadable roster refuses. The caller
    compares *owner* against the peer it established for its own shape.
    """
    transport = state.get_channel_transport(surface)
    if transport is None:
        return None, "", f"the {surface} channel is not running"
    try:
        owner = sole_direct_target(transport.configured_targets())
    except Exception:
        logger.debug("owner-DM check: %s targets unreadable", surface, exc_info=True)
        return None, "", f"the {surface} roster is unreadable"
    return transport, owner, ""


@dataclass(frozen=True)
class OwnerDmVerdict:
    """What ONE reading of a caller's outbound binding decides, as one object.

    ``refusal`` is the predicate's answer -- ``""`` when the slot is the configured
    owner's own DM (:func:`owner_dm_refusal` is this field alone). ``mirrored`` is
    the gates' fail-closed reading of the same row -- an unreadable store counts as
    mirrored -- for the ``mirrored_caller`` refusal, so a gate never re-probes the
    store to decide it. ``admission`` is the containment the verdict rests on, in the
    queue-entry ``meta`` shape (:func:`containment_meta`), built from the very row
    the clauses judged: the object :func:`record_audience_admission` stores, so the
    audience a gate admits and the audience it records cannot be two different rows.
    """

    refusal: str
    mirrored: bool
    admission: dict[str, Any]


def owner_dm_refusal(state: "DashboardState", slot: "_ChatSlot") -> str:
    """Why *slot* is not the configured owner's own DM -- ``""`` when it is.

    The refusal field of :func:`judge_owner_dm`, for callers that need the reason
    alone (the ledger's :func:`session_owner_dm_refusal`, the re-checks after an
    await). A GATE that admits on the answer must call :func:`judge_owner_dm` itself
    and record the verdict's ``admission``: the reason and the audience come from one
    read there, and this view discards the audience.
    """
    return judge_owner_dm(state, slot).refusal


def judge_owner_dm(state: "DashboardState", slot: "_ChatSlot") -> OwnerDmVerdict:
    """Judge *slot* against the owner-DM exemption on ONE read of its binding.

    The ONE predicate the three channel-containment gates consult -- the creator
    gate and the target gate here, the ledger gate in ``handlers/work_ledger.py``
    (through :func:`session_owner_dm_refusal`) -- so they cannot drift: a gate
    keying on the live link, one on the key prefix and one on the mirror store
    would each admit and refuse different slots, and a key prefix can never be
    cleared while a link can. That containment exists because a channel session
    acts on words from a thread other people are in, and what it reads lands in
    front of them. For a 1:1 DM whose only human is the operator, the "audience"
    being protected is the operator themself, and refusing it makes every Discord
    and Telegram conversation a session that can dispatch nothing.

    The answer is an :class:`OwnerDmVerdict`, and every field of it comes from ONE
    read of the slot's binding (:func:`_read_mirror_binding`): the clauses below
    judge that row, the ``mirrored`` flag is that row's fail-closed reading, and
    the ``admission`` the gates record is built from that row and no other
    (:func:`_meta_from_probe`). A gate that judged one read and recorded a second
    would leave a window between them -- a retarget landing there (a link handler
    on another thread) would be recorded as the admitted audience without ever
    having been judged, and the publisher, comparing the live row against that
    record, would find nothing changed and publish a privately-read transcript
    into the new room. Reading once closes the window by construction: whatever
    lands after the read is a difference against the record.

    Two shapes reach that audience, and the predicate admits both on the same
    positive facts -- a verified surface, a live roster naming exactly one owner
    who is the conversation's peer, and no wider room bound beside it -- differing
    only in where the conversation and its peer are read from:

    * a CHANNEL-BORN slot (``linked_session_key`` is a channel key): the DM the
      session lives in, whose key names the peer; and
    * a DASHBOARD-BORN slot whose OUTBOUND mirror (``SessionMap.get_mirror_link``)
      is such a DM -- a conductor tab the operator linked to their own phone. Its
      words come from the dashboard and its readers are the mirror's, so the
      audience is identical; what differs is that the mirror row persists a
      conversation id and no principal, so the peer is read off the transport's
      own record of the DM (``direct_peer_of``).

    Every clause is a positive fact, and the first one that cannot be established
    is the answer, so a gate that refuses can say which fact was missing without a
    second walk that could disagree with the first. Every reason is generic -- a
    surface name at most, never an id -- because it is rendered into the refusal
    the caller reads in its own channel. For the channel-born shape, in order:

    * The key parses under the canonical grammar as a DIRECT conversation with
      exactly one peer, on a surface in :data:`OWNER_DM_CONDUCTOR_SURFACES`. A
      thread, group or forum key names a wider audience; a ``unified`` bucket
      names no peer; the legacy two-segment Slack shape does not parse; a surface
      not verified for this fails closed by construction.
    * The channel's LIVE transport names exactly one owner and it is that peer:
      :func:`~kiro_crew.messaging.transport.sole_direct_target` over
      ``configured_targets()``, the same one-identity rule ``/sessions`` and the
      proactive owner DM apply, and the same reasoning -- an allow-list is a list
      of people permitted to talk to the agent, not a claim that any of them is
      the operator, so several entries name nobody. Read off the transport rather
      than the config record because it is the roster in force NOW (reloaded live,
      the very set that admits the peer's turns) and an in-memory read, which
      keeps this callable from ``close_target``'s no-suspension re-check. An
      absent transport means the channel is not running, and a session nobody can
      drive is not admitted on the strength of a stale key.
    * The outbound mirror, if any, IS the conversation the session lives in. The
      dispatcher binds the DM as its own mirror on every turn, and that mirror is
      the same audience -- but the dashboard can retarget a mirror at any thread
      or channel, and a retargeted DM republishes what it reads to people who
      are not the owner. So the mirror must equal the ORIGIN conversation the
      dispatcher recorded (``SessionManager.get_origin_link``, written on every
      inbound turn beside the mirror bind); an unknown origin, an unreadable
      store or a mirror that names anywhere else refuses. Compared as a whole
      :class:`ChannelLink` because the DM channel id is not the peer id on every
      surface (Discord's is the id ``create_dm_channel`` returned), so no
      derivation from the key could stand in for the recorded truth. An
      UNLINKED DM reads ``None`` here and is admitted: the dispatcher's first
      turn stamps the conversation's namespaced bucket into the legacy
      ``slack_channel_id`` field and ``clear_mirror_link`` pops only the
      ``mirror`` row, so ``!unlink`` / ``/unlink`` and the dashboard's
      mirror-unlink leave a threadless Slack row behind -- and
      ``SessionMap.get_mirror_link`` filters that row at the source (an empty
      ``thread_ts`` never enters Slack's thread index, so it is bookkeeping that
      names no audience) rather than handing every reader a Slack link nobody
      chose. This clause therefore carries no copy of that rule, and neither does
      ``bind_origin_mirror``. The converse row is a second audience the mirror
      read CANNOT see: ``get_mirror_link``
      returns the explicit ``mirror`` row whenever one exists and never looks at
      the Slack fields beside it, while the dashboard's slack-link writes its
      thread onto the slot's effective key -- this session, for a channel-born
      slot (``DashboardState.link_slack``) -- and the turn path posts every
      dashboard-driven reply into that thread straight off ``get_slack_link``.
      So the thread is read through ``get_slack_link`` as well, and a non-empty
      ``thread_ts`` refuses: the DM's mirror still equals its origin, and the
      Slack thread is a room full of people who are not the owner.

    For the dashboard-born shape, in order:

    * The slot is an attended chat tab -- a person's own, or one an agent created
      -- and not an unattended one (``UNATTENDED_SLOT_PREFIXES``). A cron tab's
      ``cron:`` link is not a channel link, so it reaches this shape with a plain
      key, and a cron slot is otherwise an admitted (fenced) session-control
      source; but the audience argument below rests on the owner typing the words
      the mirror republishes, which no scheduled run's tab satisfies, so such a tab
      keeps today's mirror refusal even when its mirror is the owner's DM.
    * The slot HAS an outbound mirror. An unmirrored dashboard tab is not the
      owner's DM -- it is nobody's channel conversation -- and the gates never
      consult this predicate for it, since they ask only about a linked or
      mirrored caller.
    * The mirror is on a surface in :data:`OWNER_DM_CONDUCTOR_SURFACES` and names
      no thread. The Slack mirror the store synthesizes for a threaded Slack row
      fails the first test; a Telegram forum topic fails the second. A Discord
      thread carries its snowflake as the channel id with no thread id, so it is
      caught by the peer clause instead: a guild room is never on record as
      anyone's DM.
    * The channel's LIVE transport names exactly one owner -- the same roster
      clause, through the same call -- and the transport places the mirror's
      conversation as a DM with exactly that person (``direct_peer_of``). The
      record it answers from is the one the transport writes when it opens the DM
      (the dashboard link's ``resolve_configured_target``) or when an authorized
      message arrives from it; on Discord that record lives in this process, so
      a mirror linked before a gateway restart reads as unplaced until the DM is
      re-opened or its peer writes into it, and a mirror aimed at a room reads the
      same, since a room is never anyone's DM (:data:`MIRROR_PEER_NOT_ON_RECORD`
      names both causes and the remedy for the DM it may be; a DM the transport
      places with someone other than the sole owner reads
      :data:`NOT_THE_SOLE_OWNER` instead). Nothing is derived from the
      conversation id itself: whether it equals the peer is a per-platform fact
      only the transport knows.
    * No Slack thread is bound beside the mirror row, read through
      ``get_slack_link`` for the reason given above -- the same clause, on the
      slot's own effective key.

    There is no origin clause for this shape because there is no origin: the slot
    was born in the dashboard, and the dashboard is where its words come from. The
    retarget the channel-born origin clause catches is caught here by the peer
    clause -- a mirror moved to a room or to a stranger's DM stops naming the sole
    owner -- and a retarget while a session-control call is in flight is judged at
    the next call, since every gate consults the CURRENT row and never an earlier
    admission (the ledger re-checks after each read it awaits across); what such a
    call already read is covered by the audience the gate recorded from THIS read,
    which the publisher compares against the row live at delivery.

    What this deliberately does NOT establish is unfenced reach for the channel-born
    shape: an admitted DM is creator-fenced by :func:`_caller_is_ownership_fenced`,
    so a wrong inference costs the sessions the DM created and never the person's
    own tabs. Nor does it change the fence for the dashboard-born shape, in either
    direction: a person's own tab keeps the reach it had before the link and an
    agent-created tab stays fenced by its ``_created_by`` mark (see the fence's own
    notes on why the mirror is not a fifth population). Group and thread sessions,
    every other channel, every other mirror, and
    ``channel.CHANNEL_AGENT_BLOCKED_TOOLS`` are untouched.
    """
    sessions = getattr(state, "sessions", None)
    binding: tuple[Any, str, str] | None
    probed: str | None
    if getattr(sessions, "get_mirror_link", None) is None:
        # No store to ask: "no mirror" without touching the slot, exactly as
        # :func:`_probe_channel_mirror` answers (a duck-typed slot on the queue
        # path). The clauses still refuse the shape that needs the store.
        binding, probed = None, ""
    else:
        try:
            key = slot_history_key(slot)
        except Exception:
            logger.debug("mirror-link probe failed", exc_info=True)
            key = None
        binding = None if key is None else _read_mirror_binding(sessions, key)
        probed = None if binding is None else _mirror_identity_of(binding)
    return OwnerDmVerdict(
        refusal=_owner_dm_clauses(state, slot, sessions, binding),
        mirrored=True if probed is None else bool(probed),
        admission=_meta_from_probe(state, slot, probed),
    )


def _owner_dm_clauses(
    state: "DashboardState",
    slot: "_ChatSlot",
    sessions: Any,
    binding: tuple[Any, str, str] | None,
) -> str:
    """The clauses of :func:`judge_owner_dm`, judged on the read it took.

    *binding* is the slot's ``(mirror row, Slack thread, Slack channel)`` from
    :func:`_read_mirror_binding`, ``None`` when the store could not answer -- the
    ONLY reading of the mirror this function sees, so its verdict and the audience
    recorded beside it describe one row. The origin (channel-born shape) is read
    here: it is a validation input the record does not carry, not an audience.
    """
    link = _channel_link_of(slot)
    if not link:
        # ── The dashboard-born shape: judged on its outbound mirror row. ──
        if str(getattr(slot, "key", "")).startswith(UNATTENDED_SLOT_PREFIXES):
            # A cron tab's ``cron:`` link is stripped above, so it lands here with
            # a plain key -- and a cron slot IS admitted as a session-control
            # source (fenced to what it created). What this shape admits is the
            # PERSON's tab, whose words the owner types; a scheduled run's tab
            # mirrored to the owner's DM keeps the mirror refusal it meets today,
            # and a workflow result tab is refused as a source before this runs.
            return "the session is unattended, not a person's dashboard tab"
        if sessions is None:
            return "the session store is unavailable"
        if binding is None:
            return "the session store is unreadable"
        mirror, slack_thread, _slack_channel = binding
        if mirror is None:
            return "the session is neither channel-born nor mirrored"
        surface = str(getattr(mirror, "channel_type", "") or "")
        if surface not in OWNER_DM_CONDUCTOR_SURFACES:
            return "the outbound mirror is not on a verified owner-DM surface"
        if getattr(mirror, "thread_id", None):
            return "the outbound mirror names a thread, not a 1:1 direct message"
        transport, owner, why = _sole_owner(state, surface)
        if why:
            return why
        try:
            peer = str(transport.direct_peer_of(str(getattr(mirror, "channel_id", "") or "")) or "")
        except Exception:
            logger.debug("owner-DM check: %s peer lookup failed", surface, exc_info=True)
            return f"the {surface} channel cannot place the mirror"
        if not peer:
            return MIRROR_PEER_NOT_ON_RECORD
        if not owner or owner != f"{DM_TARGET_PREFIX}{peer}":
            return NOT_THE_SOLE_OWNER
        if slack_thread:
            return SLACK_THREAD_BESIDE
        return ""
    # ── The channel-born shape: judged on the conversation it lives in. ──
    parsed = parse_session_key(link)
    if parsed is None:
        return "the session key does not name a channel conversation"
    if parsed.surface not in OWNER_DM_CONDUCTOR_SURFACES:
        return f"{parsed.surface} is not a verified owner-DM surface"
    if parsed.chat_type != CHAT_TYPE_DIRECT or len(parsed.scope) != 1:
        return "the conversation is not a 1:1 direct message"
    _transport, owner, why = _sole_owner(state, parsed.surface)
    if why:
        return why
    if not owner or owner != f"{DM_TARGET_PREFIX}{parsed.scope[0]}":
        return NOT_THE_SOLE_OWNER
    if sessions is None:
        return "the session store is unavailable"
    if binding is None:
        return "the session store is unreadable"
    try:
        origin = sessions.get_origin_link(link)
    except Exception:
        logger.debug("owner-DM check: session store unreadable for %s", link, exc_info=True)
        return "the session store is unreadable"
    mirror, slack_thread, _slack_channel = binding
    if not isinstance(origin, ChannelLink):
        return ORIGIN_NOT_ON_RECORD
    if slack_thread:
        return SLACK_THREAD_BESIDE
    if mirror is None or (isinstance(mirror, ChannelLink) and mirror == origin):
        return ""
    return "the outbound mirror points somewhere other than this conversation"


def session_owner_dm_refusal(state: "DashboardState", session_key: str) -> str:
    """:func:`owner_dm_refusal` for a caller known only by its session key.

    The ledger gate holds an ``X-Session-Key`` and no slot, so it resolves the
    slot the way every session-control verb does -- :func:`caller_slot_key`, the
    identity ``list_sessions`` reports -- and judges THAT slot. Resolving through
    the same function is what makes "the ledger gate and session control agree on
    the same slot" true by construction rather than by two lookups happening to
    coincide; a key that resolves to no open slot is refused, as
    :func:`authorize_target` refuses an unidentifiable caller.
    """
    slot_key = caller_slot_key(state, session_key)
    slot = state.get_slot(slot_key) if slot_key else None
    if slot is None:
        return "the session key resolves to no open slot"
    return owner_dm_refusal(state, slot)


# ── Drain-time re-validation of queued prompts ──
#
# Authorization is decided when a prompt is ADMITTED — `authorize_target` for
# `session_send`, the authenticated composer for a human — but a busy target
# QUEUES the prompt and delivers it later, and the containment those decisions
# rest on can change in between: a target authorized while unlinked can gain a
# channel or mirror link before its queue drains, and the queued prompt would
# then execute and republish to an audience its admission never contemplated.
# Producers stamp the constraints that held at admission on the queue entry
# (`containment_meta`); `chat_runner`'s drain recomputes them and drops any
# entry for which a constraint holds at delivery that did not hold at admission.

# Queue-entry meta key carrying the admission-time containment snapshot.
QUEUED_CONTAINMENT_META_KEY = "queued_containment"

# Queue-entry meta key naming the slot that SENT a cross-session delivery, so a
# drain-time drop can be reported back to it. It rides ``meta`` rather than a
# consumption callback because ``meta`` is one of the keys a queued prompt is
# persisted with, while an entry carrying a callback is excluded from that write
# (``slot_queue_repository._is_durable_queue_entry``): recording the sender as a
# callback would trade a relay's survival across a restart for a notice that
# cannot survive one either. A requeued steer keeps it for free — the requeue
# copies the admission dict onto the new entry's meta.
SEND_ORIGIN_META_KEY = "send_origin_slot"

# Queue-entry meta key recording the AUTHORIZATION facts of a cross-session admission:
# which session sent it and whether the ownership fence applied. Kept apart from
# ``SEND_ORIGIN_META_KEY`` on purpose, because the two are stripped for different
# reasons and must not be stripped together. The routing stamp names a WRITE TARGET --
# the drop notice is appended there -- so the restore path drops it rather than trust
# an editable line, and it is omitted when the sender has no tab identity to check.
# Neither reason applies to provenance, which is only ever compared and never written
# through. Losing it with the routing stamp turned a restart (or a tab-less sender) into
# a way past the drain's sender re-check, and the revoked sender's message then ran.
ADMISSION_PROVENANCE_META_KEY = "admission_provenance"

# Queue-entry meta key naming the CHANNEL CONVERSATION that sent a message into a
# resumed dashboard session mid-turn (``dashboard.channel_handoff``), so a
# drain-time drop can be reported back into that conversation: the channel was
# told "queued" at admission and, unlike a dashboard sender, reads neither the
# target's transcript nor the SEL. Same reasoning as the sender stamp above for
# why it is ``meta`` and how a requeued steer keeps it. It names a WRITE TARGET
# on a network surface, so the restore path strips it like the sender stamp, and
# the notice re-runs the outbound recipient authorization before it is sent.
CHANNEL_RECIPIENT_META_KEY = "channel_recipient"

# How much of a dropped delivery's own text the sender's notice quotes back, so
# a caller holding several deliveries in flight can tell which one went.
SEND_DROP_EXCERPT_CHARS = 120

# The two deliberate discards of a queued entry that are not the drain's
# authorization re-check: a person cancelling the entry's queue card, and a hard
# stop clearing the whole queue. Each is passed as the ``cause`` of
# :func:`notify_send_origin_discarded`, which words the sender's notice from it.
SEND_DISCARD_QUEUE_CANCEL = "queue_cancel"
SEND_DISCARD_HARD_STOP = "hard_stop"

# Sender-notice phrasing per discard cause. Neither discard changes a
# constraint, so the drain's containment wording would describe nothing here.
_SEND_DISCARD_REASONS = {
    SEND_DISCARD_QUEUE_CANCEL: "it was cancelled from that session's queue",
    SEND_DISCARD_HARD_STOP: "a hard stop of that session discarded its queue",
}

# Transcript-notice phrasing per snapshot field, for the drop notice a reader
# of the session must be able to understand without knowing this module.
_CONTAINMENT_CHANGE_LABELS = {
    "linked": "the session was linked to a channel",
    "mirrored": "the session gained an outbound channel mirror",
    "mirror_retarget": "the session's outbound mirror was retargeted to a different channel",
    "ephemeral": "the session became incognito/temporary",
    "app": "the session became app-scoped",
    "unattended": "the session became unattended",
    "workspace": "the session moved to a different workspace",
    "manager_unknown": "which session manages this one could no longer be read",
    "sender_no_longer_manages": (
        "the session that sent it no longer manages this one, so it may no longer drive it"
    ),
}


# Snapshot keys that are NOT constraints: carried for notice wording and
# telemetry only, never compared by :func:`newly_held_constraints`.
_NON_CONSTRAINT_KEYS = frozenset({"mirror_unverified"})

# The one constraint a directive user-origin entry is exempt from at the drain:
# a channel LINK on the entry's own session. The author of a directive entry is
# an authenticated human typing into that session's own surface, and linking it
# is that surface owner's deliberate act — dropping their already-typed messages
# when they link would destroy user speech on a supported flow (`api_chat`
# applies no linked refusal to composer input). `mirrored` is deliberately NOT
# exempt: directive content can be authored by any allowed human in a linked
# thread while only the session owner adds outbound mirror links, so a NEW
# mirror widens the audience beyond anything the message's author controlled —
# the exact republication this drain re-check catches. `session_send` and
# automation entries never carry the flag and stay fully enforced.
_AUDIENCE_CONSTRAINTS = frozenset({"linked"})


def containment_snapshot(
    state: "DashboardState", slot: "_ChatSlot", *, on_probe_failure: bool
) -> dict[str, Any]:
    """The target-side containment constraints of :func:`authorize_target`, as
    they hold for *slot* right now.

    Two call sites with OPPOSITE fail-closed directions, hence the mandatory
    ``on_probe_failure``: the enqueue-time snapshot passes ``False`` so an
    unreadable mirror link records the least-authorized admission state (the
    drain then re-validates the entry), while the drain-time snapshot passes
    ``True`` so an unreadable link refuses delivery rather than opening the
    boundary. When the drain-side probe fails, ``mirror_unverified`` is set so
    the drop notice can say the state could not be verified instead of claiming
    a mirror appeared — the refusal is the same, the wording must not lie.
    Every other field is a plain slot attribute read that cannot fail.

    ``workspace`` is the seventh refusal (:func:`authorize_target`'s
    ``workspace_mismatch``), an identity rather than a boolean: a CHANGE — the
    slot moving to another workspace while the entry waited — invalidates the
    admission, because the prompt would run with memory, lessons and project
    context its admission never saw. It is compared only when the entry
    recorded one; the unmarked fail-closed baseline stays the boolean set,
    since there is no least-authorized workspace to assume.

    ``unattended`` keys on the slot-key prefix exactly as ``authorize_target``
    does. A slot key is immutable, so this field can never flip between enqueue
    and drain for a TAGGED entry — it is carried for the unmarked fail-closed
    path, where the baseline is all-False and any held constraint must count.
    """
    return _snapshot_from_probe(
        state, slot, _probe_channel_mirror(state, slot), on_probe_failure=on_probe_failure
    )


def _snapshot_from_probe(
    state: "DashboardState", slot: "_ChatSlot", probed: str | None, *, on_probe_failure: bool
) -> dict[str, Any]:
    """:func:`containment_snapshot` over a probe already taken.

    *probed* is :func:`_probe_channel_mirror`'s tri-state for *slot*. Split out so
    :func:`judge_owner_dm` can build the audience it records from the SAME read it
    judged rather than probing the store a second time -- the second read is where
    a retarget could slip in between validation and record. Every other field is a
    plain slot attribute read that cannot fail.
    """
    snap: dict[str, Any] = {
        "linked": bool(getattr(slot, "linked_session_key", "")),
        "mirrored": on_probe_failure if probed is None else bool(probed),
        "ephemeral": getattr(slot, "memory_mode", "persistent") != "persistent",
        "app": bool(getattr(slot, "_app", "")),
        "unattended": str(getattr(slot, "key", "")).startswith(UNATTENDED_SLOT_PREFIXES),
        "workspace": str(getattr(slot, "workspace", "default") or "default"),
        # Who manages this slot is not known -- the two refusals ``require_manager_of``
        # raises, as the one boolean the drain can compare. Both go here rather than
        # into the non-containment set because the premise that let the caller-side
        # verdict go un-rechecked does not hold for them: ``not_creator`` is exempt
        # because ``created_by`` is written at birth and never mutates, while this is a
        # value a handoff MOVES and a failed read can retract. False at admission by
        # construction (an entry admitted for a widened verb got past the read), so a
        # True here is always a change, and the fail-closed baseline for an unmarked
        # entry gives the right answer too.
        "manager_unknown": _manager_unread_refusal(state, slot) is not None,
    }
    if probed is not None:
        # The mirror's identity, compared by its rooms (:func:`mirror_audience`):
        # a RETARGETED mirror (A -> B) keeps the boolean true across the wait
        # while substituting the audience, so identity is what the drain must
        # compare -- and it must compare rooms, not the string, because a room
        # dropped while the entry waited is a narrowing, not a change of audience.
        # Omitted on probe failure — there is no identity to compare then, and
        # the drain fails closed on the unverifiable boolean instead
        # (:func:`newly_held_constraints` treats ``mirror_unverified`` as a
        # mirror change regardless of the admission snapshot).
        snap["mirror_identity"] = probed
    if probed is None and on_probe_failure:
        snap["mirror_unverified"] = True
    return snap


def containment_meta(state: "DashboardState", slot: "_ChatSlot") -> dict[str, Any]:
    """Queue-entry ``meta`` recording the containment that held at admission.

    Every producer of a plain (user-speech) queue entry stamps this at enqueue;
    the drain compares it against the constraints holding at delivery and drops
    the entry when one is newly held (:func:`newly_held_constraints`). An entry
    without the stamp fails closed — it is checked against the full
    current-constraint set — so an untagged producer can never ride a queued
    prompt past a boundary the tagged paths respect.
    """
    return _meta_from_probe(state, slot, _probe_channel_mirror(state, slot))


def _meta_from_probe(
    state: "DashboardState", slot: "_ChatSlot", probed: str | None
) -> dict[str, Any]:
    """:func:`containment_meta` over a probe already taken -- the record shape an
    :class:`OwnerDmVerdict` carries, built from the read the verdict judged."""
    return {
        QUEUED_CONTAINMENT_META_KEY: _snapshot_from_probe(
            state, slot, probed, on_probe_failure=False
        )
    }


def send_origin_meta(
    state: "DashboardState",
    sender_slot_key: str,
    *,
    manager_fenced: bool,
    chain: "bool | None" = None,
) -> dict[str, Any]:
    """Queue-entry ``meta`` naming the slot a cross-session delivery came FROM.

    ``manager_fenced`` records WHETHER THE FENCE APPLIED to this admission, and it is
    what decides whether the drain may re-check the manager at all. An unfenced
    sender's admission never read the manager -- ``authorize_target`` short-circuits on
    ``ownership_fenced`` before the read -- so re-comparing it at delivery would drop a
    message against a predicate that was never applied to it, deleting text the caller
    was authorized to send and still is. Re-assert at the drain exactly the predicate
    that admitted the entry, including whether it applied.

    Stamped by the delivery paths that admit one session's text onto another
    session's queue, so a drain-time drop can be reported back to the sender
    (:func:`notify_send_origin_dropped`). The sender is told at admission that
    the message was queued rather than started; the drop itself is visible only
    on the target's transcript and in the audit trail, neither of which the
    sender reads.

    The stamp carries the sender's TAB IDENTITY beside its key, and both must be
    present or nothing is stamped. A slot key does not identify a session: the
    explicitly-named keys are deterministic (``cron-{job.id}``,
    ``workflow-{run_id}``, a channel's own), so a closed slot's key is handed to
    the next occupant, whose ``app``, ``origin`` and link scope are declared per
    creation and need not match the sender's. Resolving the notice from the key
    alone therefore appends one session's text to a DIFFERENT session's
    transcript once the sender closes and the key is reused. ``_tab_id`` is
    minted per slot object and is the identity the neighbouring save and close
    paths already compare on (``chat_persistence._slot_still_ours``).

    Empty for a caller with no slot of its own, and the key is then omitted
    rather than stamped blank: absent must mean "nobody to report to", which a
    blank string cannot be told apart from. A caller whose slot carries no tab
    identity is omitted the same way, because a stamp whose identity cannot be
    checked later is the one shape that must not produce a write.
    """
    key = str(sender_slot_key or "")
    if not key:
        return {}
    # PROVENANCE first and unconditionally: it is what the drain re-checks, and it
    # needs no tab because nothing is ever written through it.
    out: dict[str, Any] = {
        ADMISSION_PROVENANCE_META_KEY: {
            "sender": key,
            "fenced": bool(manager_fenced),
            # The sender's own IDENTITY, beside its key, because this record outlives
            # the session that made it: a slot key is a name, and the drain comparing
            # names alone hands a replacement under that name the grant the sender was
            # given and then lost. Carried in the PROVENANCE and not in the routing
            # stamp, because the restore strips routing and the re-check has to survive
            # it -- the whole point of the two being separate keys.
            "tab": str(getattr(state.get_slot(key), "_tab_id", "") or ""),
            # THE POLICY THIS ADMISSION USED, recorded here so the drain needs no
            # config read of its own. Read at the drain, that switch cost either a
            # blocking parse on the event loop or an ``await`` -- and the await was
            # worse: it put a suspension in front of the dequeue, where a concurrent
            # direct send could start a turn the drain then overwrote.
            #
            # Recording it is also the more correct answer. A re-check re-asserts the
            # predicate that ADMITTED the entry, and an operator flipping the switch
            # while a message waits does not change what admitted it.
            "chain": bool(manager_chain_enabled() if chain is None else chain),
        }
    }
    sender = state.get_slot(key)
    tab = str(getattr(sender, "_tab_id", "") or "")
    if tab:
        # ROUTING only when the notice can be addressed to the exact slot object that
        # sent it. Its absence means "nobody to report to", nothing more.
        out[SEND_ORIGIN_META_KEY] = {"slot": key, "tab": tab}
    return out


def forget_admission_provenance(entry_meta: Any) -> None:
    """Drop the admission stamp from *entry_meta*: its TEXT was replaced.

    PROVENANCE DESCRIBES WHO AUTHORISED THIS TEXT. It is stamped when one session's
    words are admitted onto another session's queue, and everything the drain does with
    it -- re-comparing the sender against the manager, counting the manager as unknown,
    reporting a drop back -- is about that text and that admission. Replace the text and
    none of it still applies: the sender did not write what is in the entry now, so its
    authorization is not the authorization under which the entry would run.

    The case that forced this is the worst one. A lead queues a prompt on a busy worker,
    the person at the dashboard edits the queued row, and the entry still carries the
    lead's stamp -- so handing the worker to another session makes the drain delete THE
    PERSON'S OWN WORDS as a lapsed cross-session grant. Fail-closed governs granting
    reach and never governs keeping data; here it was deleting the one copy of something
    a human typed, for a reach nobody was exercising.

    Unconditional on a replacement, not special-cased to the authenticated editor. An
    edit from a third session has the same shape -- the stamp would name neither the
    author of the text nor the editor -- and a rule that holds for every replacement
    cannot be forgotten by the next path that adds one. Nothing is granted by dropping
    it: the entry was already admitted, and both readers treat an unstamped entry as one
    whose manager was never read, which is the keep-the-data direction.

    The ROUTING stamp goes with it, for the same reason and not as a tidy-up: it exists
    to address a drop notice to the session that sent the text, and after a replacement
    there is no such session to tell.
    """
    if not isinstance(entry_meta, dict):
        return
    entry_meta.pop(ADMISSION_PROVENANCE_META_KEY, None)
    entry_meta.pop(SEND_ORIGIN_META_KEY, None)


def send_origin_manager_fenced(entry_meta: Any) -> bool:
    """Whether the fence applied to the admission that stamped *entry_meta*.

    False for an unstamped entry, for a stamp from before this field existed, and for a
    malformed one -- all three mean "no evidence the manager was ever read for this
    entry", and the drain must not re-check a predicate it cannot show was applied.
    Fail-closed here is NOT dropping: dropping is the destructive outcome, because the
    entry is a message a person or an authorized session already had accepted.
    """
    record = _admission_provenance(entry_meta)
    return record.get("fenced") is True


def send_origin_sender_tab(entry_meta: Any) -> str:
    """The sender's tab IDENTITY from *entry_meta*'s admission stamp, or ``""``.

    Empty for a stamp written without one, which reads as an unprovable identity and
    never as a match -- see :func:`manager_record_names`. Kept beside
    :func:`send_origin_sender` because the two are one fact: which session sent this,
    named and identified.
    """
    record = _admission_provenance(entry_meta)
    tab = record.get("tab")
    return tab if isinstance(tab, str) else ""


def send_origin_chain_applied(entry_meta: Any) -> bool:
    """Whether the manager LINE was in force for the admission that stamped *entry_meta*.

    False for an unstamped entry, for a stamp from before this field existed, and for a
    malformed one. That default is the base PR's own answer -- the one-level predicate --
    which is what admitted every entry stamped before the line existed, and it never
    drops more than that predicate drops.
    """
    record = _admission_provenance(entry_meta)
    return record.get("chain") is True


def send_origin_sender(entry_meta: Any) -> str:
    """The session whose admission put *entry_meta*'s entry on the queue, or ``""``.

    Read off PROVENANCE, not the routing stamp, so it survives the restore path and a
    sender with no tab. :func:`send_origin_slot` is the routing read and answers only
    where a notice may be addressed. They are separate fields because the restore path
    strips the routing one, and the sender the drain re-checks must not go with it.
    """
    sender = _admission_provenance(entry_meta).get("sender")
    return sender if isinstance(sender, str) else ""


def _admission_provenance(entry_meta: Any) -> "dict[str, Any]":
    """The entry's provenance record, or ``{}`` for an absent or malformed one."""
    record = entry_meta.get(ADMISSION_PROVENANCE_META_KEY) if isinstance(entry_meta, dict) else None
    return record if isinstance(record, dict) else {}


def send_origin_slot(entry_meta: Any) -> str:
    """The sending slot key stamped on a queue entry, or ``""``.

    Pairs with :func:`send_origin_tab`: the key says where to write and the tab
    says which occupant of that key is owed the notice, so a caller that resolves
    a recipient needs BOTH to agree with the live slot.

    *entry_meta* is plumbing of any shape, so a missing, non-dict or malformed
    value reads as no sender and the drop proceeds exactly as it did before the
    stamp existed.

    A stamp this reads is one THIS process admitted. The restore path drops the
    key (:func:`~kiro_crew.dashboard.slot_queue_repository.sanitize_restored_queue`)
    because the value names a write target rather than being merely read: the
    drop resolves the recipient of its notice from this stamp and appends the
    entry's own text there, so a stamp carried back off an editable line would
    put attacker-chosen text in a session the editor does not own. The price is
    one notice: a delivery that outlives a restart and is then dropped reports to
    nobody, while the delivery itself still survives.
    """
    return _send_origin_field(entry_meta, "slot")


def send_origin_tab(entry_meta: Any) -> str:
    """The sending slot's tab identity stamped on a queue entry, or ``""``.

    The notice is owed to the slot OBJECT that sent the message, not to whatever
    currently answers to its key, so this is what tells a reused key apart from
    the original sender. See :func:`send_origin_meta` for why a key alone is not
    an identity.
    """
    return _send_origin_field(entry_meta, "tab")


def _send_origin_field(entry_meta: Any, field: str) -> str:
    """One string field of the sender stamp, or ``""`` for any other shape.

    Both readers fail closed through here on the same shapes, so a half-written
    or hand-edited stamp cannot answer one question and not the other -- which is
    what would let a key be trusted while its identity check silently passed.
    """
    if not isinstance(entry_meta, dict):
        return ""
    stamp = entry_meta.get(SEND_ORIGIN_META_KEY)
    if not isinstance(stamp, dict):
        return ""
    value = stamp.get(field)
    return value if isinstance(value, str) else ""


def send_drop_excerpt(text: Any) -> str:
    """The dropped message's own opening, for the notice the sender reads.

    A caller can have several deliveries in flight to several targets, and the
    target's key alone does not say WHICH message went. Whitespace is collapsed
    so a multi-line prompt stays one line in the notice, and the cut is marked
    with an ellipsis so a truncated quote is never mistaken for the whole text.
    """
    flat = " ".join(str(text or "").split())
    if len(flat) <= SEND_DROP_EXCERPT_CHARS:
        return flat
    return flat[:SEND_DROP_EXCERPT_CHARS].rstrip() + "…"


def channel_recipient_meta(
    channel_type: str, conversation_id: str, principal: str
) -> dict[str, Any]:
    """Queue-entry ``meta`` naming the channel conversation a message came FROM.

    Stamped by :func:`~kiro_crew.dashboard.channel_handoff.hand_to_resumed_slot`
    on both of its arms (the queue entry directly; the steer through its admission
    dict, which the requeue copies onto the entry), so a drain-time drop can be
    reported into that conversation (:func:`notify_channel_recipient_dropped`).

    *principal* is the platform user id the channel authorized on inbound. It
    rides along because the outbound recipient check needs one the SESSION KEY
    cannot supply: a dashboard session names no channel peer, and a Discord DM's
    conversation id is unrelated to the user id its roster holds, so without it the
    notice would be refused as an unidentifiable recipient. Empty when the
    channel has none to give (a thread route answers on its conversation id).

    Empty when either address field is missing, and then nothing is stamped
    rather than a half-address: a stamp that cannot be delivered to must not
    produce a write.
    """
    channel_type = str(channel_type or "")
    conversation_id = str(conversation_id or "")
    if not channel_type or not conversation_id:
        return {}
    return {
        CHANNEL_RECIPIENT_META_KEY: {
            "channel_type": channel_type,
            "conversation_id": conversation_id,
            "principal": str(principal or ""),
        }
    }


def channel_recipient_of(entry_meta: Any) -> tuple[str, str, str] | None:
    """``(channel_type, conversation_id, principal)`` from an entry's stamp, or None.

    *entry_meta* is plumbing of any shape: a missing, non-dict or malformed stamp
    -- a non-string field, an empty address -- reads as no recipient, and the drop
    proceeds unreported exactly as it does for a human-typed entry. Both readers
    of a write-target stamp fail closed on the same shapes.
    """
    if not isinstance(entry_meta, dict):
        return None
    stamp = entry_meta.get(CHANNEL_RECIPIENT_META_KEY)
    if not isinstance(stamp, dict):
        return None
    channel_type = stamp.get("channel_type")
    conversation_id = stamp.get("conversation_id")
    principal = stamp.get("principal", "")
    if not isinstance(channel_type, str) or not isinstance(conversation_id, str):
        return None
    if not channel_type or not conversation_id or not isinstance(principal, str):
        return None
    return channel_type, conversation_id, principal


def newly_held_constraints(
    now: dict[str, Any],
    entry_meta: Any,
    *,
    directive_user_origin: bool = False,
    manager_fence_applied: bool = False,
) -> list[str]:
    """Containment constraints in *now* that the entry's admission never saw.

    *now* is the drain-time :func:`containment_snapshot`; *entry_meta* is the
    queue entry's ``meta`` (any shape — untrusted plumbing, so a missing or
    malformed snapshot degrades to the all-False baseline and the entry is
    checked against every currently-held boolean constraint, failing closed).

    A constraint recorded ``True`` at admission is not a change: the prompt was
    knowingly admitted under it (a human typing into a channel-born session, an
    app relaying into its own slot), and dropping it would refuse designed
    behaviour rather than close a window.

    ``workspace`` compares by identity and only when the entry recorded one —
    an unmarked entry has no least-authorized workspace to assume, so its
    fail-closed floor stays the boolean set. ``mirror_identity`` is compared
    only when the entry recorded one too, but as a SET of rooms
    (:func:`mirror_audience`) rather than one identity: a room the admission
    never saw — a mirror retargeted to a different channel, or a Slack thread
    bound beside the admitted mirror while the entry waited — is an audience
    change the boolean cannot see, reported as ``mirror_retarget``; a room that
    has since gone away (a thread unlinked) is a narrowing and is not a change.

    *directive_user_origin* exempts the LINKED constraint only, for entries
    carrying the authenticated-human provenance flag: the author typed into the
    session's own surface and linking it is that owner's deliberate act, so
    dropping their already-typed messages when they link the session would
    destroy user speech on a supported flow (``api_chat`` applies no linked
    refusal to composer input). A NEW outbound mirror is never exempt — the
    message's author does not control mirror links, so it still drops. Every
    other constraint — ephemeral, app, unattended, workspace — applies
    to directive entries too.
    """
    recorded: dict[str, Any] = {}
    if isinstance(entry_meta, dict):
        raw = entry_meta.get(QUEUED_CONTAINMENT_META_KEY)
        if isinstance(raw, dict):
            recorded = raw
    changed: list[str] = []
    for name, value in now.items():
        if name in _NON_CONSTRAINT_KEYS:
            continue
        if name == "workspace":
            admitted_ws = recorded.get("workspace")
            if isinstance(admitted_ws, str) and admitted_ws != value:
                changed.append(name)
            continue
        if name == "manager_unknown":
            if not manager_fence_applied:
                # Same gate the sender re-compare uses, for the same reason: an
                # admission the fence never touched did not read the manager, so an
                # unreadable manager at delivery is not a constraint it gained. Without
                # this an unfenced caller's accepted message is deleted whenever a
                # marked slot's log has not been read yet, and a person's typed prompt
                # with it.
                continue
            # Compared ONLY when the entry recorded it, like ``workspace`` and for the
            # same reason: there is no least-authorized value to assume for an unmarked
            # entry, so the boolean set is the wrong fail-closed floor here.
            #
            # Counting it against the all-False baseline DELETES USER SPEECH. The restore
            # path strips this stamp deliberately and a marked slot hydrates
            # ``_manager_unresolved`` -- so against that floor the first drain after a
            # restart drops every restored prompt a person typed into a handed-over
            # session, telling them the authorization that admitted it has lapsed. Their admission was the authenticated composer and
            # never rested on who manages the slot, so an unread manager is not a
            # constraint they gained.
            #
            # A STAMPED entry is still dropped, which is the boundary this closes: its
            # admission did read the manager, the drain cannot re-read it, and every
            # cross-session producer stamps at enqueue. One admitted while the value was
            # already unknown is not a change, matching every other constraint here.
            if value and recorded.get(name) is False:
                changed.append(name)
            continue
        if name == "mirrored":
            # Fail closed on an unverifiable drain-side probe REGARDLESS of the
            # admission snapshot: an entry admitted under mirror A cannot be
            # delivered when the store no longer answers, because the audience
            # may have been retargeted since admission and there is no identity
            # to compare (the probe-failure snapshot omits ``mirror_identity``).
            # Matching ``authorize_target``'s posture — unreadable state refuses
            # rather than opens the boundary; the notice wording says the state
            # could not be verified (``mirror_unverified``), never that a mirror
            # appeared.
            if value and (now.get("mirror_unverified") or not bool(recorded.get(name, False))):
                changed.append(name)
            continue
        if name == "mirror_identity":
            # Room-set comparison: a mirror RETARGETED while the entry waited
            # (A -> B) keeps ``mirrored`` true at both ends while substituting
            # the audience, so the boolean can never see it, and a room ADDED
            # beside the admitted one (a Slack thread bound onto the session)
            # widens the audience the same way. Both hold a room the admission
            # never saw, which is the test. A room that has since gone away -- a
            # thread unlinked, a mirror cleared -- is a NARROWING: every room the
            # delivery can now reach was admitted, so it is not a change. Fires
            # only when both sides carry a verified, non-empty identity — a newly
            # GAINED mirror is the boolean's job, and an unverifiable side omits
            # the key. Never exempt for directive entries: the message's author
            # does not control mirror links.
            admitted_id = recorded.get("mirror_identity")
            if value and isinstance(admitted_id, str) and admitted_id:
                if mirror_audience(value) - mirror_audience(admitted_id):
                    changed.append("mirror_retarget")
            continue
        if directive_user_origin and name in _AUDIENCE_CONSTRAINTS:
            continue
        if value and not bool(recorded.get(name, False)):
            changed.append(name)
    return changed


#: The drop reason for a queued cross-session delivery whose SENDER does not hold the
#: target any more. Not a containment-snapshot key: the snapshot describes the target alone, and
#: this is a relation between the target's current manager and one entry's stamped
#: sender, so it is per-entry and cannot be a field on a per-slot snapshot.
SENDER_LOST_REACH = "sender_no_longer_manages"


def sender_lost_its_reach(state: "DashboardState", slot: "_ChatSlot", entry_meta: Any) -> bool:
    """Whether a queued delivery's SENDER is still allowed to drive *slot*.

    The caller-side half of the admission, re-asserted at the drain. The target-side
    half is the containment snapshot; this is the half a creator-read fence needs no
    re-check for, because ``_created_by`` is written at birth and never mutated, so the
    relation that admitted an entry cannot flip while it waits. A handoff MOVES that
    relation, which is the whole point of it, so with it the relation can: a lead queues a
    message on a busy worker, hands the worker to a conductor, and the message runs
    afterwards under an authorization the handoff revoked.

    Read through the SAME predicate ``authorize_target`` admitted the entry with -- not
    a second manager read placed here, and not a narrower one. With the line off that is
    :func:`_managed_by_other`; with it on it is the line, because that is what admitted
    the entry. Either goes through :func:`require_manager_of`, so an unread manager
    raises rather than answering, and the caller treats that as a drop for the reason
    the snapshot's ``manager_unknown`` names.

    THE ENTRY'S OWN STAMP decides which of the two, and there is no parameter to
    override it with. One stood here, for "a caller that has already resolved the
    switch", and no such caller existed: every production path reaches this through the
    drain, which has nothing resolved and must not read config on the loop. What the
    stamp answers is also the more correct question -- a re-check re-asserts the
    predicate that ADMITTED the entry, and an operator flipping the switch while a
    message waits does not change what admitted it.

    Only a CROSS-SESSION delivery is re-checked, and the entry's own provenance is what
    says it is one. A human-typed entry carries no :data:`ADMISSION_PROVENANCE_META_KEY`,
    so there is
    no sender to re-compare and nothing is dropped: that admission was the
    authenticated composer typing into the session's own surface and never read the
    manager.

    A RESTORED entry is re-checked like a live one. Its provenance
    (:data:`ADMISSION_PROVENANCE_META_KEY`) survives the restore, which strips only the
    routing stamp, so a message a revoked sender queued before a restart does not come
    back as an unfenced one and run. If the manager is still unread after the restart
    this raises, and the drain HOLDS the entry rather than dropping it.
    """
    if not send_origin_manager_fenced(entry_meta):
        # The fence did not apply to this admission, so the manager was never read for
        # it and there is no relation to re-compare. Covers an unfenced sender -- a
        # person's own tab driving session control -- plus every human-typed entry,
        # which carries no provenance at all.
        return False
    sender = send_origin_sender(entry_meta)
    if not sender:
        return False
    # THE SENDER'S OWN IDENTITY, read once and passed to every answer below, because a
    # key is a name and a replacement answers to the same name. Threaded through the
    # walk as well as the one-level read: an identity checked at the first hop and
    # dropped at the second is the key-only answer again, one hop further in.
    sender_tab = send_origin_sender_tab(entry_meta)
    # FROM THE ENTRY, not from config. The drain is synchronous and on the gateway's
    # loop: reading the switch there parses the file on a cache miss, and awaiting it
    # put a suspension in front of the dequeue. The entry knows what admitted it.
    chain = send_origin_chain_applied(entry_meta)
    if not chain:
        return _managed_by_other(state, slot, sender, sender_tab)
    # THE SAME POLICY THE ADMISSION USED, which with the line on is the whole line and
    # not the direct manager. Left on the one-level predicate, this re-check is STRICTER
    # than the admission it re-asserts: a lead that reached its grandchild worker through
    # its conductor is admitted by the fence, queues a prompt on a busy worker, and the
    # drain then reads "not its manager" and deletes the message -- a send accepted and
    # silently dropped with the reporting line never having changed. A re-check may
    # refuse what the admission refused; it may never refuse what the admission allowed.
    outside, cause = _outside_manager_line(state, slot, sender, sender_tab)
    if cause == "forgotten":
        # A RECORD THE CAP DROPPED IS NOT A LOST REACH, and it is not a wait either:
        # nothing re-arms it, so holding the entry would stop this session draining at
        # all. So the question is answered by the reach a gateway has WITHOUT this
        # feature -- the one-level relation, which is a fact about this process and not
        # about a record that aged out. A sender that is still the direct manager keeps
        # its delivery; one that is not gets the ordinary drop it would have got before
        # the line existed. Neither outcome rests on the forgotten record.
        return _managed_by_other(state, slot, sender, sender_tab)
    if cause in _CAUSES_THAT_HOLD_A_MESSAGE:
        # A LINE THIS GATEWAY CANNOT READ IS NOT A LOST REACH, and the difference is a
        # person's words. An ancestor mid-close answers "unread" and the close publishes
        # or rolls back in a moment; a record the cap dropped answers "unread" and never
        # comes back. Returned as "the sender lost its reach", either answer DELETES the
        # message -- a message the sender was told had been accepted, destroyed for a
        # condition that was never a decision about the sender at all.
        #
        # So it RAISES instead, which the drain already handles by holding the entry
        # until a read answers. Refuse an authorization; never destroy accepted text.
        raise SessionControlError(
            # GENERIC, like every other refusal raised from a manager READ. The body
            # named the sender's key and the target's TITLE, and both are facts about
            # sessions the reader of this refusal has not been shown it may reach: this
            # raise happens inside the authorization re-check, before anything has
            # decided that. Nothing needs them -- the drain is the only caller and reads
            # the CODE -- so what they bought was a disclosure surface waiting for a
            # second reader, which is how the same wording rule was broken on this line
            # before. The code stays specific because the caller has to know to hold.
            "a reporting line this delivery depends on could not be read to its end, "
            "so the delivery is held rather than judged",
            code="manager_line_pending",
            status=409,
        )
    return outside


#: The walk causes a queued delivery WAITS on, and only the ones that clear by
#: themselves may be in here. A hold is not free: the drain stops starting turns on that
#: session while the entry sits at the head, so holding on an answer nothing re-arms
#: wedges the queue -- including the person's own messages behind it -- until a restart.
#:
#: ``pending`` clears: an ancestor mid-close publishes or rolls back in a moment, and
#: the read is re-armed by the refusal itself. ``forgotten`` never clears, so it is
#: answered below instead of waited on.
_CAUSES_THAT_HOLD_A_MESSAGE = frozenset({"pending"})

#: THE REFUSAL CODES a queued entry is HELD on, as against dropped. The set above is
#: about one walk's causes; this one is about the codes the refusals carry, which is what
#: a caller outside this module has to read -- and there are two of them, so a reader
#: comparing against one code drops an entry the other code promised to hold. That is
#: not a style point: it is a person's accepted text deleted by a refusal whose own
#: message says it will be retried.
#:
#: Both clear by themselves, which is the only thing that may be in here. The unread
#: manager re-arms its own read; a hop mid-publication is published or rolled back by
#: the close or append that is already in flight. Every other manager refusal is a
#: decided answer about a record that STANDS, and holding on one of those shuts the
#: queue -- the person's own typed prompts included -- for the life of the process.
_MANAGER_REFUSALS_THAT_HOLD = frozenset({MANAGER_READ_PENDING, "manager_line_pending"})


def manager_refusal_holds_the_entry(code: str) -> bool:
    """Whether a manager refusal carrying *code* means HOLD this entry, not drop it.

    One function so the drain asks rather than lists. The list lived at the asking site
    as a single code, and this feature then added a second refusal that waits -- so the
    site kept comparing against the one it knew and deleted the accepted message the new
    refusal had just promised to hold. A reader cannot forget a member of a set it does
    not spell.
    """
    return code in _MANAGER_REFUSALS_THAT_HOLD


def describe_containment_change(constraints: list[str], *, mirror_unverified: bool = False) -> str:
    """One transcript-ready phrase naming what changed, for the drop notice.

    *mirror_unverified* swaps the mirrored wording: when the drain-side probe
    failed, the refusal stands (fail closed) but the notice must describe an
    unverifiable state, not assert a mirror appeared.
    """
    labels = dict(_CONTAINMENT_CHANGE_LABELS)
    if mirror_unverified:
        labels["mirrored"] = "the session's channel-mirror state could not be verified"
    return "; ".join(labels.get(c, c) for c in constraints)


def notify_send_origin_dropped(
    state: "DashboardState",
    *,
    origin: str,
    origin_tab: str = "",
    target_slot: "_ChatSlot",
    text: Any,
    constraints: list[str],
    mirror_unverified: bool = False,
) -> bool:
    """Tell the SENDING session that its queued delivery was dropped at the drain.

    ``send_to_target`` answers ``started: False`` when a busy target queues the
    message, and on its own that receipt says the message will run later. The
    drop notice, the retracted queue card and the broadcast all land on the
    TARGET, which the sender does not read, so the outcome a caller most needs —
    the message will never run — is the one it cannot see, and a caller polling
    the target's transcript waits for a reply that cannot come. This notice is
    what closes that.

    Returns whether a notice was appended. Four cases answer False and are not
    failures:

    * no stamp (``origin`` empty) — a human typed this into the composer, and
      there is no peer session waiting on it;
    * ``origin`` equals the target — a session that queued onto itself already
      has the target's own notice in the transcript it is reading, and a second
      row would report one drop twice;
    * the sending slot is gone — it was closed while the message waited, so
      there is no transcript left to write to. The SEL row still records the
      drop against the sender (:func:`audit_queued_drop`), which is what makes
      the outcome recoverable after the session is gone;
    * the key is live but holds a DIFFERENT occupant — ``origin_tab`` does not
      match the slot's ``_tab_id``. Named slot keys are deterministic and get
      reused, so this is the same case as the one above wearing the previous
      tenant's name, and writing anyway would put the sender's text on a session
      that never sent it. Treated as "the sender is gone", because it is.

    An absent ``origin_tab`` answers False whenever a slot is found, so a stamp
    that cannot be identity-checked never writes: the check is not skippable by
    omitting its input.

    Best-effort, like the target-side notice: a failure here is logged and the
    drop still proceeds. Withholding the message is the authorization decision,
    and it must not depend on the report landing.
    """
    target_key = str(getattr(target_slot, "key", ""))
    sender = _send_origin_recipient(state, origin, origin_tab, target_key)
    if sender is None:
        return False
    try:
        excerpt = send_drop_excerpt(text)
        sender.append(
            "notice",
            f"⚠️ Message you sent to {target_key} was dropped before it ran: "
            + describe_containment_change(constraints, mirror_unverified=mirror_unverified)
            + " after it was queued, so the authorization that admitted it no "
            + "longer holds. It was not delivered and will not run."
            + (f' Text: "{excerpt}"' if excerpt else ""),
            "msg msg-info",
        )
    except Exception:  # pragma: no cover - reporting must not block the drop
        logger.exception(
            "Failed to report a dropped delivery to its sender (origin=%s, target=%s)",
            origin,
            target_key,
        )
        return False
    return True


def notify_send_origin_discarded(
    state: "DashboardState",
    *,
    entry_meta: Any,
    target_slot: "_ChatSlot",
    text: Any,
    cause: str,
) -> bool:
    """Tell the SENDING session that a deliberate discard removed its queued delivery.

    The drain is not the only place a stamped entry leaves the queue unrun. A
    person cancelling its queue card (``api_chat_slot_queue_cancel``) and a hard
    stop clearing the queue (``stop_slot_turn``) discard it too, and the sender's
    last word on the message is still the ``started: False`` receipt that said it
    would run later. Every other sign of the discard lands on the target, which
    the sender does not read.

    *cause* is :data:`SEND_DISCARD_QUEUE_CANCEL` or :data:`SEND_DISCARD_HARD_STOP`
    and selects the wording: a deliberate discard changes no authorization, so the
    notice names what discarded the entry rather than a constraint.

    *entry_meta* is the discarded entry's own ``meta``, read here rather than by
    the caller so both discard sites read the stamp the way the drain does. The
    recipient is resolved exactly as :func:`notify_send_origin_dropped` resolves
    it, so the same four cases answer False without writing: no stamp, a
    self-send, a closed sender, and a key now held by a different tab.

    Best-effort: a failure is logged and the discard stands, because the person
    or the stop already decided it.
    """
    target_key = str(getattr(target_slot, "key", ""))
    origin = send_origin_slot(entry_meta)
    sender = _send_origin_recipient(state, origin, send_origin_tab(entry_meta), target_key)
    if sender is None:
        return False
    try:
        reason = _SEND_DISCARD_REASONS[cause]
        excerpt = send_drop_excerpt(text)
        sender.append(
            "notice",
            f"⚠️ Message you sent to {target_key} was discarded before it ran: "
            + reason
            + ". It was not delivered and will not run."
            + (f' Text: "{excerpt}"' if excerpt else ""),
            "msg msg-info",
        )
    except Exception:
        logger.exception(
            "Failed to report a discarded delivery to its sender (origin=%s, target=%s)",
            origin,
            target_key,
        )
        return False
    return True


def _send_origin_recipient(
    state: "DashboardState", origin: str, origin_tab: str, target_key: str
) -> "_ChatSlot | None":
    """The live sending slot a sender notice may be written to, or None.

    One resolver for both sender notices, so the four non-notice cases
    :func:`notify_send_origin_dropped` documents cannot hold on one discard path
    and not another: no stamp, a self-send, a closed sender, and a reused key
    whose current occupant is a different tab. An absent *origin_tab* never
    matches a live slot, so the identity check cannot be skipped by omitting it.
    """
    if not origin or origin == target_key:
        return None
    sender = state._slots.get(origin)
    if sender is None:
        return None
    if str(getattr(sender, "_tab_id", "") or "") != str(origin_tab or ""):
        return None
    return sender


def notify_channel_recipient_dropped(
    state: "DashboardState",
    *,
    entry_meta: Any,
    target_slot: "_ChatSlot",
    text: Any,
    constraints: list[str],
    mirror_unverified: bool = False,
) -> bool:
    """Tell the CHANNEL CONVERSATION that sent a message that the drain dropped it.

    The channel counterpart of :func:`notify_send_origin_dropped`. A message a
    channel handed to a resumed dashboard slot's queue
    (``dashboard.channel_handoff``) was confirmed "queued" in that conversation,
    and the conversation reads neither the target's transcript nor the SEL -- so
    without this the one outcome the author most needs, that the message will
    never run, is the one they are never told, and they wait for a reply that
    cannot come.

    Returns whether a notice was SCHEDULED: True once the entry carries a channel
    stamp and the task below exists. Answers False, and is not a failure, when the
    entry carries no stamp (a dashboard-typed or restored entry) or when no loop
    is running to carry the task. Whether the notice is then SENT is decided
    inside the task, and its refusals are logged and audited there.

    The send goes through the cross-surface ladder every proactive channel
    delivery takes (``chat_runner._resolve_channel_target``: channels governance,
    a registered transport that can send proactively, and the RECIPIENT
    re-check), with the principal the stamp recorded -- the platform user the
    channel authorized on inbound -- because a dashboard session key names no
    channel peer for the ladder to derive one from. A revoked recipient gets no
    notice; the refusal is audited by the ladder. The mirror pause is NOT
    consulted: this is a delivery receipt to the message's own author, not turn
    output, and the pause mutes output.

    Nothing of the ladder runs on the calling thread. The drain that drops the
    entry is synchronous on the event loop by contract (no suspension between its
    snapshot and the dequeue), so the send cannot be awaited there -- and the
    ladder's governance vet is the call every other async caller offloads,
    because it reads and validates policy on the shared loop. Both therefore run
    inside the scheduled task: the resolve through ``asyncio.to_thread``, the
    send awaited after it. The task is held in the state's background set so it
    cannot be collected mid-flight. Best-effort throughout: a failure is logged
    and the drop, which is the authorization decision, stands.

    The quoted excerpt is redacted through the same egress chain every channel
    delivery uses: the author typed the text, but this is a network write and the
    conversation may be read on a shared screen.
    """
    recipient = channel_recipient_of(entry_meta)
    if recipient is None:
        return False
    channel_type, conversation_id, principal = recipient
    # circular import: chat_runner imports this module's helpers at module level.
    from kiro_crew.dashboard.chat_runner import _resolve_channel_target
    from kiro_crew.dashboard.chat_utils import _redact_for_display

    link = ChannelLink(channel_type, channel_id=conversation_id)
    session_key = slot_history_key(target_slot)
    excerpt = _redact_for_display(sanitize_outbound(send_drop_excerpt(text)))
    notice = (
        "⚠️ Your queued message to that session was dropped before it ran: "
        + describe_containment_change(constraints, mirror_unverified=mirror_unverified)
        + " after it was queued, so the authorization that admitted it no longer "
        + "holds. It was not delivered and will not run; send it again if it still applies."
        + (f' Text: "{excerpt}"' if excerpt else "")
    )

    async def _send() -> None:
        try:
            target = await asyncio.to_thread(
                _resolve_channel_target, state, session_key, link, principal=principal
            )
        except Exception:
            logger.warning(
                "channel drop notice: send ladder failed for %s; the drop is not reported",
                channel_type,
                exc_info=True,
            )
            return
        if target is None:
            return
        resolved_link, transport = target
        try:
            await transport.send_message(
                resolved_link.channel_id, notice, thread_id=resolved_link.thread_id
            )
        except Exception:
            logger.warning("channel drop notice: send to %s failed", channel_type, exc_info=True)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.warning("channel drop notice: no running loop to carry the send to %s", channel_type)
        return False
    task = loop.create_task(_send())
    background = getattr(state, "_background_tasks", None)
    if isinstance(background, set):
        background.add(task)
        task.add_done_callback(background.discard)
    return True


def audit_queued_drop(
    slot: "_ChatSlot", queue_id: str, constraints: list[str], *, origin: str = ""
) -> None:
    """Record one drain-time drop in the SEL, best-effort and off the loop.

    Logged as a denied tool invocation on the TARGET's EFFECTIVE session — a
    linked slot's turns run under ``linked_session_key``, so filing under the
    slot key would hide exactly the drops this feature exists to record. The
    slot key stays in ``resources``/``metadata``.

    *origin* is the sending slot for a cross-session delivery, read from the
    entry's own stamp (:data:`SEND_ORIGIN_META_KEY`), so the trail names who was
    waiting on the dropped message. A human-typed entry carries no stamp and the
    field is omitted rather than recorded empty.
    """
    _audit_queue_drain(
        slot, outcome="denied", queue_ids=[queue_id], newly_held=constraints, origin=origin
    )


def audit_queued_allow(slot: "_ChatSlot", queue_ids: list[str]) -> None:
    """Record that re-validated queued entries were AUTHORIZED to become a turn.

    The allow side of the same permission decision :func:`audit_queued_drop`
    records the deny side of — both outcomes are auditable, matching
    ``authorize_target``'s convention of logging ``allowed`` operations and not
    only refusals. Emitted at CONSUMPTION (the moment the drain hands the
    entries to a turn), not per sweep pass, so an entry that waits across
    several drains produces one row when it actually executes rather than one
    per re-check. One row covers the whole consumed batch.
    """
    _audit_queue_drain(slot, outcome="allowed", queue_ids=queue_ids, newly_held=None)


def _audit_queue_drain(
    slot: "_ChatSlot",
    *,
    outcome: str,
    queue_ids: list[str],
    newly_held: list[str] | None,
    origin: str = "",
) -> None:
    slot_key = str(getattr(slot, "key", ""))
    session_key = effective_session_key(slot)
    metadata: dict[str, Any] = {
        "target": slot_key,
        "queue_ids": ",".join(queue_ids),
    }
    if newly_held is not None:
        metadata["newly_held"] = ",".join(newly_held)
    if origin:
        # Omitted rather than recorded empty: absent means "no sending slot was
        # stamped" (a human typed it), which a reader must be able to tell from a
        # cross-session delivery whose sender happens to be unnamed.
        metadata["origin"] = origin

    def _do() -> None:
        sel().log_tool_invocation(
            session_key=session_key,
            agent="",
            source="dashboard",
            tool_name="queue_drain_revalidation",
            tool_kind="command",
            outcome=outcome,
            resources=f"target={slot_key}",
            metadata=metadata,
        )

    _sel_off_loop(_do, "queue-drain revalidation audit")


def _refuse_moved_caller_identity(
    state: "DashboardState",
    caller_key: str,
    caller_slot: "_ChatSlot",
    identity: tuple[str, str, str],
) -> None:
    """Refuse a caller whose memory identity differs from *identity*.

    *identity* is the ``(history key, agent, memory store)`` triple
    :func:`create_session` captured before its first suspension point; the live
    slot is re-read and compared here, at every later point that derives a
    verdict from the caller. A caller that is gone, or whose key has been
    re-minted onto another session, is refused as not open; one that survived
    but changed what it is -- a channel link landing on it changes its history
    key, a reassignment changes its store or agent -- is refused as the identity
    change it is, so the decisions taken on the earlier identity are never
    applied to the new one and the refusal names the cause rather than whichever
    downstream gate happened to read the new identity first.
    """
    live = state.get_slot(caller_key)
    if live is None or live is not caller_slot:
        raise SessionControlError("caller session is not open", code="caller_not_open", status=404)
    if (slot_history_key(live), live.agent, live.memory_store) != identity:
        raise SessionControlError(
            "caller session changed memory assignment while the session was being created",
            code="caller_memory_changed",
        )


def _refuse_ineligible_creator(state: "DashboardState", caller_slot: "_ChatSlot") -> None:
    """Refuse a caller that may not manufacture a session.

    A caller that may not CONTROL a peer may not manufacture one either --
    otherwise a channel-bound session creates a session and then drives it,
    reaching the same place the caller-side refusals exist to prevent. This set
    therefore mirrors `authorize_target`'s caller half exactly; a refusal present
    there and missing here is a hole.

    Extracted so it can be applied TWICE: once on entry, so an ineligible caller
    is refused before any work is done and with the refusal precedence a caller
    can rely on, and again immediately before the slot is allocated. Two of these
    answers are not stable -- `_has_channel_mirror` reads the live session store,
    and a dashboard-born session can be given an outbound mirror link at any
    moment -- so an eligibility decided before a suspension point says nothing
    about eligibility at the moment of allocation.
    """
    if getattr(caller_slot, "_app", ""):
        # An app-scoped session is confined to its own app's slots. Creating a
        # plain user-origin slot would put a persistent, sidebar-visible session
        # outside that confinement, owned by the app.
        raise SessionControlError(
            "app-scoped sessions cannot create sessions", code="app_scoped_caller"
        )
    if (refusal := _app_owned_cron_refusal(state, getattr(caller_slot, "key", ""))) is not None:
        # The same confinement, reached through an app's cron rather than its
        # session -- a cron tab carries no ``_app`` tag for the check above to
        # read. See :func:`_app_owned_cron_refusal`.
        raise SessionControlError(refusal[0], code=refusal[1])
    if getattr(caller_slot, "memory_mode", "persistent") != "persistent":
        # An incognito/temporary caller is defined by leaving nothing behind.
        # A persistent child it owns would outlive it, carrying its work into
        # storage the caller was promised would not retain anything.
        raise SessionControlError(
            "incognito and temporary sessions cannot create sessions",
            code="ephemeral_caller",
        )
    # The channel link and mirror refusals share ONE exemption with
    # `authorize_target`'s caller half: :func:`owner_dm_refusal` answering ``""``,
    # a 1:1 DM whose only human is the configured owner -- the DM the session lives
    # in, or the DM a dashboard tab mirrors to -- with no wider room bound beside
    # it. It waives both together, because the predicate has already established
    # that the mirror IS the DM -- waiving the link alone would refuse every owner
    # DM on the origin mirror its dispatcher binds each turn. The refusal names the
    # clause that failed: the code stays the same, but a DM that lost its origin to
    # a gateway restart, or a mirror whose peer this process holds no record of, is
    # told to send a message rather than left hunting for a link it cannot clear.
    # Both refusals are decided from the ONE read the verdict took: its ``mirrored``
    # flag is that row's fail-closed reading, so the gate never probes the store a
    # second time to decide what the first read already saw.
    verdict = judge_owner_dm(state, caller_slot)
    if why := verdict.refusal:
        if _channel_link_of(caller_slot):
            # A cron tab's link is its own run transcript, not a channel thread,
            # and is exempt -- see CRON_LINK_PREFIX. Everything else is a channel
            # link.
            raise SessionControlError(
                "channel-linked sessions cannot create sessions; the owner-DM "
                f"exemption is withheld because {why}",
                code="linked_session_caller",
            )
        if verdict.mirrored:
            raise SessionControlError(
                "sessions mirrored to a channel cannot create sessions; the owner-DM "
                f"exemption is withheld because {why}",
                code="mirrored_caller",
            )
    # Admitted. A creation's result -- the child's key and title -- becomes part of
    # the caller's reply exactly as a read does, and ``create_session`` and a
    # ``fork_session`` of the caller's own transcript never pass ``authorize_target``,
    # so this is the ONE gate such a turn passes: record the audience here too, from
    # the row this verdict judged. Applied twice per creation (entry and allocation),
    # the second write is the no-op the audience-keyed record makes it.
    record_audience_admission(caller_slot, verdict.admission)


def _resolve_slot(
    state: "DashboardState",
    target: str,
    *,
    candidates: "list[_ChatSlot] | None" = None,
) -> "_ChatSlot | None":
    """Find the live slot *target* names: by slot key, transcript stem, or title.

    All three forms are things a caller actually holds. ``list_sessions`` reports
    FILENAME STEMS (``dashboard_chat-7``), not slot keys (``chat-7``), and the
    tool description tells callers to pass what it returned — so matching only
    ``slot.key`` refused the documented happy path with ``target_not_found``.
    Title matching covers what the caller sees on screen; it is exact and
    case-insensitive.

    Every form is resolved before anything is returned, and a string that matches
    two DIFFERENT slots across forms is refused as ambiguous. Returning on the
    first key hit would silently prefer it over a title the caller was reading off
    the screen, and picking the wrong conversation is exactly the outcome this
    function must never produce — ``session_stop`` discards a live turn's work.
    The doctrine is already the module's own for title-vs-title collisions; it
    applies no less when the collision crosses forms.

    ``candidates`` restricts every form, including direct key lookup, to a set the
    caller already proved safe to name. Broadcast uses that form so resolving a
    guessed private title cannot reveal the slot key that carries it.
    """
    found: list[_ChatSlot] = []
    pool = list(state._slots.values()) if candidates is None else list(candidates)

    def _add(candidate: "_ChatSlot") -> None:
        if not any(c is candidate for c in found):
            found.append(candidate)

    slot = state.get_slot(target)
    if slot is not None and any(candidate is slot for candidate in pool):
        _add(slot)
    for candidate in pool:
        try:
            if transcript_stem(slot_history_key(candidate)) == target:
                _add(candidate)
        except Exception:
            continue
    wanted = target.strip().casefold()
    if wanted:
        for candidate in pool:
            if (candidate.display_title or "").strip().casefold() == wanted:
                _add(candidate)

    if len(found) > 1:
        raise SessionControlError(
            f"{len(found)} sessions match {target!r} (as a session key, transcript "
            "name, or title) — address it by its session key instead",
            status=409,
            code="ambiguous_target",
        )
    return found[0] if found else None


async def load_target_window(state: "DashboardState", target: str) -> None:
    """Load the pending window of the live session *target* names, off the loop.

    For a verb whose gate and delivery read the target's rows with no suspension
    between them: it runs before that verb's prewarm, the last suspension ahead
    of the gate. It resolves *target* as the gate does and decides nothing, so a
    target the gate then refuses has had only the load its next reader would do.
    """
    target = (target or "").strip()
    if not target:
        return
    try:
        slot = _resolve_slot(state, target)
    except SessionControlError:
        return
    if slot is not None:
        await load_window_off_loop(slot)


def _broadcast_resolution_slots(
    state: "DashboardState",
    *,
    caller_key: str,
    caller_slot: "_ChatSlot",
    ownership_fenced: bool,
) -> "list[_ChatSlot]":
    """Live slots whose names *caller_key* may safely resolve before delivery.

    This mirrors :func:`authorize_target`'s target containment clauses without
    replacing that gate: every retained key is authorized again at delivery.
    Building the set BEFORE reading a candidate's title prevents an ownership-
    fenced caller from using title resolution to learn a private slot key: a name
    outside the set stays the caller's own string all the way into its refusal row,
    so a guessed title is never answered with the slot key that carries it.

    THE KEY IS WHAT THIS WITHHOLDS, NOT EXISTENCE. The unresolved name is still
    delivered to :func:`authorize_target`, whose ``_resolve_slot`` call is
    UNRESTRICTED -- so a guessed title that happens to match a slot the caller may
    not touch comes back as that slot's containment class (``not_creator``,
    ``mirrored_target``, ``ephemeral_target``, ``app_scoped_target``,
    ``workspace_mismatch``) rather than ``target_not_found``, and the difference
    tells the caller something is there. That is not a leak this function opened
    and not one it can close: ``session_send`` on the same name answers identically,
    which is the documented contract ("this can reach nothing a session_send
    could not"). The codes are also the only guidance a caller gets about a target
    it named legitimately, so collapsing them here would cost a conductor the
    reason its own worker is unreachable. The narrow property this set buys is the
    one that matters for a guessing caller: a refusal row never hands back a
    durable handle it did not already hold.
    """
    out: list[_ChatSlot] = []
    for slot in list(state._slots.values()):
        if slot.key == caller_key or slot.key.startswith(UNATTENDED_SLOT_PREFIXES):
            continue
        if getattr(slot, "memory_mode", "persistent") != "persistent":
            continue
        if getattr(slot, "_app", "") or getattr(slot, "linked_session_key", ""):
            continue
        if _has_channel_mirror(state, slot):
            continue
        if getattr(slot, "workspace", "default") != getattr(caller_slot, "workspace", "default"):
            continue
        # The MANAGER reading, not the creator's, because this set decides which names a
        # broadcast may resolve and a broadcast delivers ``send`` -- one of the verbs a
        # handoff widens. Read ``_created_by`` here and a conductor handed a worker gets
        # that worker in its audience and its roster while its title resolves to
        # nothing, so the one session it was given is the one it cannot address by name.
        #
        # Guarded the way the audience scan is (:func:`_holds_for_audience`): an
        # unreadable manager EXCLUDES the name rather than raising, because this
        # function lists rather than authorizes, and every retained key is authorized
        # again at delivery where the real refusal carries the real reason.
        if ownership_fenced and not _holds_for_audience(state, slot, caller_key):
            continue
        out.append(slot)
    return out


async def create_session(
    state: "DashboardState",
    *,
    caller_session_key: str,
    title: str = "",
    agent: str = "",
    agent_kind: str = "",
    folder_id: str = "",
    model: str = "",
    caller_fenced: bool | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Open a new session in the caller's workspace, persisted at birth.

    The new slot is an ordinary dashboard session -- it appears in the sidebar, the
    user can read it, type into it and close it -- so this gives a workstream a home
    of its own rather than a private channel the user cannot see. It starts empty:
    the person is the one who types the first message into it.

    The caller's own eligibility is checked against the SAME caller-side refusal
    set `authorize_target` applies (`authorize_target` cannot be reused here:
    there is no target yet), and the child inherits the caller's workspace. Both
    matter because a caller refusal missing here, or a workspace not inherited,
    would hand back a session outside the boundary the other verbs enforce.

    The caller's session POSTURE (``_trust`` / ``_trust_reads``) transfers to the
    child, so a trusted operator's dispatched worker does not stall on a prompt
    nobody is watching -- the posture ``parent_trusted`` already gives a
    ``spawn_run`` subagent. Per-command grants (``_trusted_patterns``) and a
    ``SafetyOverride`` scoped grant (``_trust_scope``) are both deliberately
    excluded, and the transferred value is the one held at allocation time rather
    than at entry, so revoking mid call yields an untrusted child. See the block
    around the assignment.

    ``folder_id`` files the slot as part of creation: it is assigned in
    the same synchronous window that configures the slot, the whole
    allocation-to-persist span runs under ``suspend_slots_push`` so the slot's
    first broadcast frame already shows it filed, and the placement rides in the
    persist-at-birth metadata so it survives a restart. An unknown folder
    refuses the whole create -- nothing exists yet, so refusal loses nothing,
    matching the move path's posture -- and existence is confirmed READ-ONLY
    under the folder-store lock (``read_folders``) before the allocation; the
    Model-B un-hide runs only after the filing has landed, so a refused create
    leaves no folder-tree mutation behind. Authorization needs no new path: the
    folder tree cannot be reshaped from here (the id must already exist), and
    every caller class the move path's app-ownership rule exists to stop is
    already refused above it -- an app-scoped caller cannot create a session at
    all (`app_scoped_caller`).

    ``model`` pins the model the child starts on, as the person's own pick in the
    model dropdown would: same rejection guard, same pick-generation bump, and
    recorded in the persist-at-birth metadata so an idle child keeps it across a
    restart. Empty leaves the slot on the agent's or the global default, exactly
    as before. It is not a privilege the caller lacks -- ``spawn_run`` already
    takes a per-run ``model`` -- and it moves no memory boundary.

    ``caller_fenced`` is the ownership-fence verdict the HTTP gate already settled
    on the caller's VERIFIED scope, carried in for the same reason
    ``authorize_target`` takes ``precomputed_ownership_fenced``: the inline
    predicate re-derives member status from the MUTABLE config record, and this
    coroutine suspends many times before it is consulted. ``None`` means "not
    settled", and the fence is then evaluated inline. It is read by the
    private-store authorization below and nothing else.

    ``agent_kind`` is the selection NAMESPACE for ``agent``, and this route takes
    one value: ``"template"`` names the shared provider template of that name,
    and ``""`` keeps the name-first resolution (a crew member wins over a
    same-named template). It exists for the one case the bare name cannot
    express: a template that a crew member of the same name shadows. The stock
    ``kirocrew-worker`` is such a name on installs where an older agent sync
    enrolled it as a crew member with its own private memory -- a bare
    ``agent="kirocrew-worker"`` then selects the MEMBER, and a conductor that may
    not bind that member's store (a member DM thread, a fenced deputy) is refused
    ``memory_delegation_denied`` for a worker that needs no memory of its own.
    ``agent_kind="template"`` dispatches the worker template on the caller's own
    store instead, exactly as the chat route's explicit template pick "never pins
    member memory". There is no ``"member"`` kind here: a bare name already
    selects the member, so the value would only restate that resolution. The kind
    is selection input, never authority: every gate below still applies, and a
    template child takes the caller's store the same way an unshadowed template
    already does. It is recorded on the slot (``agent_kind``) and in the birth
    metadata, so a restart rehydrates the child in the namespace it was committed
    in rather than re-resolving the bare name. A stated kind requires ``agent``:
    the namespace qualifies a selection, and an inherited caller agent is not one.

    ``dry_run`` runs every gate up to the allocation, including the two slot
    ceilings, and returns ``{"dry_run": True}`` instead of minting a slot. It
    spends no rate-limit token and writes nothing. The MCP ``session_create``
    asks it first when its ``folder`` path still has segments to create, so a
    create that would be refused is refused BEFORE any folder exists: without
    it the path walk's folders outlive the refusal as empty sidebar rows.
    ``folder_id`` is then the deepest folder that already exists, which is the
    one the new segments would inherit a project directory from.
    """
    caller_key = caller_slot_key(state, caller_session_key)

    if not caller_key:
        raise SessionControlError(
            "caller session could not be identified", code="caller_unidentified"
        )
    # The caller is resolved BEFORE the config gate so a member DM session —
    # for which dispatching work into workers is the operating model, not an
    # opt-in — passes without `agent.session_control`. That member bypass is
    # itself gated by the operator ceiling `agent.member_dispatch` (default
    # true = today's behaviour); with it off the member falls back under the
    # switch like any other caller. Every other caller still needs the switch.
    # The member's automatic grant is bounded by ownership in `authorize_target`,
    # not here: creation makes the caller the owner by construction.
    if not session_control_enabled() and not _member_bypass(state, caller_key):
        raise SessionControlError(
            "session control is disabled in config (agent.session_control)",
            code="session_control_disabled",
        )
    if caller_key.startswith(UNATTENDED_SLOT_PREFIXES) and not _cron_caller(caller_key):
        raise SessionControlError(
            "unattended sessions (scheduled runs) cannot create sessions",
            code="unattended_caller",
        )
    caller_slot = state.get_slot(caller_key)
    if caller_slot is None:
        raise SessionControlError("caller session is not open", code="caller_not_open", status=404)
    _refuse_ineligible_creator(state, caller_slot)
    caller_memory_identity = (
        slot_history_key(caller_slot),
        caller_slot.agent,
        caller_slot.memory_store,
    )

    try:
        caller_execution = await asyncio.to_thread(
            read_session_execution, caller_memory_identity[0]
        )
    except (ValueError, OSError):
        raise SessionControlError(
            "caller execution context is unavailable", code="memory_unavailable"
        ) from None

    # This process's own word on the caller, snapshotted in the SAME breath as the
    # record above so the two sources the own-store admission compares below are
    # taken at one instant. Reading it down there instead would make a privacy
    # change landing mid-resolution surface as a delegation refusal, when the
    # re-gate further down exists precisely to name that case. A plain dict read
    # under a lock, so it adds no suspension point here.
    caller_vouched = read_vouched_session_execution(caller_memory_identity[0])

    if caller_execution is not None and caller_execution.memory_mode != "persistent":
        raise SessionControlError(
            "incognito and temporary sessions cannot create sessions", code="ephemeral_caller"
        )

    # The child is created in the CALLER'S workspace, not the default one.
    # Workspace is the memory boundary and `authorize_target` refuses a
    # cross-workspace target, so a child left in "default" would be a boundary
    # crossing its own creator could not then read or stop.
    workspace = getattr(caller_slot, "workspace", "default") or "default"

    # Match dashboard-native creation: filing a dispatched session in a
    # project-linked folder gives the child that folder's nearest inherited
    # project directory, while the workspace and memory boundary stay the
    # caller's. Capture the raw inherited value so the late folder re-check can
    # refuse a concurrent reparent or project edit instead of creating against
    # stale folder intent.
    folder_project_raw: str | None = None
    folder_project = ""
    if folder_id:
        folder_snapshot = await state.read_folders(
            lambda folders: [dict(folder) for folder in _safe_folder_tree(folders)]
        )
        if not any(str(folder.get("id") or "") == folder_id for folder in folder_snapshot):
            raise SessionControlError("folder not found", code="folder_not_found")
        folder_project_raw, folder_project_error = _folder_declared_project(
            folder_snapshot, folder_id
        )
        if folder_project_error:
            raise SessionControlError(
                f"invalid folder project: {folder_project_error}",
                code="folder_project_invalid",
            )
        folder_project, folder_project_error = await asyncio.to_thread(
            _resolve_folder_project_dir, folder_snapshot, folder_id
        )
        if folder_project_error:
            raise SessionControlError(
                f"invalid folder project: {folder_project_error}",
                code="folder_project_invalid",
            )
    # An unnamed agent inherits the CALLER'S, not the global default: the caller is
    # already running in this workspace, so its agent is the one bound here, and
    # falling to the global default would put the child on another workspace's
    # memory store the moment the default is bound elsewhere. It also matches what
    # creating a session to hand work to means -- the same kind of session.
    # Sanitized like `title` below, and for the same reason: this value arrives
    # from the calling model, is persisted verbatim to the metadata line, and is
    # pushed to every dashboard client. The schema caps its LENGTH; sanitizing is
    # what keeps a credential-shaped string out of storage and out of the sidebar.
    # An inherited caller agent is already internal, but running both through the
    # same call keeps the guard on the field rather than on one of its sources.
    agent_name = sanitize_outbound(agent.strip() or (getattr(caller_slot, "agent", "") or ""))
    # The selection namespace, validated HERE and not left to the HTTP route or
    # the MCP schema: `create_session` is also reached in-process, and a kind
    # outside the closed set would otherwise travel into `resolve_agent_bindings`
    # as a silent name-first resolution. A kind without an explicit `agent`
    # qualifies nothing -- the inherited caller agent is not a selection -- so it
    # is refused rather than ignored.
    if agent_kind not in ("", "template"):
        raise SessionControlError('agent_kind must be "template"', code="invalid_agent_kind")
    if agent_kind and not agent.strip():
        raise SessionControlError(
            "agent_kind qualifies an explicit agent; name the agent too",
            code="agent_kind_requires_agent",
        )

    log = state.conversation_log
    if log is None:
        # No durable store means the session cannot be persisted at birth, so it
        # would vanish on the next restart. Refusing is the honest answer;
        # returning a key would hand back a session that is dead on arrival.
        raise SessionControlError(
            "session history is unavailable, so the session cannot be persisted",
            code="history_unavailable",
        )

    # Resolved BEFORE the slot exists, because `get_or_create_slot` publishes into
    # the slot table and `await` is a suspension point: a slot that is visible
    # while its agent and project are still unset can be addressed in that window,
    # and `/api/chat` would then resolve bindings from a blank agent -- running the
    # turn against the DEFAULT workspace's memory store rather than this one.
    # `default_project_dir` needs only the workspace name, so nothing forces it to
    # run after construction.
    #
    # Offloaded: it resolves a realpath, stats the directory and screens it against
    # the sensitive-path list, so it is filesystem work the loop should not wait on.
    # The rule's own tiebreaker applies -- a leaked worker thread is survivable, a
    # frozen loop is not.
    project_dir = folder_project or await asyncio.to_thread(default_project_dir, workspace)

    # ONE invariant covers every branch of agent resolution: the agent that will
    # actually ANSWER must be bound to the caller's workspace. Authorization reads
    # `slot.workspace` while execution follows the agent's own binding, so any
    # branch where those disagree carries another workspace's memory store into
    # the child. Enumerating the branches instead of stating the invariant is how
    # the empty-agent case was missed:
    #
    #   agent given, binding matches   -> allowed, dispatches that agent
    #   agent given, binding differs   -> refused (agent_workspace_mismatch)
    #   agent given, name unresolvable -> refused (agent_unresolved), because the
    #                                     default would answer under the requested
    #                                     name
    #   agent omitted                  -> `resolve_agent_bindings` falls to
    #                                     config.default_agent, so the SAME check
    #                                     applies to whatever would answer; an
    #                                     omitted agent is not an unchecked one
    #   config unreadable              -> refused (agent_unverifiable), because
    #                                     "cannot verify" must not read as "fine"
    #
    # Resolved with the child's own `project_dir`: a materialized kiro agent is
    # declared per project directory rather than registered in `config.agents`, so
    # resolving without it reports an app's agent as unresolvable and would refuse
    # a name that does resolve for the session being created.
    try:
        # Offloaded: a cache miss reads and validates the config file, so leaving it
        # on the loop stalls every other gateway task, not just this request. It is
        # awaited HERE, still ahead of the caller re-resolve below, so the decisions
        # that authorize the allocation are all made after the last suspension.
        cfg = await asyncio.to_thread(KiroCrewConfig.load)
        # Resolved in the STATED namespace when the caller gave one, so a
        # template pick takes the template even when a crew member carries the
        # same name -- the chat create route's contract, kept identical here.
        bindings = await asyncio.to_thread(
            resolve_agent_bindings,
            cfg,
            agent_name,
            project_dir,
            validate_memory_files=False,
            selection_kind=agent_kind,
        )
    except Exception:
        raise SessionControlError(
            "cannot verify the effective agent's workspace binding",
            code="agent_unverifiable",
        ) from None
    agent_workspace = _workspace_name_for_dir(cfg, bindings.workspace_dir)
    if agent_workspace != workspace:
        who = repr(agent_name) if agent_name else "the default agent"
        raise SessionControlError(
            f"{who} is bound to workspace {agent_workspace!r}, not the caller's " f"{workspace!r}",
            code="agent_workspace_mismatch",
        )
    if not bindings.requested_resolved:
        # The workspace check above passed for whatever ANSWERS -- the default
        # agent -- so no memory boundary is crossed. What would be wrong is the
        # record: `slot.agent` stores this name, and `ResolvedBindings` states the
        # contract for exactly this caller class, that a caller storing the
        # requested name must not advertise it when the request was not honored.
        # A session that names one agent while another answers misleads every later
        # reader of the sidebar and of `list_sessions`.
        #
        # Refused rather than silently rewritten to the effective agent, because
        # the caller asked for a specific one and nothing is lost by refusing: no
        # session exists yet, and a corrected name is one retry away. (An existing
        # slot is the opposite case -- there the stored name is the user's own
        # intent and is kept verbatim, since a momentarily stale resolution must
        # not permanently rebind it.) A stated namespace never falls back to
        # whoever answers by default either, and its refusal names the namespace
        # so the caller can tell "no such template" from "no such agent".
        if agent_kind:
            raise SessionControlError(
                f"{agent_name!r} does not resolve to a configured {agent_kind}",
                code="agent_unresolved",
            )
        raise SessionControlError(
            f"{agent_name!r} does not resolve to a configured agent",
            code="agent_unresolved",
        )
    # The namespace the selection RESOLVED in, taken from the resolver's verdict
    # before `bindings` is rewritten below to carry the child's execution route
    # (whose `selection_kind` is the child's identity, which for a template child
    # of an id-less member is still "member"). This is what the slot records when
    # the caller stated a kind -- the chat create route's `chosen.selection_kind`.
    committed_kind = bindings.selection_kind if agent_kind else ""

    # The model the child starts on. Checked by the SAME guard the dashboard's
    # model picker runs (`_model_rejected_reason`), against the provider from the
    # config snapshot already loaded off the loop, so an id the picker would refuse
    # is refused here too -- before anything is allocated, where refusing loses
    # nothing. Length and charset are bounded HERE, not left to the MCP schema:
    # SESSION_CREATE_SCHEMA runs only in mcp_dashboard, and the HTTP route
    # forwards the body's model string as-is, so without this bound an
    # internal-secret caller could persist, broadcast and audit-log an
    # arbitrarily long or arbitrarily shaped value.
    model_name = _normalize_model(model.strip())
    if model_name:
        if len(model_name) > MAX_SHORT_STRING or not _MODEL_NAME_RE.fullmatch(model_name):
            raise SessionControlError(
                "model id is too long or contains characters outside the model-id charset",
                code="model_rejected",
            )

        # Never persist or broadcast a value the security scrubber classifies.
        if redact(model_name) != model_name:
            raise SessionControlError(
                "model id looks like a credential and was refused",
                code="model_rejected",
            )

        # circular import: chat_handlers imports session_control lazily, and this
        # module is imported by the dashboard package before chat_handlers loads.
        from kiro_crew.dashboard.chat_handlers import _model_rejected_reason

        model_reason = _model_rejected_reason(model_name, provider=cfg.agent.provider or "")
        if model_reason:
            raise SessionControlError(model_reason, code="model_rejected")

    # Capture the child route once. Inherited member/store identity survives
    # renamed aliases or changed config; an explicit member selection is resolved
    # from the config snapshot admitted above.
    try:
        if agent.strip() and bindings.selection_kind == "member":
            child_execution = resolve_member_execution(
                cfg,
                bindings.resolved_alias or agent_name,
                memory_mode=getattr(caller_slot, "memory_mode", "persistent"),
                validate_memory_files=False,
            )
            if caller_execution is not None:
                child_execution = child_execution.with_mode(caller_execution.memory_mode)
        elif caller_execution is not None:
            child_execution = caller_execution.with_mode(
                getattr(caller_slot, "memory_mode", "persistent")
            )
            if agent.strip():
                # An explicit template selects the child's PERSONA, never its memory:
                # the store, and the member identity that store is bound to, stay
                # the caller's, while the selection namespace becomes the template's
                # -- the same split the subagent admission gate makes for a
                # `spawn_run(agent=...)` delegate. ContextBuilder reads that
                # namespace: a member's delegate that was picked to do the work
                # itself keeps the member's identity and rules but is not handed
                # the member's operating protocol, which would send it to delegate
                # again. A member with no persisted id is named by its selection
                # alone, so the split would leave its child attributed to no member
                # and drop its [PERMANENT RULES] with its persona; that child keeps
                # the selection and takes only the template, whole desk included,
                # until the record can say "this member, under that template".
                if child_execution.member_id is None and child_execution.selection_kind == "member":
                    child_execution = replace(child_execution, template_id=bindings.kiro_agent)
                else:
                    child_execution = child_execution.with_template(bindings.kiro_agent, agent_name)
        elif bindings.selection_kind == "member":
            child_execution = resolve_member_execution(
                cfg,
                bindings.resolved_alias or agent_name,
                memory_mode=getattr(caller_slot, "memory_mode", "persistent"),
                validate_memory_files=False,
            )
        else:
            child_execution = ExecutionContext(
                None,
                MemoryStoreRef(bindings.memory_store_name or "default"),
                "template",
                bindings.kiro_agent,
                getattr(caller_slot, "memory_mode", "persistent"),
                selection_name=agent_name,
            )
        bindings = replace(
            bindings,
            execution_context=child_execution,
            memory_store_name=child_execution.store.legacy_name,
            kiro_agent=child_execution.template_id,
            selection_kind=child_execution.selection_kind,
        )
    except (ValueError, OSError):
        raise SessionControlError(
            "selected execution context is unavailable", code="memory_unavailable"
        ) from None

    # ONE authorization for the child's PRIVATE binding, at the single point where
    # every branch above has already produced its final route. A per-branch check
    # is not equivalent: the selected-member branch, the inherited-agent branch
    # and the caller-agent fallback all reach a member store, so a check on any
    # one of them leaves the others open.
    #
    # A private member store is reachable on exactly two authorities:
    #
    # * the store is the caller's OWN, read from its protected execution record --
    #   the same-store worker that is a private caller's own model. The record is
    #   the only admissible input: `slot.agent` and `slot.memory_store` are
    #   metadata a later write can change, and the SELECTION itself is a
    #   caller-supplied string, so deriving authority from either lets the request
    #   authorize itself by naming a member's agent.
    # * the caller is the owner's own dashboard session, which is what "not
    #   ownership-fenced" means. That is a shipped capability: an owner reopening
    #   member conversations and dispatching member workers. It is preserved here.
    #
    # Refused, therefore, is every population `_caller_is_ownership_fenced`
    # already treats as untrusted for the sessions it may reach: a cron slot, a
    # member's DM slot naming a PEER member's agent, and anything either of them
    # created -- the fenced caller's unfenced deputy the fence exists to catch.
    # An app-token caller never arrives (`_require_internal` refuses it) and an
    # app-scoped one cannot create at all, so the fence is the whole population.
    #
    # Fail-closed in both directions. An unbound caller has no own-store
    # admission, so it needs the unfenced one. A fence verdict is only ever read
    # as a REFUSAL here, so the mutable-record window the carried verdict exists
    # to close can widen nothing: a record that stops saying "member" between the
    # gate and this line turns a refusal into an admission the owner already has,
    # never the reverse.
    #
    # The own-store admission needs TWO sources to agree, and neither alone is
    # enough. `caller_execution` can come from the caller's own transcript record
    # when this process holds no carrier, and that record is written by the very
    # session being judged -- so on its own it answers "whose store is this?" with
    # the subject's own claim. `read_vouched_session_execution` answers with what
    # this process committed, which no session can write, but it can fall behind:
    # several modules publish an execution record without going through
    # `bind_session_execution`, so a legitimate reassignment can leave the vouched
    # entry stale.
    #
    # Requiring agreement fails closed against both. A forged record cannot match
    # a vouched entry it did not write. A stale vouched entry cannot match a record
    # that has moved on. Only an identity this process itself committed, and that
    # the record still carries, admits -- and a caller refused here is not
    # refused outright, it simply falls through to the fence below, which an owner
    # passes.
    if child_execution.member_id is not None:

        def _own_store_agreed() -> bool:
            return (
                caller_execution is not None
                and caller_execution.member_id is not None
                and caller_execution.store == child_execution.store
                and caller_vouched is not None
                and caller_vouched.store == caller_execution.store
            )

        own_store_agreed = _own_store_agreed()
        if not own_store_agreed and caller_vouched is None and caller_execution is not None:
            # The rehydrate self-heal. Agreement fails on the ONE shape
            # a restart or a cap eviction produces -- the durable record survives
            # but this process holds no vouched word -- so re-establish that word
            # at THIS gate-verified admission rather than stranding own-store
            # dispatch until the owner re-selects the agent. The trust source is
            # the VERIFIED session key: `revouch_at_verified_admission` re-vouches
            # a member DM key whose slug the durable record AGREES with, or any
            # other key whose disk vouch copy in `vouched-executions/` (written
            # when the gateway vouched it -- a member-born child) agrees with the
            # record. A caller that forged its record to name a peer's store gets
            # nothing: neither source names the peer. `caller_key` is the key the HTTP
            # gate authenticated; `caller_memory_identity[0]` is the same session's
            # history key, which carries the `member-<slug>` form for a member DM.
            # Off the loop: it resolves the member's store from config (filesystem
            # work), and `cfg` was already loaded off-loop above, so no blocking
            # read runs on the gateway loop.
            if await asyncio.to_thread(
                revouch_at_verified_admission, caller_memory_identity[0], caller_execution, cfg
            ):
                caller_vouched = read_vouched_session_execution(caller_memory_identity[0])
                own_store_agreed = _own_store_agreed()
        if own_store_agreed:
            # Recency follows USE, not birth, so a later overflow at the cap drops
            # an idle key rather than the member session still dispatching through
            # it. Grants nothing: an absent key is a no-op and the value is not
            # touched, so the only thing it changes is which entry is dropped.
            refresh_vouched_session_execution(caller_memory_identity[0])
        else:
            # The fence is read LIVE, and it reads the caller's channel link (an
            # owner-DM caller is fenced), so a caller whose identity moved during
            # the awaits above -- a link landing on it, a store or agent change --
            # must be named as such HERE, before any verdict is derived from its
            # new identity. The re-gate before allocation exists precisely to name
            # that case, and a link landing mid-resolution must surface as the
            # identity change it is, not as a delegation refusal.
            _refuse_moved_caller_identity(state, caller_key, caller_slot, caller_memory_identity)
            # Off-loop: the walk reads live slot state up the creation chain. Only
            # reached when the carried verdict is absent, and only for a private
            # selection, so an ordinary create pays nothing. Confined to this gate
            # on purpose: the owner-rooted allowance never widens the per-verb
            # ownership boundary in `authorize_target`.
            fenced = (
                caller_fenced
                if caller_fenced is not None
                else await asyncio.to_thread(_delegation_lineage_fenced, state, caller_key)
            )
            if fenced:
                # Server-side ONLY, and it says the one thing the caller's refusal
                # must not: WHICH source failed. An operator reading this can tell
                # authority this process never held or has since dropped (a restart,
                # or cap churn -- the session re-binds and recovers) from a record
                # that disagrees with what this process committed, which is the
                # forgery shape the agreement exists to refuse. Names neither the
                # store nor the member, so the log is not a second disclosure
                # channel for what the refusal withholds.
                #
                # Lineage is named FIRST and on its own, because it is the operative
                # condition for the caller class the vouched-identity causes below
                # cannot describe: a Global-store session an agent created is never
                # vouched (`bind_session_execution` vouches only a truthy
                # `member_id`), so without this every such refusal would log "no
                # vouched identity" and point an operator at restart/cap churn that
                # no re-bind will clear -- the fence here is `_created_by`, which is
                # immutable. The member and cron caller classes are fenced too but
                # are not reached as a private-member CREATE caller the way an
                # agent-created Global conductor is, so this names the condition
                # that actually reaches this line.
                #
                # A caller whose own record is on the SHARED store is named next,
                # for the same reason: it too is never vouched, so without this it
                # logs "no vouched identity" and points the operator at a re-bind
                # that cannot help -- a shared-store caller has no private store
                # of its own for the selected member's to match, so no vouch, and
                # no re-selection, ever admits it. This is the shape a Global-store
                # crew member's DM thread meets when it names a crew member that
                # owns private memory (the stock conductor dispatching a worker an
                # older agent sync enrolled with its own store); the way out is a
                # template dispatch (`agent_kind="template"`) or the owner's own tab,
                # and the log should say so instead of describing a lost vouch. The
                # store is tested beside the missing id because the id alone does
                # not place a record on the shared store: a V1 crew member bound to
                # its own named store also carries no `member_id`, and this line
                # would be false for it.
                caller_slot_now = state.get_slot(caller_key)
                if (
                    caller_fenced is None
                    and caller_slot_now is not None
                    and caller_slot_now._created_by
                    and not _member_caller(state, caller_key)
                    and not _cron_caller(caller_key)
                    and not _channel_link_of(caller_slot_now)
                ):
                    cause = "the caller is fenced by its creation lineage (_created_by)"
                elif (
                    caller_execution is not None
                    and caller_execution.member_id is None
                    and caller_execution.store.store_id == "default"
                ):
                    cause = (
                        "the caller runs on the shared store, so it holds no private "
                        "store of its own for the selected member's to match (dispatch "
                        "the template instead, or create from the owner's own tab)"
                    )
                elif caller_vouched is None:
                    cause = "this process holds no vouched identity for the caller"
                elif (
                    caller_execution is not None and caller_vouched.store != caller_execution.store
                ):
                    cause = "the caller's record disagrees with this process's vouched identity"
                else:
                    cause = "the caller's record does not name the selected store"
                logger.warning("memory delegation refused for %s: %s", caller_key, cause)
                # The delegation refusal, in the words and under the code this
                # surface already uses for it, so a caller sees one refusal for the
                # whole class. Deliberately says nothing about the store, the
                # member, or why this caller is fenced: a refusal must not confirm
                # which member owns the agent the caller guessed at.
                raise SessionControlError(
                    "cannot verify delegation within the caller's memory assignment",
                    code="memory_delegation_denied",
                    status=403,
                )

    # SlotOrigin.USER, not SYSTEM: the visibility semantics must match an
    # ordinary session, because the point of creating it here is that the user
    # can see and take over the work. SYSTEM-origin slots fall outside the
    # `slots:user` WS scope, which would hide it from the sidebar.
    #
    # A CRON caller is the exception, and it is the one case where USER would be
    # wrong rather than merely coarse. `inject_cron_result_to_dashboard` tags a
    # cron's own slot CRON precisely so its output stays out of `slots:user` ("a
    # USER label would expose it to any app holding `slots:user`"), and the trust
    # model states the same rule from the other side: inferring USER for a
    # background caller "put cron output inside `slots:user`". Minting a
    # USER-labelled child would hand a cron the exposure its own slot is denied,
    # by the simple route of creating a session and writing there instead.
    #
    # The tag therefore follows the caller's AUTHORITY, not its key prefix, and
    # for the same reason the ownership fence does: a created child INHERITS its
    # creator's agent, so a cron's child can itself call this verb, and a
    # prefix-only test mints that grandchild USER (its caller key is a plain
    # `chat-`) -- the two-hop version of the very route this comment says must be
    # denied. Reading the caller slot's own ``_origin`` closes it transitively: the
    # child carries CRON, so ITS children do too, at any depth.
    #
    # Nothing is lost by the narrower tag: only APP tokens are filtered by origin
    # (`_serialize_for_client` returns the unfiltered payload to a dashboard
    # user), so a CRON-origin descendant stays in the sidebar exactly as today's
    # cron tabs do -- which is the property the paragraph above is protecting.
    #
    # Computed from the RE-RESOLVED caller below rather than here, because it is a
    # decision input to the allocation and everything above this point was read
    # before the coroutine suspended.

    if folder_id:
        # Confirmed under the folder-store lock -- the only place existence and
        # inherited project intent cannot go stale against a concurrent delete,
        # reparent, or project edit. READ-ONLY on purpose: the Model-B un-hide is
        # a durable mutation and runs only after filing lands.
        def _current_folder_project(
            folders: list[dict[str, Any]],
        ) -> tuple[bool, str | None, str | None]:
            tree = _safe_folder_tree(folders)
            exists = any(str(folder.get("id") or "") == folder_id for folder in tree)
            raw_project, error = _folder_declared_project(tree, folder_id)
            return exists, raw_project, error

        folder_exists, current_folder_project, current_folder_error = await state.read_folders(
            _current_folder_project
        )
        if not folder_exists:
            raise SessionControlError("folder not found", code="folder_not_found")
        if current_folder_error or current_folder_project != folder_project_raw:
            raise SessionControlError(
                "folder project changed while the session was being created",
                code="folder_target_changed",
                status=409,
            )

    # Re-resolved and re-gated HERE, adjacent to the allocation, because every
    # decision above was made before this coroutine suspended -- for the
    # project directory, agent bindings, memory delegation and folder confirmation -- and the
    # inputs to those decisions are live state that can flip inside any of those
    # windows.
    #
    # Re-reading the slot TABLE is the part that matters most: closing the caller's
    # tab removes its slot, and a Python reference to the removed object stays
    # perfectly usable, so re-running the gate on the object resolved earlier would
    # authorize against a caller whose authority has already ended. Identity is
    # compared rather than mere presence, because the key can be re-minted onto a
    # different session inside the same window. `_has_channel_mirror` reads the
    # session store, so an outbound mirror link registered while this waited would
    # otherwise leave a now-channel-backed caller publishing a persistent session
    # outside its containment; `live_slot_count` reads the slot table, so two
    # concurrent creations could each pass the ceiling and then both land over it.
    #
    # Nothing suspends between this point and the fully-configured slot below, so
    # the gate and the act it authorizes stay adjacent -- the same discipline
    # `stop_target` keeps by prewarming its SEL logger ABOVE its gate rather than
    # between gate and act.
    live_caller = state.get_slot(caller_key)
    if live_caller is None or live_caller is not caller_slot:
        raise SessionControlError("caller session is not open", code="caller_not_open", status=404)
    # A slot that survived but MOVED workspaces has invalidated both decisions that
    # read it: the memory boundary the child inherits, and the agent-binding check
    # above, whose whole question was whether the answering agent is bound to THIS
    # workspace. Re-running that check here is not an option -- it needs
    # `KiroCrewConfig.load()`, which is filesystem work that must not run on the
    # event loop -- so a moved caller is refused instead of re-authorized.
    if (getattr(live_caller, "workspace", "default") or "default") != workspace:
        raise SessionControlError(
            "caller session changed workspace while the session was being created",
            code="caller_workspace_changed",
        )
    _refuse_moved_caller_identity(state, caller_key, caller_slot, caller_memory_identity)
    _refuse_ineligible_creator(state, live_caller)
    # The child's origin tag, read off the caller that is live NOW -- see the
    # reasoning above the folder gate. `_cron_caller` covers a cron's own tab;
    # `_origin` carries the tag onward to every descendant of one.
    child_origin = (
        SlotOrigin.CRON
        if _cron_caller(caller_key) or getattr(live_caller, "_origin", "") == SlotOrigin.CRON
        else SlotOrigin.USER
    )
    # The RATE guard, ahead of the capacity ceilings below. Those bound how many
    # sessions can exist; this bounds how fast one caller may open them, which is
    # the property an auto-approved verb loses -- a waived prompt leaves a loop
    # nothing to push back on. Deliberately the control that needs no durable
    # state: a lifetime quota means nothing across a restart unless every
    # rehydrate path carries its attribution, while a five-minute window buys a
    # restart one window rather than a clean slate.
    #
    # A dry run asks the same question without spending the token: a preview
    # must not use up the create it previews.
    admitted = (
        has_create_budget(SESSION_CREATE, caller_key)
        if dry_run
        else allow_create(SESSION_CREATE, caller_key)
    )
    if not admitted:
        raise SessionControlError(
            "too many sessions created recently; retry shortly",
            code="create_rate_limited",
            status=429,
        )
    if state.live_slot_count() >= MAX_LIVE_SLOTS:
        raise SessionControlError(
            f"slot cap reached ({MAX_LIVE_SLOTS})",
            code="slot_cap_reached",
            status=429,
        )
    # Then the per-creator sub-ceiling. The global cap above bounds the TOTAL but
    # not the distribution, so without this one caller can hold all 500 and the
    # person's own next chat tab gets the 429 -- the resource is bounded, but not
    # from anyone else's point of view. This is the bound that makes the verb safe
    # to auto-approve: the worst case of an automated creator looping on it is its
    # own 50 slots, not everyone's 500.
    if state.creator_slot_count(caller_key) >= MAX_SLOTS_PER_CREATOR:
        raise SessionControlError(
            f"per-caller slot cap reached ({MAX_SLOTS_PER_CREATOR})",
            code="creator_slot_cap_reached",
            status=429,
        )
    # The dry run ends HERE, after the last refusal and before the first write.
    # Any new refusal gate belongs above this line, or the preview would pass a
    # create the real call then refuses, and the MCP path walk would leave the
    # folders it made for that create empty.
    if dry_run:
        return {"dry_run": True}

    # The agent rides in the constructor rather than being assigned afterwards, for
    # the same reason: it decides which workspace actually EXECUTES the turn, so it
    # must never be observable as empty. Everything after this point is synchronous
    # until the slot is fully configured.
    #
    # The whole allocation-to-persist span runs under `suspend_slots_push`:
    # `get_or_create_slot` broadcasts on a leading edge, so without the suspend an
    # idle gateway serializes and sends the new slot BEFORE `folder_id` is
    # assigned -- every client (and any app on `slots:user`) would render the
    # session at the top level for a frame -- the observable unfiled state this
    # suspend removes. It also covers the persist and its failure retraction, so
    # a slot whose birth write fails is never broadcast at all. Same pattern the
    # move path uses ("file the slot before the coalesced broadcast").
    with state.suspend_slots_push():
        slot = state.get_or_create_slot(
            None, agent=agent_name, workspace=workspace, origin=child_origin
        )
        # Attribute the slot to the caller that asked for it, which is what makes the
        # per-creator ceiling above countable. Written here, inside the synchronous
        # window that follows the mint, so no suspension point separates the cap test
        # from this write -- otherwise two concurrent creates could both pass a ceiling
        # that one of them had already filled. Only this entry point sets it: a
        # person's own tab and a fork reach `get_or_create_slot` directly and stay
        # unattributed, so ordinary human use never consumes an automated caller's
        # share.
        slot._created_by = caller_key
        # Freeze the creator's ACP session id HERE, at mint, from the live caller
        # handle we just authorized -- not later at the child's first turn. The
        # creator slot can be closed and replaced between this mint and that turn,
        # and a replacement is a distinct handle with its own session id; reading
        # the id live at emit would then cite the replacement's crew log and corrupt
        # the child's immutable `session/opened` lineage with no recovery path.
        # `live_caller` is the same object the authorization gate above resolved,
        # so this is the id that was live when the child was made. Empty when the
        # caller's handle has no ACP session yet, which is recorded as absent.
        # Bounded HERE, at retention, by the one constant every store of a
        # backend-authored session id shares: an id past it is dropped, not
        # truncated, so an oversize backend id can neither grow the slot's
        # metadata nor make the child's ``session/opened`` entry too large to
        # land -- the sid is optional, its absence is a legal record.
        _creator_sid = crew_log_emit.session_id_of(getattr(live_caller, "_acp_client", None))
        slot._created_by_sid = _creator_sid if len(_creator_sid) <= MAX_ACP_SESSION_ID_LEN else ""
        # Witness that THIS process stamped the two fields above at mint. Neither
        # the flag nor the sid is persisted: the transcript is a file an agent's
        # file tools can edit, and the crew log is fenced from those tools exactly
        # so nothing in it can be forged as gateway-authored -- so the child's
        # first turn writes `session/opened.parent` only when this flag is set,
        # never from `created_by` read back off disk. A restart between mint and
        # the child's first turn therefore loses the link rather than trusting
        # metadata for it.
        slot._lineage_minted = True
        # The creator's interactive auto-approve grant follows the work it is
        # handing off. Without this a trusted operator dispatches a worker that
        # then blocks on an approval prompt nobody is watching -- the same failure
        # `parent_trusted` already closes for `spawn_run` subagents, which read the
        # parent's stored policy and start auto-approved. A dispatched session is
        # the same delegation with a sidebar tab, so it takes the same posture.
        #
        # Read off `live_caller`, not the entry-time `caller_slot`: the two are
        # identity-checked to be the same object above, but the grant itself is
        # mutable state the operator can revoke inside any of the suspensions this
        # coroutine took, so the value that transfers is the one held NOW, in the
        # synchronous window that follows the last gate. Revoking before the create
        # lands means the child is born untrusted, which is the direction that
        # fails safe.
        #
        # What transfers is SESSION POSTURE, and only that. Two fields carry, two
        # deliberately do not, and the exclusions are the load-bearing part:
        #
        # * `_trust` -- the human's "trust this session" click. It does not expire,
        #   the click is its own audit record, and copying it changes no property
        #   of the grant. The session-store half needs no write here: the child has
        #   no ACP session yet (`set_approval_policy` would silently no-op on a
        #   missing session), and `chat_runner` already assigns the persistable
        #   policy from `_trust` on every session create/resume, so the subagent
        #   spawn gate sees it from the child's first turn.
        # * `_trust_reads` -- the same posture, narrowed to read-only bash. It has
        #   to carry too, or the setting a CAUTIOUS operator picks is the one whose
        #   own workers still stall. Bounded by construction: what it admits has no
        #   side effects, which is what separates it from the command grants below.
        #
        # * `_trusted_patterns` -- NOT inherited. These are per-command grants
        #   ("`npm test` is fine"), not a posture, and the distinction decides it:
        #   a pattern is judged against the session the operator was LOOKING at,
        #   while a dispatched worker runs model-authored work they have not seen,
        #   so the same glob can admit a command the grant was never asked about.
        #   Inheriting them also buys nothing where it would be safe -- with
        #   `_trust` set the child already auto-approves via `_slot_is_trusted`, so
        #   the pattern list is dead weight; it changes the outcome ONLY when the
        #   operator withheld session trust and granted single commands instead,
        #   which is exactly the case that must keep asking. So the child starts
        #   with `_ChatSlot.__init__`'s empty set and earns its own grants.
        # * `_trust_scope` -- NOT inherited. It names a TTL-bounded, SEL-audited
        #   `SafetyOverride` grant that is re-checked on every approval; forking
        #   the key would hand a second session a credential whose revocation
        #   nothing here can observe. An unattended worker that needs one gets its
        #   own, armed by whatever owns its lifecycle.
        #
        # Not persisted at birth, matching every other slot: trust is in-memory by
        # construction, so a restart returns the child to interactive along with
        # its creator.
        inherited_trust = bool(getattr(live_caller, "_trust", False))
        inherited_trust_reads = bool(getattr(live_caller, "_trust_reads", False))
        slot._trust = inherited_trust
        slot._trust_reads = inherited_trust_reads
        # The agent's memory silo, from the bindings already resolved above. Held
        # on the slot so every later save can name it: `memory_store` is
        # slot-owned metadata, so a save that could not read it would drop the
        # key and silently return this session to the global store.
        slot.memory_store = bindings.memory_store_name
        slot.memory_mode = child_execution.memory_mode
        # The namespace the pick was committed in, recorded only when the caller
        # stated one -- the chat create route's rule ("a name-only pick writes no
        # agent_kind; only an explicit template pick says this was never a
        # member session"). The resolver's verdict rather than the request's word,
        # so the slot records what RESOLVED. Without it a rehydrated template child
        # of a shadowed name would read as a bare name and re-resolve member-first.
        if committed_kind:
            slot.agent_kind = committed_kind
        # cwd must follow the workspace too, or file search and project-scoped agents
        # resolve against a directory the slot does not claim -- the same
        # authorization-vs-execution split as the agent binding, one layer down.
        if not slot.project:
            slot.project = project_dir
        if folder_id:
            # Filed inside the same synchronous window that configures the slot, so
            # the session is never observable unfiled -- that atomicity is the point.
            # Existence was confirmed under the store lock above, and folder
            # mutations run on this loop, so the folder cannot have been deleted
            # between that check and this assignment. No `_folder_changed` flag: the
            # slot's first turn carries the armed first-turn breadcrumb injection
            # (`is_new` in chat_runner), so the [FOLDER] line reaches the model
            # without it.
            slot.folder_id = folder_id
        if model_name:
            # Pinned the way a person's pick in the model dropdown pins it: the
            # slot has no provider session yet, so there is nothing to switch --
            # the first turn starts on this model. The pick-generation bump marks
            # it as an explicit choice, so the fallback restore probe treats it
            # exactly as it treats a human pick rather than as a backfilled value.
            slot.model = model_name
            slot._model_pick_gen += 1
        if title.strip():
            slot.title = sanitize_outbound(title.strip())[:200]
            slot._titled = True
        # Persist at birth. `save_slot_off_loop` cannot do this: the save it wraps
        # returns early on an empty message window -- a full save has nothing to
        # write -- so a freshly created session, which has no messages by
        # definition, would write nothing at all. The tool would then hand back a
        # session that does not survive a restart.
        #
        # Awaited, and a failure RETRACTS the slot rather than merely propagating: an
        # unpersisted slot stays in the table, usable in memory and addressable by its
        # creator, then vanishes on restart. Reporting the failure while leaving that
        # behind is the worse of the two outcomes, because the caller sees an error and
        # the session exists anyway. Same retraction the fork path uses on a failed
        # build.
        session_key = slot_history_key(slot)
        native_context = state.sessions.get_provider(session_key) is not None or bool(
            state.sessions.resumable_sid(session_key)
        )
        birth_persisted = False

        def _persist_birth(metadata: dict[str, Any]) -> None:
            nonlocal birth_persisted
            if read_session_execution(caller_memory_identity[0]) != caller_execution:
                raise SessionControlError(
                    "caller execution context changed during creation",
                    code="caller_memory_changed",
                )
            if (
                child_execution.member_id is not None
                and read_session_execution(session_key) is None
            ):
                if native_context or log.has_messages(session_key):
                    from kiro_crew.memory_stores import UnknownMemoryStore

                    raise UnknownMemoryStore("Existing session context requires a new conversation")
            # Establishing: this child's store came from the authorization above,
            # not from any record the child or its caller can write, so this is one
            # of the few publications entitled to vouch.
            bind_session_execution(session_key, child_execution, vouch=True)
            log.update_metadata(session_key, metadata)
            birth_persisted = True

        try:
            await drained_to_thread(
                _persist_birth,
                {
                    "_type": "metadata",
                    # The slot's OWN durable identity, and its origin, both of which
                    # the normal save path writes -- but a slot created here may never
                    # reach that path: `_save_slot_to_history` runs a full save only
                    # when the window has messages, so for a session that is created
                    # and then sits idle THIS dict is the only record on disk.
                    # Omitting `origin` is silently destructive on the next restart:
                    # rehydrate falls back to the fail-closed empty sentinel, so a
                    # session opened as USER comes back unattributed and `slots:user`
                    # subscribers stop seeing it. Checked field-by-field against the
                    # save path; these are the only fields a slot carries at birth
                    # that it does not already write.
                    "tab_id": slot._tab_id,
                    "origin": slot._origin,
                    "created_at": metadata_now_iso(),
                    "workspace": slot.workspace,
                    "agent": slot.agent or "",
                    "project": slot.project or "",
                    "title": slot.title or "",
                    "memory_mode": getattr(slot, "memory_mode", "persistent"),
                    # The committed selection namespace, only when the caller
                    # stated one -- the metadata codec writes `agent_kind` the same
                    # way (truthy only), and for an idle newborn this dict is the
                    # only record, so without it a restart would rehydrate a
                    # template child of a shadowed name as a bare name.
                    **({"agent_kind": slot.agent_kind} if slot.agent_kind else {}),
                    # Only when filed, mirroring the normal save path, which omits
                    # `folder_id` from the metadata line when empty. Without this
                    # the filing would not survive a restart: for an idle newborn
                    # THIS dict is the only record of the placement on disk.
                    **({"folder_id": slot.folder_id} if slot.folder_id else {}),
                    # The pinned model, only when one was asked for -- the normal
                    # save path writes `model` too, but for an idle newborn this
                    # dict is the only record, and without it a restart would
                    # bring the session back on the default model.
                    **({"model": slot.model} if model_name else {}),
                    # Creator attribution, only when this entry point set it. The
                    # member ownership boundary in `authorize_target` reads it, so
                    # losing it on restart would strand every worker a member
                    # dispatched — controllable in memory, orphaned after reboot.
                    **({"created_by": slot._created_by} if slot._created_by else {}),
                    # `created_by_sid` is deliberately NOT written: the transcript
                    # is agent-editable, so nothing read back from it may become
                    # crew-log lineage. The sid lives on the slot for this process
                    # only (see `_lineage_minted`).
                    # The agent's memory silo, recorded ONLY when it is not the
                    # default. This is what lets the consolidator write an agent's
                    # semantic, episodic and lesson rows into its own store
                    # instead of the global one, and this dict is the only record
                    # for a session that is created and then sits idle.
                    #
                    # Omitted for the default store on purpose: absence is the
                    # signal for "global", so a default user's metadata line stays
                    # byte-identical and a session written before crews had stores
                    # reads the same as one written now.
                    **(
                        {"memory_store": _named_store}
                        if (_named_store := named_store_or_empty(slot.memory_store))
                        else {}
                    ),
                },
            )
        except (Exception, asyncio.CancelledError):
            # Retract, but never at the cost of work already in flight. The slot is
            # addressable from the moment `get_or_create_slot` publishes it, which is
            # before this await, so a turn can have started on it while the write was
            # in the worker thread. Popping the slot then would leave that turn running
            # with nothing pointing at it -- unreachable, unstoppable, and invisible to
            # the stop verb. A phantom session that vanishes on the next restart is the
            # lesser harm, so liveness wins over tidiness and the slot stays.
            # Drain before deciding: cancellation cannot leave a worker writing
            # identity/history after this handler retracts its slot. A completed
            # birth survives cancellation just as a turn already in flight does.
            # A completed identity record remains available to a concurrent turn.
            if (
                not birth_persisted
                and not slot.running
                and not slot.messages
                and state._slots.get(slot.key) is slot
            ):
                state._slots.pop(slot.key, None)
            state.push_slots_update()
            raise
        if slot.folder_id:
            # Model-B un-hide, applied only NOW that the filing has actually
            # landed -- running it any earlier persists `hidden = False` for a
            # create a later gate can still refuse, durably reversing a choice
            # the user made for a call that failed. The move path holds the same
            # order (assign, confirm, then un-hide). If the folder was deleted
            # while the persist was in the worker thread, the delete's own sweep
            # already unfiled this slot (it is published), so the guard reads
            # the fresh value and skips; the metadata line can then briefly
            # carry a dangling folder_id, the same accepted residual a move
            # racing a delete leaves, and readers fall back to "(unfiled)".
            #
            # Best-effort: the create is already COMMITTED (slot published,
            # persisted at birth), so a folder-store write failure here must not
            # propagate -- the request would report failure for a session that
            # exists, and the caller's retry would create a duplicate. A folder
            # left hidden with a session inside is the recoverable lesser harm.
            try:
                await _unhide_folder(state, slot.folder_id)
            except Exception:
                logger.warning(
                    "create_session: filing committed for %s but un-hiding folder %s failed",
                    slot.key,
                    slot.folder_id,
                    exc_info=True,
                )
        state.push_slots_update()
    lineage_recorded = await _record_spawn(caller_key, slot)
    _audit(
        caller_session_key=caller_key,
        operation="create",
        slot_key=slot.key,
        outcome="allowed",
        detail={
            "agent": slot.agent or "",
            "folder_id": slot.folder_id or "",
            "model": model_name,
            # What the child was BORN with, so an auto-approved tool call in it is
            # traceable to the creator's grant rather than appearing unexplained.
            # Always present: "false" is the record that the grant did not transfer.
            "inherited_trust": "true" if inherited_trust else "false",
            "inherited_trust_reads": "true" if inherited_trust_reads else "false",
            # Whether the creating edge reached the creator's crew log. "false" means
            # the child nests from this process's witness alone and would float at top
            # level after a restart. The audit log is this fact's durable home -- the
            # API reply has no reader for it (and _record_spawn also logs a warning).
            "lineage_recorded": "true" if lineage_recorded else "false",
        },
    )
    return {
        "ok": True,
        "target": slot.key,
        "title": slot.title or slot.key,
        **({"model": model_name} if model_name else {}),
    }


#: Maximum ``title`` a forked child accepts, matching ``create_session``'s cap.
_MAX_FORK_TITLE_CHARS = 200


def _fork_refusal(response: Any) -> SessionControlError:
    """Translate a ``chat_fork`` refusal into this module's error type.

    ``chat_fork`` refuses with a finished ``web.json_response`` -- its coded
    ``{"error", "code"}`` body IS the refusal, and its sites stay there so the
    error-code ratchet keeps pinning them. This surface speaks
    :class:`SessionControlError`, so the body is read back out rather than the
    fork core learning a second refusal type. A body that does not decode is
    still a refusal, just an unlabelled one.
    """
    error, code = "fork refused", "fork_refused"
    try:
        body = json.loads(response.body or b"{}")
        error = str(body.get("error") or error)
        code = str(body.get("code") or code)
    except (ValueError, AttributeError, TypeError):
        pass
    return SessionControlError(error, status=int(getattr(response, "status", 400)), code=code)


async def fork_session(
    state: "DashboardState",
    *,
    caller_session_key: str,
    source: str = "",
    title: str = "",
    folder_id: str = "",
    at_message_index: int | None = None,
    caller_fenced: bool | None = None,
) -> dict[str, Any]:
    """Open a new session that CARRIES a transcript: the dashboard's Fork, for an agent.

    ``create_session`` opens an empty session; this opens one holding a copy of
    *source*'s messages up to and including ``at_message_index`` (the whole
    visible transcript when omitted -- a head fork; tail forks are not offered
    here). The copy is made by the same core the human Fork button runs
    (``chat_fork.fork_slot``), so what the child inherits -- agent, model,
    memory store and mode, project, folder, tags, the ``forked_from`` link --
    is exactly what a person's fork inherits, and for the same memory-boundary
    reasons no override of agent, model or mode is taken here. The one exception
    is the sidebar pin: a person's fork of a pinned session is pinned, this
    child never is, because pinned marks a session as human-owned.

    *source* defaults to the CALLER'S OWN session, which is the case this verb
    exists for: an agent splitting its own long investigation into several
    sessions that each start with the context it already built. Naming another
    session is a READ of that session's transcript, so it is authorized exactly
    as ``read_messages`` is (``authorize_target``, operation ``fork``): the
    caller must be allowed to read the source. Either way the caller must also
    be an eligible CREATOR -- the same refusal set ``create_session`` applies,
    because a fork manufactures a session the caller then owns.

    What the child gets on top of the human fork: ``title`` (final; else the
    fork's own ``Fork of <parent>``, which the fork's background title pass may
    rename), ``folder_id`` (else the parent's folder, as the human
    fork inherits it), creator attribution (``created_by`` = the caller, so the
    other verbs reach it afterwards and the per-creator ceiling counts it) and the
    caller's session posture (``_trust`` / ``_trust_reads`` -- the same two
    fields ``create_session`` carries, with the same exclusions). It starts IDLE:
    the copied transcript is history, and nothing runs until the caller
    ``session_send``\\ s into it or the person types.

    ``caller_fenced`` is the HTTP gate's ownership-fence verdict, forwarded to
    ``authorize_target`` for the reason every other verb forwards it.
    """
    caller_key = caller_slot_key(state, caller_session_key)
    if not caller_key:
        raise SessionControlError(
            "caller session could not be identified", code="caller_unidentified"
        )
    # Same gate order as `create_session`, for the same reasons: the member
    # bypass is resolved on the caller, then the config switch, then the
    # unattended prefix.
    if not session_control_enabled() and not _member_bypass(state, caller_key):
        raise SessionControlError(
            "session control is disabled in config (agent.session_control)",
            code="session_control_disabled",
        )
    if caller_key.startswith(UNATTENDED_SLOT_PREFIXES) and not _cron_caller(caller_key):
        raise SessionControlError(
            "unattended sessions (scheduled runs) cannot fork sessions",
            code="unattended_caller",
        )
    caller_slot = state.get_slot(caller_key)
    if caller_slot is None:
        raise SessionControlError("caller session is not open", code="caller_not_open", status=404)
    # A fork manufactures a session the caller owns, so the caller must be an
    # eligible creator before anything else is read -- see the note on
    # `_refuse_ineligible_creator` for why this set mirrors `authorize_target`'s.
    _refuse_ineligible_creator(state, caller_slot)

    if at_message_index is not None and (
        isinstance(at_message_index, bool) or at_message_index < 0
    ):
        raise SessionControlError(
            "at_message_index must be a non-negative integer", code="invalid_field_type"
        )

    # Resolve the source. An empty `source` is the caller itself. A named source
    # that resolves to the caller is the same case spelled out -- `authorize_target`
    # would refuse it as `self_target`, and rightly so for stop/send/read, but a
    # session reading its OWN transcript to copy it crosses no boundary. Anything
    # else is a peer, and copying a peer's transcript is a read of it, so it is
    # authorized as `read_messages` is: same verb-level requirement, same fence.
    source_ref = (source or "").strip()
    if source_ref:
        try:
            resolved = _resolve_slot(state, source_ref)
        except SessionControlError:
            resolved = None
    else:
        resolved = caller_slot
    if resolved is caller_slot:
        source_slot = caller_slot
    else:
        source_slot = authorize_target(
            state,
            caller_session_key=caller_session_key,
            target=source_ref,
            operation="fork",
            precomputed_ownership_fenced=caller_fenced,
        )

    log = state.conversation_log
    if log is None:
        # Same answer `create_session` gives: without a durable store the copy
        # cannot be persisted, and a fork that vanishes on restart is not a fork.
        raise SessionControlError(
            "session history is unavailable, so the session cannot be persisted",
            code="history_unavailable",
        )

    if folder_id:
        # Confirmed READ-ONLY under the folder-store lock, exactly as
        # `create_session` does and for the same reasons; the Model-B un-hide runs
        # only once the filing has landed on the child.
        def _exists(folders: list[dict[str, Any]]) -> bool:
            return any(str(f.get("id") or "") == folder_id for f in _safe_folder_tree(folders))

        if not await state.read_folders(_exists):
            raise SessionControlError("folder not found", code="folder_not_found")

    # Re-gate adjacent to the allocation, as `create_session` does: every input
    # above was read before this coroutine suspended (the folder confirmation),
    # and both the caller's eligibility and the source's liveness are live state.
    live_caller = state.get_slot(caller_key)
    if live_caller is None or live_caller is not caller_slot:
        raise SessionControlError("caller session is not open", code="caller_not_open", status=404)
    _refuse_ineligible_creator(state, live_caller)
    if state.get_slot(source_slot.key) is not source_slot:
        raise SessionControlError(
            "the source session closed while the fork was being prepared",
            code="target_not_found",
            status=404,
        )
    child_origin = (
        SlotOrigin.CRON
        if _cron_caller(caller_key) or getattr(live_caller, "_origin", "") == SlotOrigin.CRON
        else SlotOrigin.USER
    )
    # A fork spends the same budget and counts against the same ceilings as a
    # create: it is a session the caller manufactured, whatever it starts with.
    if not allow_create(SESSION_CREATE, caller_key):
        raise SessionControlError(
            "too many sessions created recently; retry shortly",
            code="create_rate_limited",
            status=429,
        )
    if state.live_slot_count() >= MAX_LIVE_SLOTS:
        raise SessionControlError(
            f"slot cap reached ({MAX_LIVE_SLOTS})",
            code="slot_cap_reached",
            status=429,
        )
    if state.creator_slot_count(caller_key) >= MAX_SLOTS_PER_CREATOR:
        raise SessionControlError(
            f"per-caller slot cap reached ({MAX_SLOTS_PER_CREATOR})",
            code="creator_slot_cap_reached",
            status=429,
        )

    audit_caller = f"session:{caller_key}"
    # The copy below reads the source's rows: a source restored as a sidebar
    # row loads its window first, off the loop. The source is re-admitted after
    # these awaits, as it already is for the ones that follow.
    await load_window_off_loop(source_slot)
    fork_source = await resolve_fork_source(
        source_slot, audit_caller=audit_caller, audit_operation="session_control.fork"
    )
    if not isinstance(fork_source, ForkSource):
        raise _fork_refusal(fork_source)

    # The session-control half of the child's identity, mirrored from
    # `create_session`. Applied INSIDE `fork_slot`, on the child, before its
    # birth save: `save_slot_off_loop` writes `created_by`, `title` and
    # `folder_id` into the metadata line it creates, so attribution is on disk in
    # the same write as the transcript and before the slot is broadcast. There is
    # no second persistence window -- a child that exists is an attributed child,
    # and a save that fails withdraws the whole child (`fork_slot` pops it), so a
    # retry cannot leave an unreachable duplicate behind.
    _creator_sid = crew_log_emit.session_id_of(getattr(live_caller, "_acp_client", None))
    # Session POSTURE only -- `_trust` and `_trust_reads` -- never
    # `_trusted_patterns` or `_trust_scope`; `create_session` states why. Read
    # INSIDE `_stamp`, not here: `fork_slot` suspends for the transcript read and
    # the memory bind, and an operator revoking the caller's trust in that window
    # (a per-slot revoke cannot reach a child that does not exist yet) must not
    # see the child born with the grant they just withdrew. The values are
    # recorded for the audit line after the stamp has taken them.
    posture: dict[str, bool] = {}
    clean_title = sanitize_outbound(title.strip())[:_MAX_FORK_TITLE_CHARS] if title.strip() else ""

    def _folder_exists_now() -> bool:
        # The COMMITTED folder list, read synchronously: folder mutations run on
        # this loop, so between this read and the assignment `_stamp` makes there
        # is no point at which a delete can land. Same value `read_folders` hands
        # its reader; the lock there exists for readers that hop off the loop.
        return any(str(f.get("id") or "") == folder_id for f in _safe_folder_tree(state._folders))

    def _recheck() -> None:
        # Every containment answer above was read before `fork_slot` suspended
        # (transcript read, memory bind). Re-asserted synchronously at the two
        # points that matter -- the mint and the copy -- so a caller that lost
        # eligibility, a source that gained a channel mirror or moved out of
        # reach, or a folder deleted meanwhile, refuses the fork instead of being
        # copied around. Mirrors the "gate adjacent to the act" discipline
        # `create_session` keeps for its own allocation.
        live = state.get_slot(caller_key)
        if live is None or live is not caller_slot:
            raise SessionControlError(
                "caller session is not open", code="caller_not_open", status=404
            )
        _refuse_ineligible_creator(state, live)
        if state.get_slot(source_slot.key) is not source_slot:
            raise SessionControlError(
                "the source session closed while the fork was being prepared",
                code="target_not_found",
                status=404,
            )
        if source_slot is not caller_slot:
            readmitted = authorize_target(
                state,
                caller_session_key=caller_session_key,
                target=source_ref,
                operation="fork",
                precomputed_ownership_fenced=caller_fenced,
                skip_enabled_check=True,
            )
            if readmitted is not source_slot:
                raise SessionControlError(
                    "the source session changed while the fork was being prepared",
                    code="target_not_found",
                    status=404,
                )
        if folder_id and not _folder_exists_now():
            raise SessionControlError("folder not found", code="folder_not_found")
        # The ceilings, re-read here rather than only at entry: two forks in
        # flight could each pass the entry check and both suspend before their
        # mints. The rate budget above is consumed atomically and bounds the
        # burst; these keep the count itself honest at the mint, which is the
        # same synchronous gate-then-act window `create_session` holds.
        if state.live_slot_count() >= MAX_LIVE_SLOTS:
            raise SessionControlError(
                f"slot cap reached ({MAX_LIVE_SLOTS})", code="slot_cap_reached", status=429
            )
        if state.creator_slot_count(caller_key) >= MAX_SLOTS_PER_CREATOR:
            raise SessionControlError(
                f"per-caller slot cap reached ({MAX_SLOTS_PER_CREATOR})",
                code="creator_slot_cap_reached",
                status=429,
            )

    def _stamp(child: Any) -> None:
        child._created_by = caller_key
        child._created_by_sid = _creator_sid if len(_creator_sid) <= MAX_ACP_SESSION_ID_LEN else ""
        child._lineage_minted = True
        posture["trust"] = bool(getattr(live_caller, "_trust", False))
        posture["trust_reads"] = bool(getattr(live_caller, "_trust_reads", False))
        child._trust = posture["trust"]
        child._trust_reads = posture["trust_reads"]
        if clean_title:
            child.title = clean_title
            child._titled = True
            # The caller chose this name, so it is final: the fork's own
            # background title pass and the refresh both leave it alone.
            child._title_origin = "user"
        if folder_id:
            child.folder_id = folder_id

    result = await fork_slot(
        state,
        fork_source,
        at_index=at_message_index,
        at_message_id=None,
        direction=_FORK_DIRECTION_HEAD,
        prompt="",
        mode_override=None,
        # An agent-made child: not app-scoped, not a human request-layer session
        # for the session-count survey, and never Jev-routed -- arming a second
        # routed session is the owner's own click, which no caller here made.
        request_app="",
        origin=child_origin,
        count_user_session=False,
        jev_route_allowed=False,
        audit_caller=audit_caller,
        audit_operation="session_control.fork",
        stamp=_stamp,
        recheck=_recheck,
    )
    if not isinstance(result, ForkResult):
        raise _fork_refusal(result)
    child = result.slot

    if folder_id:
        try:
            await _unhide_folder(state, folder_id)
        except Exception:
            logger.warning(
                "fork_session: filing committed for %s but un-hiding folder %s failed",
                child.key,
                folder_id,
                exc_info=True,
            )
    state.push_slots_update()
    lineage_recorded = await _record_spawn(caller_key, child)
    _audit(
        caller_session_key=caller_key,
        operation="fork",
        slot_key=child.key,
        outcome="allowed",
        detail={
            "source": source_slot.key,
            "messages": str(result.messages),
            "folder_id": child.folder_id or "",
            "inherited_trust": "true" if posture.get("trust") else "false",
            "inherited_trust_reads": "true" if posture.get("trust_reads") else "false",
            # See create_session: whether the creating edge landed is kept in the audit
            # log, its durable home, rather than the API reply, which has no reader.
            "lineage_recorded": "true" if lineage_recorded else "false",
        },
    )
    return {
        "ok": True,
        "target": child.key,
        "title": child.title or child.key,
        "source": source_slot.key,
        "messages": result.messages,
        "folder_id": child.folder_id or None,
    }


async def _record_spawn(caller_key: str, child: Any) -> bool:
    """Write the child's creating edge to the CREATOR's crew log (``session/spawned``)
    and wait, bounded, for it to land. Returns whether the edge is on disk.

    The mint witness (``_lineage_minted``) lives only in this process, and the child's
    own ``session/opened.parent`` is not written until its first turn, so neither
    survives a restart in between. The creator's log already exists, so the edge is
    recorded there, and the caller is told the create succeeded only after this has
    settled: a reply sent while the entry is still queued would be lost with it on a
    crash, and the child would restore at the top level.

    Taken from the values stamped at mint, and only when this process stamped them:
    the creator sid frozen at mint names the log that was live when the child was
    made. ``True`` with nothing written when there is no edge to record -- no witness,
    or a creator with no ACP session yet, which has no log and keeps the in-memory
    witness alone. Never raises: the create has already committed, so a failure here
    is reported as ``False`` rather than as a failed create, which a caller would
    retry into a duplicate session.

    The write is queued before the wait, so a cancellation of the wait does not
    withdraw it: the entry still lands with the birth it belongs to.
    """
    if not bool(getattr(child, "_lineage_minted", False)):
        return True
    creator_slot = str(getattr(child, "_created_by", "") or "")
    creator_sid = str(getattr(child, "_created_by_sid", "") or "")
    child_slot = str(getattr(child, "key", "") or "")
    if not creator_sid or not creator_slot or creator_slot != caller_key:
        return True
    try:
        if not crew_log_emit.enabled():
            return True
        landed = await crew_log_emit.awaiting_commit(
            lambda settled: crew_log_emit.on_session_spawned(
                creator_sid,
                creator_slot=creator_slot,
                child_slot=child_slot,
                on_settled=settled,
            ),
            what="session/spawned",
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("session_control: could not record session/spawned", exc_info=True)
        return False
    if not landed:
        logger.warning(
            "session_control: the creating edge for %s did not land in its creator's crew "
            "log; it nests under %s only until this gateway restarts",
            child_slot,
            creator_slot,
        )
    return landed


#: The refusal factory's product: it RETURNS the error to raise rather than raising,
#: so the audit write happens exactly once at the point of refusal and a caller
#: collecting per-target outcomes can record one without aborting its own loop.
Deny = Callable[..., SessionControlError]


def _deny_factory(*, caller_session_key: str, operation: str, target: str) -> Deny:
    """Build the ``deny`` used by every gate in this module.

    Lifted out of :func:`authorize_target` so the verbs that have NO target
    (:func:`created_session_status`) refuse through the same audited path rather
    than growing a second one. ``target`` is the audit's ``resources`` field and is
    empty for a targetless verb, which reads in the trail as "this refusal was not
    about a particular session".
    """

    def deny(reason: str, code: str, status: int = 403) -> SessionControlError:
        # Off the loop for the same reason `_audit` is: this can be the process's
        # FIRST `sel()`, which constructs the log. A denial is the likeliest
        # first-ever session-control call on a fresh gateway -- the feature refuses
        # before it ever allows -- so this path is not the rare one.
        #
        # Redacted BEFORE the write, because the audit sink is durable and served
        # back: `sel.py` documents that on-disk records are not redacted by the
        # writer, and `/api/sel/events` returns `recent()` rows verbatim to the
        # dashboard. `target` is raw MCP input, and `target_not_found` interpolates
        # it into `reason`, so both carry caller text. Redacting at this chokepoint
        # rather than at the one interpolating call site keeps a future `deny`
        # caller from reopening it. `redact` is what `sel._forward_event` already
        # applies to events on the forward path; this closes the same gap on the
        # path the dashboard reads.
        #
        # Only the audit copy is redacted: the returned message goes to the caller
        # that supplied the string, so it keeps naming the target it was given.
        _audit_target = redact(target)
        _audit_reason = redact(reason)
        _sel_off_loop(
            lambda: sel().log_api_access(
                caller=f"session:{caller_session_key or 'unknown'}",
                operation=f"session_control.{operation}",
                outcome="denied",
                source="mcp",
                resources=f"target={_audit_target}:{code}",
                error=_audit_reason,
            ),
            "session-control denial audit",
        )
        return SessionControlError(reason, status=status, code=code)

    return deny


def refuse_caller_identity(
    state: "DashboardState",
    *,
    caller_session_key: str,
    deny: Deny,
    skip_enabled_check: bool = False,
) -> str:
    """The caller-side gate that runs BEFORE any target is resolved.

    Returns the caller's slot key. Extracted from :func:`authorize_target` so the
    targetless verbs share ONE copy of these three refusals rather than a second
    sequence that can drift from this one; every check and every code is the same
    text it was inline, and :func:`authorize_target` still calls it at the same
    point in its own order, so no precedence the suite pins has moved.

    Why these three are on this side of the resolution is the existence-oracle
    argument the ``_app_owned_cron_refusal`` comment gives: a caller refused for
    its OWN identity must learn nothing from the attempt, and a gate that resolves
    first answers ``target_not_found`` (404) for a session that does not exist and
    a 403 for one that does.
    """
    caller_key = caller_slot_key(state, caller_session_key)
    if not caller_key:
        # Without a resolved caller the self-target guard is blind, and a session
        # that can reach every peer while being unidentifiable is exactly the
        # shape this surface must not have.
        raise deny("caller session could not be identified", "caller_unidentified")
    # Resolved before the config gate: a member DM session is authorized
    # WITHOUT `agent.session_control` — dispatching and patrolling workers is
    # its operating model — while the operator ceiling `agent.member_dispatch`
    # (default true = today's behaviour) is on. Turn that ceiling off and the
    # member falls back under the switch. The member's reach stays bounded by
    # the ownership check below, which restricts it to slots it created itself.
    if (
        not skip_enabled_check
        and not session_control_enabled()
        and not _member_bypass(state, caller_key)
    ):
        raise deny(
            "session control is disabled in config (agent.session_control)",
            "session_control_disabled",
        )
    if caller_key.startswith(UNATTENDED_SLOT_PREFIXES) and not _cron_caller(caller_key):
        raise deny(
            "unattended sessions (scheduled runs) cannot control other sessions",
            "unattended_caller",
        )
    if (refusal := _app_owned_cron_refusal(state, caller_key)) is not None:
        # The app-confinement refusal, reached through an app's cron rather than
        # its session -- a cron tab carries no ``_app`` tag for the check further
        # down to read. See :func:`_app_owned_cron_refusal`.
        raise deny(refusal[0], refusal[1])
    return caller_key


def refuse_caller_surface(
    state: "DashboardState",
    *,
    caller_key: str,
    deny: Deny,
) -> "_ChatSlot":
    """The caller-side gate about the caller's own SURFACE. Returns its slot.

    The caller's own isolation gates it too, and for the same reasons the target's
    does: an incognito or temporary session is one the user asked to leave no
    trace, and an app-scoped session belongs to its app. Either one reaching a
    persistent peer would launder content across the boundary it was created to
    have — in the direction the target-side checks cannot see.

    A second helper rather than one with :func:`refuse_caller_identity`, because
    :func:`authorize_target` runs these two on OPPOSITE sides of its target
    resolution and merging them would move a refusal's precedence. A targetless
    verb resolves nothing, so it calls both back to back.
    """
    # The slot-field half lives in :func:`_check_caller_slot_fields` so a
    # synchronous last-word check can re-assert it adjacent to a publish
    # (``revive_session``'s ``final_check``); the two channel refusals stay here
    # because their owner-DM exemption reads the session store.
    caller_slot = _check_caller_slot_fields(state, caller_key, deny)
    # The one exemption from both caller-side channel refusals below, shared with
    # `_refuse_ineligible_creator` so the two halves cannot drift on WHO is exempt:
    # :func:`owner_dm_refusal` answering ``""``, a 1:1 DM whose only human is the
    # configured owner -- the DM the session lives in, or the DM a dashboard tab
    # mirrors to -- with no wider room bound beside it. It waives both refusals
    # together, because it has already established that the mirror IS the DM --
    # waiving the link alone would refuse every owner DM on the origin mirror its
    # dispatcher binds each turn. An admitted channel-born DM is creator-fenced
    # further down; an admitted dashboard tab keeps the fence verdict it had. The
    # refusal names the clause that failed, code unchanged. ONE read: the verdict's
    # ``mirrored`` flag and the ``admission`` recorded below come from the row the
    # clauses judged, so nothing here probes the store a second time.
    verdict = judge_owner_dm(state, caller_slot)
    if why := verdict.refusal:
        if _channel_link_of(caller_slot):
            # The exfiltration direction, and the reason this is not merely the
            # mirror of the target-side check: a linked caller's own conversation
            # is a channel thread, so anything it reads lands in front of whoever
            # is in that channel. `session_read_message` would hand a private
            # dashboard transcript to Slack/Discord readers who were never party
            # to it.
            #
            # `CHANNEL_AGENT_BLOCKED_TOOLS` already blocks these tools for channel
            # AGENTS, but that guard keys on the agent identity; a linked SLOT is
            # a second route to the same surface and has to be closed on its own.
            #
            # A cron tab's link is exempt because it is not a channel: it names
            # the job's own run transcript and republishes to nobody, so a read
            # through it reaches no audience the caller did not already have. See
            # CRON_LINK_PREFIX.
            raise deny(
                "channel-linked sessions cannot control other sessions; the owner-DM "
                f"exemption is withheld because {why}",
                "linked_session_caller",
            )
        if verdict.mirrored:
            # The exfiltration direction again, via the outbound mechanism: a
            # mirrored caller republishes its own turns to a channel, so a peer's
            # transcript it reads lands in front of that channel's audience.
            raise deny(
                "sessions mirrored to a channel cannot control other sessions; the "
                f"owner-DM exemption is withheld because {why}",
                "mirrored_caller",
            )
    # Admitted. What this verb returns -- a peer's transcript, the roster of
    # created sessions and their titles, a delivery receipt -- becomes part of the
    # caller's reply, which the caller's own turn publishes to its mirror LATER,
    # resolved live. Record the audience it was admitted under -- the row the
    # verdict judged, not a fresh read -- here at the one gate every verb passes
    # rather than per verb, so no read path can miss it.
    record_audience_admission(caller_slot, verdict.admission)
    return caller_slot


def _check_caller_slot_fields(
    state: "DashboardState",
    caller_key: str,
    deny: "Callable[..., SessionControlError]",
) -> "_ChatSlot":
    """The caller-slot refusals answerable from the slot's own fields, no store.

    Split out so a synchronous last-word check can re-assert them adjacent to a
    publish (``revive_session``'s ``final_check``); the two channel refusals stay
    in :func:`refuse_caller_surface`, because their owner-DM exemption reads the
    session store.
    """
    # The caller's own isolation gates it too, and for the same reasons the
    # target's does: an incognito or temporary session is one the user asked to
    # leave no trace, and an app-scoped session belongs to its app. Either one
    # reaching a persistent peer would launder content across the boundary it
    # was created to have — in the direction the target-side checks cannot see.
    caller_slot = state.get_slot(caller_key)
    if caller_slot is None:
        raise deny("caller session is no longer open", "caller_gone")
    if getattr(caller_slot, "_app", ""):
        raise deny("app-scoped sessions cannot control other sessions", "app_scoped_caller")
    if getattr(caller_slot, "memory_mode", "persistent") != "persistent":
        raise deny(
            "incognito and temporary sessions cannot control other sessions",
            "ephemeral_caller",
        )
    return caller_slot


def _live_target_refusal(slot: "_ChatSlot") -> tuple[str, str] | None:
    """The target-side containment refusal a LIVE slot earns, or ``None``.

    One spelling of the four fields that decide whether a live session may be
    reached at all -- the unattended prefix, memory mode, app scope and channel
    link -- shared by :func:`authorize_target` (on the resolved target) and
    :func:`revive_session` (on the slot the resume hydrated), so the two cannot
    drift. The outbound-mirror probe and the workspace compare sit beside it in
    each caller: the first needs the state and a thread choice, the second the
    caller's slot.

    A channel-linked session's conversation is mirrored to Slack/Telegram, so
    reaching it crosses a surface boundary in both directions: a message would
    surface to whoever reads that thread, and a read would pull the channel's
    content back. It is also the one target whose STOP cannot be honoured: the
    stop path addresses the session as ``dashboard:<slot>`` while a linked slot's
    turns actually run under its ``linked_session_key``, so the cancel would miss
    and the target would keep executing after a reported success. Refusing is the
    honest answer until the stop path resolves the effective key.
    """
    if slot.key.casefold().startswith(UNATTENDED_SLOT_PREFIXES):
        return ("unattended sessions (scheduled runs) cannot be controlled", "unattended_target")
    if getattr(slot, "memory_mode", "persistent") != "persistent":
        return ("incognito and temporary sessions are not addressable", "ephemeral_target")
    if getattr(slot, "_app", ""):
        return ("app-scoped sessions are not addressable", "app_scoped_target")
    if getattr(slot, "linked_session_key", ""):
        return ("channel-linked sessions are not addressable", "linked_session_target")
    return None


def _not_creator_reason(
    state: "DashboardState",
    caller_key: str,
    caller_slot: "_ChatSlot",
    precomputed_ownership_fenced: bool | None,
) -> str:
    """The wording of an ownership-fence refusal; see the note in authorize_target.

    The cron prefix and the channel link are readable without config; the inline
    path keeps the per-class wording since it is already reading config anyway.
    """
    if _cron_caller(caller_key):
        return "a scheduled run can only control sessions it created itself"
    elif _channel_link_of(caller_slot):
        return "an owner-DM channel session can only control sessions it created itself"
    elif precomputed_ownership_fenced is not None:
        return "this session can only control sessions it created itself"
    elif _member_caller(state, caller_key):
        return "a crew member can only control worker sessions it created itself"
    else:
        return "an agent-created session can only control sessions it created itself"


# The key prefix of an audience admission in ``_ChatSlot._steer_audience_fences``,
# so an admission and a peer steer's uuid-keyed entry cannot collide and a reader
# of the record can tell the two writers apart.
AUDIENCE_ADMISSION_KEY_PREFIX = "admission:"


def record_audience_admission(slot: Any, admission: dict[str, Any]) -> None:
    """Record *admission* on *slot* as the audience its current turn was admitted under.

    *admission* is the ``admission`` of the :class:`OwnerDmVerdict` the gate just
    judged -- the containment of the very row the clauses saw, in the queue-entry
    ``meta`` shape. This function takes it rather than probing the store itself, on
    purpose: a record built from a second read would describe whatever row was live
    at the second read, and a retarget landing between the two would be recorded as
    the admitted audience without having been judged -- the publisher, comparing the
    live row against that record, would then see no change and publish the private
    read into the new room. Handed the judged row, the record can only ever be the
    audience that was actually admitted.

    The read-side twin of the peer steer's record: the publisher compares the
    containment holding at reply delivery against every entry in
    ``_steer_audience_fences`` and withholds the cross-surface legs when a
    constraint newly holds -- a mirror gained, a mirror retargeted, a Slack thread
    bound beside it -- because a sent reply cannot be recalled. The channel-neutral
    leg (``chat_runner._deliver_cross_surface_reply``) judges the ONE binding it
    resolves its target from (:func:`publication_withheld`); the Slack legs ask
    ``cross_surface_withheld``. Written at the caller-side admission gates
    (:func:`refuse_caller_surface` for every targeted session-control verb,
    :func:`_refuse_ineligible_creator` for a creation, which passes no other gate,
    the work ledger's ``_caller_key`` for every ledger route) rather than per verb,
    so a read path cannot be added without it.

    ONE entry per distinct audience per turn: the key is the snapshot itself, so a
    conductor polling a worker every few seconds re-records the same audience as
    a no-op and the record stays the size of the number of DISTINCT containment
    states the turn passed through -- each of which is a real change the
    publisher must see -- never the number of calls. That is what bounds it
    without a cap, and a cap is exactly what this record must not have: evicting
    an admission would turn the fail-closed publication gate fail-open. The turn's
    teardown empties the record, so it is turn-scoped like the steer's entries.
    Keyed under :data:`AUDIENCE_ADMISSION_KEY_PREFIX` so it cannot collide with a
    steer's uuid key, which the steer path pops by name.

    Written ONLY inside a turn whose teardown clears it (:func:`_in_runner_turn`):
    the dashboard runner's ``_run_chat`` publishes ``_active_turn_session_key`` on
    the slot at turn start and retires it in the same ``finally`` that empties this
    record. A channel-born session's own turns run in the messaging driver, which
    sets no such marker and runs no such teardown -- so an owner-DM conductor that
    calls these tools from its channel turn would leave an entry that nothing
    clears, and a mirror change afterwards would make the publisher withhold the
    mirror leg of an unrelated dashboard reply on that slot, judged by an admission
    that was never about it. Nothing is lost by not recording there: the
    dispatcher publishes its own replies to the conversation the turn came from,
    and the runner's publisher, the only reader of this record, publishes only
    runner turns.

    A caller with no slot record of its own leaves nothing, which is the behaviour
    such a caller has today.

    NO OPT-OUT from the in-a-turn guard, because every caller is inside the turn its
    admission is about. A producer that merely has a turn PENDING cannot be: an entry
    nothing clears makes the publisher withhold the mirror leg of an unrelated later
    reply, judged by an admission that was never about it. The work-ledger wake is
    the one shape that could want it -- it delivers into a turn it starts -- and it
    carries a pointer to ``work_ledger_read`` rather than a board, so the read that
    needs an admission is the conductor's own call inside its own turn.
    """
    fences = getattr(slot, "_steer_audience_fences", None)
    if slot is None or not isinstance(fences, dict):
        return
    if not _in_runner_turn(slot):
        return
    key = AUDIENCE_ADMISSION_KEY_PREFIX + json.dumps(
        admission.get(QUEUED_CONTAINMENT_META_KEY, {}), sort_keys=True, default=str
    )
    if key not in fences:
        fences[key] = admission


def _in_runner_turn(slot: Any) -> bool:
    """Whether *slot* is inside a dashboard-runner turn -- one ``_run_chat`` is running.

    Read off ``_active_turn_session_key``, the identity ``_run_chat`` publishes on
    the slot once its turn is admitted and compare-and-clears in its teardown. It
    is the runner's own marker, so it names exactly the turns whose ``finally``
    empties ``_steer_audience_fences``: a channel turn driven by the messaging
    driver never sets it, and neither does the gap between two stages of a plan.
    Preferred over ``turn_running`` for that reason -- the task predicate also
    counts a stage controller that owns the slot while no runner turn, and no
    runner teardown, is in flight.
    """
    return bool(getattr(slot, "_active_turn_session_key", "") or "")


def publication_withheld(
    state: "DashboardState", slot: Any, binding: tuple[Any, str, str] | None
) -> bool:
    """Whether *slot*'s turn must NOT publish its reply to the audience *binding* names.

    The channel-neutral delivery leg's half of the audience fence. *binding* is the
    ONE :func:`_read_mirror_binding` that ``chat_runner._deliver_cross_surface_reply``
    resolves its target from, judged here against every admission the turn recorded
    in ``_steer_audience_fences`` -- :func:`record_audience_admission`'s, the peer
    steer's, the handoff's hold -- through the same comparison the Slack legs ask of
    ``cross_surface_withheld``: :func:`newly_held_constraints` over a snapshot built
    from that read (:func:`_snapshot_from_probe`). The decision and the delivery see
    one row. Judged from a second read, the decision would describe one row and the
    send would land in another whenever a retarget fell between them -- the
    mirror-link writer runs off the loop (``chat_mirror._claim_binding`` under
    ``asyncio.to_thread``), so nothing orders it against two reads on the loop --
    and a transcript the decision never saw would reach the new room. Handed the
    row the send resolves from, the judgement can only ever be about that room; a
    retarget landing after this read changes what the NEXT reply is judged against,
    not where this one goes.

    A binding of ``None`` is an unreadable store and is judged as an unverifiable
    mirror (``mirror_unverified``), which withholds whenever the turn recorded an
    admission: the drain's fail-closed reading, for the drain's reason. A slot with
    no record -- a caller outside a turn, a turn that admitted nothing -- withholds
    nothing, as it does today.
    """
    fences = getattr(slot, "_steer_audience_fences", None)
    if not fences:
        return False
    probed = None if binding is None else _mirror_identity_of(binding)
    now = _snapshot_from_probe(state, slot, probed, on_probe_failure=True)
    return any(newly_held_constraints(now, admission) for admission in fences.values())


def authorize_target(
    state: "DashboardState",
    *,
    caller_session_key: str,
    target: str,
    operation: str,
    skip_enabled_check: bool = False,
    precomputed_ownership_fenced: bool | None = None,
    precomputed_manager_chain: bool | None = None,
    allow_self: bool = False,
) -> "_ChatSlot":
    """Resolve *target* and decide whether *caller* may act on it.

    Deny-by-default: every refusal raises :class:`SessionControlError` and is
    recorded in the SEL, so an attempt to reach a session that is out of bounds
    is visible after the fact even though nothing happened.

    ``skip_enabled_check`` omits ONLY the ``session_control_enabled()`` config
    read. It exists for a re-check that must run SYNCHRONOUSLY with no event-loop
    suspension (``close_target``'s point-of-no-return callback): the feature was
    already confirmed enabled when the operation was first authorized, whether
    session control got switched off mid-operation is not a containment boundary,
    and the config read is the one part of this function that can touch the disk
    on a cache miss. Every containment and identity refusal still runs.

    ``precomputed_ownership_fenced`` is an ownership-fence verdict the caller of
    this function already holds, honoured instead of re-deriving one here. Two
    callers hold one:

    * The HTTP gate (``handlers/session_control.py``'s ``_private_caller_refusal``)
      admits a crew member on its VERIFIED private scope and passes ``True`` down
      through every route. The inline fence (:func:`_caller_is_ownership_fenced`
      → :func:`_member_caller` → :func:`_store_is_member_owned`) re-reads the
      MUTABLE config record, and an operator's own writer can flip that record —
      un-assign the member, drop ``memory_version`` (coerced to ``1`` by the
      loader), drop the entry — in the awaits between the gate and this call. A
      member admitted as one must stay bounded to what it created for the whole
      request, so the verified decision travels with the request rather than
      being recomputed from whatever the record says at the fence.
    * ``close_target`` resolves the verdict ONCE up front (behind
      ``prewarm_enabled_check``) and passes it to both its initial gate and its
      SYNCHRONOUS point-of-no-return re-check, so no ``KiroCrewConfig.load()`` runs
      on the loop inside ``close_slot``'s no-suspension window — the same
      blocking-IO hazard ``skip_enabled_check`` closes for the switch read.

    When ``None`` (an owner or agent-created caller the gate did not admit as a
    member) the fence is evaluated inline as before.

    ``allow_self`` waives the self-target refusal, and with it the ownership fence for
    that one case. Three verbs pass it. A release, where the target itself is a
    legitimate caller because a session taken over must not depend on its holder still
    running to get out. A close, so a session that has finished its one job can archive
    its own tab instead of leaving it for the person or its creator to dismiss; the close
    is the same recoverable archival a peer close is, and it cancels the caller's own
    running turn, which is the turn asking for it. And a rename, because a session naming
    itself acts on nothing but its own sidebar label. It waives nothing else -- an ephemeral, app-scoped or
    channel-linked caller is still refused, and a target that is not the caller is
    still judged by every rule above.
    """

    deny = _deny_factory(caller_session_key=caller_session_key, operation=operation, target=target)

    # NO manager gate here, deliberately. The refusal owed when who manages a slot has
    # not been read yet lives on the READ (:func:`require_manager_of`), not at this
    # entry: a condition here has to name which operations consult the value, and that
    # list was wrong at three sites that reach the value without passing this function
    # -- the handoff verb's ``already_manager`` check, its commit's ``previous_manager``
    # and :func:`broadcast_audience`. It also refused callers whose answer could not
    # change: the fence below short-circuits on ``ownership_fenced``, so an unfenced
    # caller never reads the manager at all and is never refused for it.
    caller_key = refuse_caller_identity(
        state,
        caller_session_key=caller_session_key,
        deny=deny,
        skip_enabled_check=skip_enabled_check,
    )

    try:
        slot = _resolve_slot(state, target)
    except SessionControlError as exc:
        raise deny(exc.message, exc.code, status=exc.status) from exc
    if slot is None:
        # 404 rather than 403: naming a session that is not open is a mistake,
        # not an authorization failure. Only sessions the dashboard currently
        # holds are addressable — a closed tab is out of scope, because waking
        # one would resurrect a conversation the user put away.
        raise deny(f"no open session matches {target!r}", "target_not_found", status=404)

    if slot.key == caller_key and not allow_self:
        raise deny("a session cannot control itself", "self_target")
    if (refusal := _live_target_refusal(slot)) is not None:
        raise deny(refusal[0], refusal[1])
    if _has_channel_mirror(state, slot):
        # Same boundary as the channel-link refusal, reached by the other
        # mechanism: an outbound mirror republishes this session's turns to a
        # channel, so a read would pull that channel's content back and a stop
        # would act on a conversation other people are party to.
        raise deny("sessions mirrored to a channel are not addressable", "mirrored_target")

    caller_slot = refuse_caller_surface(state, caller_key=caller_key, deny=deny)

    if getattr(slot, "workspace", "default") != getattr(caller_slot, "workspace", "default"):
        # Workspaces are the memory boundary; reaching across one would let a
        # session act on work it cannot see.
        raise deny("target session belongs to a different workspace", "workspace_mismatch")
    # Resolve the fence verdict ONCE. A caller passing ``precomputed_ownership_fenced``
    # already holds it — the HTTP gate's verified member admission, or
    # ``close_target``'s up-front pass — so honour that value rather than
    # re-deriving it from the config record here (see the docstring). Everyone
    # else evaluates it inline.
    ownership_fenced = (
        _caller_is_ownership_fenced(state, caller_key)
        if precomputed_ownership_fenced is None
        else precomputed_ownership_fenced
    )
    # A caller addressing ITSELF is not reaching a peer, so the fence has nothing to
    # protect and is waived -- reachable only under ``allow_self``, since the
    # self-target refusal above denies this case for every other verb. Without the
    # waiver an agent-created session could never release itself from a parent,
    # because its own ``_created_by`` names its creator and not itself.
    self_addressed = slot.key == caller_key
    # The fence reads the slot's MANAGER LINE for the verbs a handoff widens, and its
    # creator for every other verb. The line starts at :func:`manager_of`, which
    # answers ``_created_by`` until something writes ``_manager``, and then climbs.
    #
    # So this WIDENS what the fence admits, and it does so whether or not anything was
    # ever handed over. A handoff is not a precondition: on a gateway where nothing has
    # been handed over, the line IS the creation chain, and a caller that created the
    # session that created the target now reaches the target. Only a one-level chain
    # decides what ``_created_by_other`` decided before the field existed -- its walk
    # has nowhere to go -- and a lead that opens a conductor which opens workers is
    # exactly the two-level case this admits.
    #
    # That is the intent and not a side effect. A caller on the line above already
    # drives the middle session, so it can already have the target driven by telling
    # the middle one to; the walk removes the relay, not a boundary. And it is what
    # keeps the line usable when the middle session closes, which is the ordinary end
    # of that session's job rather than an exceptional state.
    #
    # Split by OPERATION rather than by caller, because what a handoff moves is the
    # right to run a session, not the right to mint one against the giver's memory
    # binding. ``create`` and ``fork`` are not in the widened set and the memory
    # delegation gate reads ``_created_by`` alone, so a manager that was given a
    # worker cannot turn it into a deputy with the giver's store.
    #: The line walk's reason for not finishing, if it ran, so the refusal can name
    #: which refusal it is. A dropped record and a caller who was never on the line are
    #: different situations and must not read alike.
    line_cause: "list[str]" = []

    def _manager_fence(slot: Any, caller_key: str) -> bool:
        """The widened fence: the one-level predicate, or the whole line when it is on.

        Wraps the base PR's own ``_managed_by_other`` rather than replacing it. With
        the chain OFF this IS that predicate and nothing else runs, so off is the base
        PR's behaviour and not a mode of this one. With it on, the walk decides and its
        reason is kept for the refusal below.

        The asker's own IDENTITY goes with its key into BOTH answers, read from the key
        it is asking under rather than captured from the enclosing scope: an identity
        belonging to a different key than the one being judged is the defect this
        parameter exists to close, wearing the right name. Redundant here by
        construction, since a live caller is the occupant of the key it was authorized
        as, and passed for the reason :func:`_managed_by_other` gives -- the asking site
        that omits it is the defect, so no site omits it.
        """
        asker_tab = _tab_identity(state, caller_key)
        chain_enabled = (
            manager_chain_enabled()
            if precomputed_manager_chain is None
            else precomputed_manager_chain
        )
        if not chain_enabled:
            return _managed_by_other(state, slot, caller_key, asker_tab)
        outside, cause = _outside_manager_line(state, slot, caller_key, asker_tab)
        line_cause[:] = [cause]
        return outside

    fence_other = _manager_fence if operation in HANDOFF_WIDENED_OPERATIONS else _created_by_other
    # The manager refusal surfaces HERE and only here, because this is the line that
    # reads the value: ``_managed_by_other`` goes through :func:`require_manager_of`,
    # which raises when the manager has not been read yet. Re-raised through ``deny``
    # so the refusal carries this gate's audit shape and operation wording like every
    # other one -- the code and the 409 come from the read, which is what decided.
    #
    # Short-circuit order is the availability property: ``ownership_fenced`` and
    # ``self_addressed`` are evaluated first, so an unfenced caller and a caller
    # addressing itself never reach the read and are never refused for an unread
    # manager. On a gateway where nothing was ever handed over, that is every caller.
    try:
        fenced_out = ownership_fenced and not self_addressed and fence_other(slot, caller_key)
    except SessionControlError as exc:
        raise deny(exc.message, exc.code, status=exc.status) from exc
    if fenced_out:
        # The fence every exempted caller class is bounded by, plus anything they
        # created. It reaches ONLY the sessions the caller made itself
        # (`created_by` is written at birth and rehydrated on restart). Always
        # enforced -- even when the global switch is on -- so no exemption
        # silently widens to the user's own sessions because of an unrelated
        # opt-in.
        #
        # For a cron caller this fence stands in place of the `unattended_caller`
        # refusal every other unattended caller gets: a scheduled job reaches the
        # sessions it dispatched and nothing else. Fail-closed on an unowned slot,
        # which is what an ownerless rehydrate looks like.
        #
        # The reason is cosmetic (the error string only). A carried verdict says
        # WHETHER the caller is fenced, not WHY, and telling the member wording
        # from the agent-created wording needs ``_member_caller``, which can read
        # config — the ``close_target`` re-check runs in a no-suspension window and
        # MUST NOT reach it. So on a carried verdict the text names the rule
        # rather than a class it cannot see; the cron prefix is still readable
        # without config, and the inline path keeps the three-way wording since
        # it is already reading config anyway.
        # A RETENTION refusal reads differently from an ownership one, because they are
        # different situations: "you were never on this line" and "this line was
        # readable until a cap dropped a record" need different answers from whoever is
        # debugging a denial. The cap is reached by ordinary use -- every slot
        # construction takes a key -- so without this a previously authorized caller
        # starts getting a permission error with nothing anywhere naming the cause.
        #
        # A reused key is a THIRD situation and keeps its own wording too: forgotten
        # means the record is gone, reused means the record cannot say which occupant
        # an edge meant. Both refuse; neither explains the other.
        cause = line_cause[0] if line_cause else ""
        if cause == "forgotten":
            # The CAUSE is named only when this process has actually dropped something.
            # The same walk answer is given for a record that was never written and for
            # one a close removed because it could never settle -- and telling those
            # callers "the line passed through 512 other sessions" sends whoever is
            # debugging the denial to look for churn that did not happen. What is true
            # in every case is that the record is not available; the retention limit is
            # an additional fact, and it is stated when it is one.
            raise deny(
                (
                    (
                        f"{target!r} is reachable only through a session whose manager "
                        f"record this gateway has dropped: the line passed through more "
                        f"than {_KEY_OCCUPANCY_MAX} other sessions since it closed, so the "
                        "reach is gone. This is a retention limit rather than a decision "
                        "about you"
                    )
                    if _key_occupancy_evictions[0]
                    else (
                        f"{target!r} is reachable only through a session whose manager this "
                        "gateway has no usable record of, so the line cannot be read to the "
                        "end. This is not a decision about you, and it does not clear by "
                        "retrying"
                    )
                ),
                "manager_line_forgotten",
            )
        if cause == "reused":
            raise deny(
                f"{target!r} is reachable only through a session key that has since "
                "been taken by a different session, so which one the line meant cannot "
                "be established and the reach is refused",
                "manager_line_ambiguous",
            )
        if cause == "inadmissible":
            # A FOURTH situation, and it is the only one of the four that is a decision
            # about somebody: a hop on this line was handed to a manager that may not
            # hold it -- closed, or running under another principal's store. Saying
            # "dropped" or "ambiguous" here would send whoever is debugging it to look
            # for a retention limit that is not there.
            raise deny(
                f"{target!r} is reachable only through a session whose own manager was "
                "recorded and is not admissible, so the line stops there and the reach "
                "is refused until that session is handed over again",
                "manager_line_inadmissible",
            )
        fence_reason = _not_creator_reason(
            state, caller_key, caller_slot, precomputed_ownership_fenced
        )
        raise deny(fence_reason, "not_creator")

    return slot


def _slot_tree_parent(slot_key: str) -> "tuple[bool, str, dict[str, Any]]":
    """Whether the tree is KNOWN right now, *slot_key*'s parent in it, and the tree.

    The first value is the one that must not be collapsed into the others. "This slot
    has no parent" and "I cannot see the tree" are different answers, and an adoption
    that treats them alike skips its own cycle guard: an unseeded projection folds to
    no nodes, and a guard given no nodes admits everything. That state is not exotic --
    it is every gateway between boot and the first lineage seed, so it recurs on each
    restart, and the cycle it would admit is not repaired by the fold, which flattens
    the branch instead.

    ``False`` for an unreadable tree: the crew log is off, the projection is not seeded
    for the store configured now, the fold is INCOMPLETE, or the read raised. ``True``
    with an empty parent means the tree was read whole and the slot is a root.

    NO I/O and never blocks, which is what lets it run on the loop: it reads the
    projection's in-memory fold and asks first whether that fold is seeded for the
    store configured now. An unseeded projection answers "not known" rather than
    seeding itself here -- a seed is a disk read, and a verb that blocks the loop is
    the wrong trade when refusing is honest and the next call succeeds.
    """
    try:
        from kiro_crew.crew_log import emit as crew_log_emit

        if not crew_log_emit.enabled():
            return False, "", {}
        from kiro_crew.crew_log.session_tree_projection import projection

        proj = projection()
        if not proj.seeded_for_current_store:
            return False, "", {}
        # ``reading`` rather than ``nodes``, because this caller DECIDES on an edge.
        # ``TreeReading.incomplete`` says a unit's bytes could not be read or the
        # population was capped, so an edge the guard needs may simply be missing from
        # an otherwise well-formed fold. Admitting an adoption on that fold is a
        # confident wrong answer, and the shape it admits is the cycle this guard
        # exists to refuse -- so an incomplete fold is treated as no tree at all.
        reading = proj.reading()
        if reading.incomplete:
            return False, "", {}
        nodes = reading.nodes
    except Exception:
        logger.debug("session tree parent could not be resolved", exc_info=True)
        return False, "", {}
    node = nodes.get(slot_key)
    parent = getattr(node, "parent_slot", None) if node is not None else None
    return True, (parent or ""), nodes


def _live_sid_of(state: "DashboardState", slot_key: str) -> str:
    """The ACP session id *slot_key*'s crew log is written under, or ``""``.

    For the slot the operation is ALREADY HOLDING, which is the whole of this
    function's contract. A caller resolving some OTHER slot wants
    :func:`_recorded_sid_of` instead: the mapping this reads is one process's, so a
    caller with no context on the slot it is asking about cannot see a stale or
    absent answer, and these ids are written into append-only entries.

    Read from the DURABLE session map rather than from the slot's in-turn ACP client.
    The client is published when a turn starts and cleared when it ends, so reading it
    would answer only for a session that happens to be working -- and the sessions a
    takeover is aimed at are the idle ones.

    ``mapped_sid`` rather than ``get``, which is the difference between two questions.
    ``get`` asks whether the id can still be RESUMED and prunes the entry when the ACP
    transcript is gone; this caller is recording history into a crew log, and a crew log
    unit outlives a truncated transcript. It is also read-only and in-memory, so it is
    safe on the event loop.

    ``""`` for a slot with no mapping: a session whose log this gateway cannot name.
    The caller refuses rather than writing into a log it guessed at.
    """
    try:
        sessions = getattr(state, "sessions", None)
        if sessions is None:
            return ""
        sid = sessions._session_map.mapped_sid(f"dashboard:{slot_key}")
        return sid if isinstance(sid, str) else ""
    except Exception:
        logger.debug("session id for %s could not be resolved", slot_key, exc_info=True)
        return ""


def _replay_pending(state: "DashboardState", slot_key: str) -> bool | None:
    """Whether *slot_key* still owes conversation replay, or ``None`` when unknown.

    The window in which the session map deliberately names the generation BEFORE the
    store the slot is writing: allocation leaves the prior resumable id mapped for a
    replay-pending ACP session on purpose, so a restart can still resume it. A history
    reader taking the mapping's answer inside this window records the wrong log.

    ``None`` rather than ``False`` when the session store cannot answer, because the
    two have opposite consequences for the caller: an unknown window must not license
    the mapping read that a known-closed one does.

    A slot with NO live session is one of those unknown answers, and asking for it
    explicitly is what makes the tri-state real.
    ``SessionRegistry.provider_switch_replay_pending`` is ``bool(session is not None
    and session.provider_switch_replay)``, so a missing session and a live one that
    owes nothing both come back ``False`` -- and only ``False`` licenses the mapping.
    That collapse points at the very case the licence is least safe in: the mapping
    keeps answering a dropped id after a session closes, and a cold start has not
    written one yet, so the id it names is most likely to be the older generation
    exactly when there is nobody to ask about replay.
    """
    try:
        sessions = getattr(state, "sessions", None)
        if sessions is None:
            return None
        key = f"dashboard:{slot_key}"
        if not sessions.has_session(key):
            return None
        return bool(sessions.provider_switch_replay_pending(key))
    except Exception:
        logger.debug("replay state for %s could not be read", slot_key, exc_info=True)
        return None


def _slot_opened_sid(state: "DashboardState", slot_key: str) -> str:
    """The crew log this process recorded when *slot_key* opened, or ``""``."""
    try:
        slot = state._slots.get(slot_key)
        if slot is None:
            return ""
        sid = getattr(slot, "_crew_log_opened_sid", "")
        return sid if isinstance(sid, str) and sid else ""
    except Exception:
        logger.debug("opened crew log for %s could not be read", slot_key, exc_info=True)
        return ""


def _slot_object(state: "DashboardState", slot_key: str) -> "object | None":
    """The live slot OBJECT for *slot_key*, or ``None``, for an identity comparison.

    Every other read here is by key, and a key is not an identity: a slot can close and
    a new session can reopen under the same key, which no key-only check can tell from
    the original. Captured before a suspension and compared after, the object itself
    can.
    """
    try:
        return state._slots.get(slot_key)
    except Exception:
        logger.debug("slot object for %s could not be read", slot_key, exc_info=True)
        return None


def _target_replaced(state: "DashboardState", slot_key: str, observed: "object | None") -> bool:
    """Whether the slot under *slot_key* differs from *observed*, the captured session.

    The identity read-back that a tree verb runs after its final authorization. *observed*
    is the target slot object captured BEFORE the verb's first suspension; this re-reads
    the object under the same key and reports whether it has changed. A close plus a
    reopen under the SAME key inside the suspensions between capture and append puts a
    different session's object there -- open, passing every fence, with a log of its own
    -- which no key-only check can tell from the original. A verb that finds this true
    refuses rather than appending an adoption or release against a session that never took
    part, in an entry nothing later corrects.

    Encapsulated rather than inlined for the reason :func:`_freshest_sid` is: the slot
    captures a verb takes before its resolutions are what the capture-ordering invariant
    rests on, and a bare :func:`_slot_object` read in the verb body AFTER the suspending
    resolution would read as a late capture. The read-back lives here, where it is a
    comparison and not a capture.

    ``True`` when nothing is there now and something was (the target simply closed), which
    is a refusal too -- a verb cannot record a change to a session that is gone. The pure
    close is also caught earlier by the gate's own re-resolve, which is why the close
    tests assert ``target_not_found``; this guard's own case is the reopen the gate admits.
    A ``None`` capture (the target was never a held object) can only become non-``None``,
    which is still a change, so the comparison stays correct at the edges.
    """
    return _slot_object(state, slot_key) is not observed


def _freshest_sid(
    state: "DashboardState", slot_key: str, resolved: str, observed: "object | None"
) -> str:
    """*resolved*, refreshed from the slot's own record when that has moved since.

    SYNCHRONOUS on purpose, and that is the whole reason it exists separately from
    :func:`_recorded_sid_of`. That resolver has to suspend -- it reads the durable
    store -- so a verb resolves its ids BEFORE the final authorization, and the
    authorization's own config warm suspends as well. A slot that opens its next
    store inside that hop advances its ``_crew_log_opened_sid``
    (``Slot.take_crew_log_previous``), and the id resolved before the hop then names
    the store that slot has just replaced. Written into an append-only
    ``session/adopted`` or ``session/released`` entry, that is the same permanent
    wrong answer this resolver exists to prevent, arriving one hop later.

    Re-reading the slot's own record is enough to close the window because it is the
    resolver's FIRST preference and the only one of its three sources that can move
    during the hop: the durable store and the mapping are consulted only when that
    record is empty, and neither is more current than a statement this process just
    wrote about which store the slot is on.

    A dict lookup and an attribute read, so it belongs after the final
    authorization, where nothing may suspend -- refreshing before that gate would
    leave the same window open behind it.

    An empty slot record keeps *resolved*: a slot that CLOSED during the hop does not
    make the store's answer about which log it was on wrong, and falling back to
    ``""`` there would drop an id that is still the best available one.

    *observed* is the slot object read when *resolved* was resolved, and refreshing is
    conditional on it still being the slot under that key. A key is not an identity: a
    close plus a reopen under the SAME key inside the hop puts a different session's
    object there, and its opened-store record names a lineage that never held the
    target -- so refreshing from it would replace a right answer with a confident wrong
    one. Together with the empty-record rule above this makes the refresh never worse
    than not refreshing: it moves an id only when the same slot moved it.
    """
    if observed is None or _slot_object(state, slot_key) is not observed:
        return resolved
    return _slot_opened_sid(state, slot_key) or resolved


async def _recorded_sid_of(state: "DashboardState", slot_key: str) -> str:
    """The crew log *slot_key* is writing, or ``""`` when no id can be written.

    For a slot the caller is NOT holding. The id goes into an append-only
    ``session/adopted`` or ``session/released`` entry as ``parent`` or
    ``previous_parent``, so a wrong id there is a permanent wrong answer about which
    conversation a session came from, and there is no later write that corrects it.
    The citation's slot half distinguishes a parent whose log is unnamed from no parent.

    Resolution checks the live slot's own opened-store record, then the durable store,
    then the process-local mapping. The slot record leads because it is this process's
    own statement about which store the slot is on, and the only source that can name a
    store whose unit the background writer has not written yet. An undecided store
    answer is never a licence to guess from the mapping. A decided empty answer may use
    the mapping only outside the replay-pending window, where the mapping is current.

    Keep this separate from ``chat_runner._slot_predecessor_store``: this resolver runs
    inside a live verb and can read the slot's replay state, while that resolver runs
    before its turn's session exists and cannot ask whether replay is pending.
    """
    if not slot_key:
        return ""

    opened = _slot_opened_sid(state, slot_key)
    if opened:
        return opened

    derived, decided, _complete = await asyncio.to_thread(
        crew_log_emit.slot_previous_store, slot_key
    )
    if derived:
        return derived
    if not decided:
        return ""
    pending = _replay_pending(state, slot_key)
    if pending is not False:
        return ""
    mapped = _live_sid_of(state, slot_key)
    if mapped:
        return mapped
    return ""


async def _reparent_target(
    state: "DashboardState",
    *,
    caller_session_key: str,
    target: str,
    parent_key: str,
    operation: str,
    noun: str,
    caller_fenced: bool | None = None,
    gate: "Callable[[_ChatSlot], None] | None" = None,
    commit: "Callable[[_ChatSlot, str], Awaitable[None]] | None" = None,
    fence_parent: bool = False,
    after_commit: "Callable[[], None] | None" = None,
) -> "dict[str, Any]":
    """Put *target* under *parent_key* in the tree, and record that it moved.

    The body the two verbs that move a session SHARE: :func:`adopt_target`, where the
    new parent is the caller, and :func:`handoff_target`, where the caller names the
    session it is handing over to. It is one function rather than two because what it
    is defending is one invariant with a lot of edges -- a key is not an identity, the
    lock wait is unbounded, an id resolved before a suspension can name a store the
    slot has already replaced -- and a second reading of those edges would drift from
    this one the first time either verb changed. Every refusal below is therefore made
    once, for both verbs, in the order the adoption made them.

    *noun* names the operation in the refusals a caller reads ("adoption", "handoff").
    *operation* is the audit verb and the word the fence reads
    (:data:`HANDOFF_WIDENED_OPERATIONS`), so a verb's reach is decided by the gate and
    not here.

    *gate* is an extra synchronous authorization, run at the pre-lock gate AND again
    inside the lock right after the target's re-authorization -- the only safe place
    for a caller rule this function does not itself know, because the state it reads
    can move across the lock wait exactly as the fence's can. It must not suspend: the
    final authorization and the append have nothing between them on purpose.

    *commit* is the durable write a verb owes AFTER its entry is on disk, awaited
    inside the lock. Ordered that way because the tree entry is append-only and a
    field is not: a field written first and an append that then failed would leave the
    gateway acting on a move no record carries, while an entry written first and a
    field that then failed leaves the reach where it was, which is the fail-closed
    direction. :func:`adopt_target` passes none -- for an adoption the entry IS the
    whole operation.

    A *commit* that REFUSES is compensated: the parent edge this function already made
    durable is put back, durably, before the refusal reaches the caller. Without that
    the two halves of a handoff come apart -- the target hangs under the recipient while
    the giver is still its manager, and the giver's own ``release`` then answers
    ``not_parent``, so neither session can undo it. The commit must therefore refuse
    BEFORE it publishes anything, which is the contract :func:`handoff_target`'s commit
    keeps: every refusal in it precedes its write to ``_manager``.
    """
    # Warmed off-loop with NOTHING suspending before the gate, which is this function's
    # whole reason for existing: ``authorize_target`` is synchronous and its
    # ``session_control_enabled()`` re-reads and validates the config file on the first
    # call after an edit, so an unwarmed gate blocks the shared loop for every other
    # session. The pre-lock gate and the in-lock re-check each get their own warm, since
    # the lock wait between them is exactly the suspension that invalidates the first.
    await prewarm_enabled_check()
    slot = authorize_target(
        state,
        caller_session_key=caller_session_key,
        target=target,
        operation=operation,
        precomputed_ownership_fenced=caller_fenced,
    )
    with _audit_denials(
        caller_session_key=caller_session_key, operation=operation, slot_key=slot.key
    ):
        # INSIDE the audit block. The verb's own caller rule refuses on its own codes --
        # ``already_manager``, and a recipient that changed under its key -- and a
        # refusal nobody records is a refusal nobody can review: the SEL is where a
        # denied attempt to take over another session has to show up, and these are
        # exactly the attempts worth seeing.
        if gate is not None:
            gate(slot)
        caller_key = caller_slot_key(state, caller_session_key)
        from kiro_crew.crew_log import emit as crew_log_emit

        # The TARGET's object, captured before the first suspension, so the final
        # authorization can tell the session it admitted apart from a same-key
        # replacement. Everything about the target below -- the gate's re-resolve, the
        # ``_live_sid_of`` mapping read -- is keyed by ``slot.key``, and a key is not an
        # identity: the target can close and a NEW session can open under the same slot
        # key in the awaits ahead (the gate's warm, the lock wait, the id resolution's
        # store read), and every key-only check then answers for the replacement. The
        # move would be appended against a conversation that never took part, in an
        # entry nothing later corrects. The object itself does not move, so comparing it
        # after the gate is what makes the refusal below possible.
        target_at_resolve = _slot_object(state, slot.key)

        target_sid = _live_sid_of(state, slot.key)
        if not crew_log_emit.enabled() or not target_sid:
            # Checked FIRST because it is the PERMANENT one of the two refusals below: with
            # no recorded tree there is nowhere to write the edge and no later retry helps,
            # so a caller must not be told to come back. Reported rather than answered with a
            # success the tree will not show -- the record IS the edge, and a verb whose
            # record cannot be written has not done anything.
            raise SessionControlError(
                "the session tree is not being recorded on this gateway, so sessions "
                f"cannot take part in a {noun}",
                status=409,
                code="tree_unavailable",
            )
        async with _tree_mutation_lock():
            _refuse_if_append_pending(slot.key, operation=noun)
            if fence_parent:
                # BOTH endpoints, because the cycle check below reads the tree and the
                # tree does not yet carry an append that has not settled. The lock alone
                # does not cover this: a caller whose own wait is bounded unwinds out of
                # the lock with its append still queued, and the next mover through the
                # lock then asks ``is_ancestor`` about a tree missing that edge. Fencing
                # the target alone leaves the REVERSE move admitted -- it names the other
                # endpoint as its target, so the target-only fence never sees it -- and
                # when both land the tree holds a loop no fold can present and no verb
                # can undo, because each half is a valid entry in an append-only log.
                _refuse_if_append_pending(parent_key, operation=noun)
            tree_known, previous_parent, nodes = _slot_tree_parent(slot.key)
            if not tree_known:
                # REFUSED rather than moved past the guard. The cycle check below is the
                # only thing standing between a takeover and a loop the fold cannot present,
                # and a guard handed no tree admits everything -- so an unreadable tree has
                # to stop the write, not wave it through. Reachable on every gateway between
                # boot and the first lineage seed, which is why it is a refusal a caller can
                # retry rather than a condition worth blocking on: the seed is already being
                # requested by the sidebar's own read path.
                raise SessionControlError(
                    f"the session tree is not readable yet on this gateway, so a {noun} "
                    "cannot be checked for a loop -- retry in a moment",
                    status=409,
                    code="tree_not_ready",
                )
            # Deferred: ``holders`` reaches the storage package, and this module is on the
            # dashboard's boot path. The ancestor walk follows exactly the tree's own rules,
            # which is why it is imported rather than re-implemented -- a second walk with
            # its own reading of "a cited creator with no log" would refuse or admit cases
            # the fold does not.
            from kiro_crew.crew_log.holders import is_ancestor

            # The SETTLED tree plus every queued edge. The projection advances only when
            # an append settles, so a check against the settled tree alone cannot see a
            # move the writer is already holding -- and two such moves can close a loop
            # neither of them closes on its own. The endpoint fences above refuse the
            # case where this move shares a slot with a queued one; this is the case
            # where it shares none, which is the one they cannot reach.
            try:
                view = _tree_with_pending_edges(nodes)
            except Exception:
                # REFUSED rather than checked against the settled tree. "No loop" read
                # off a view that could not be built is a confident wrong answer, and
                # the whole guard below is only worth its own refusal.
                logger.warning(
                    "session %s: the queued tree edges could not be folded into the "
                    "tree, so the %s cannot be checked for a loop",
                    slot.key,
                    noun,
                    exc_info=True,
                )
                raise SessionControlError(
                    f"the queued session-tree writes could not be read, so a {noun} "
                    "cannot be checked for a loop -- retry in a moment",
                    status=409,
                    code="tree_not_ready",
                ) from None
            if is_ancestor(slot.key, parent_key, view):
                raise SessionControlError(
                    f"{target!r} is already above {parent_key!r} in the tree, so putting "
                    "it under that session would make a loop",
                    status=409,
                    code="would_cycle",
                )
            # Resolve ids before the final authorization because nothing may suspend between
            # that authorization and handing the append to the writer.
            #
            # The slot OBJECTS are captured first, before the resolution's own store read
            # suspends, so the identity check covers every suspension between reading an id
            # and writing it -- not just the gate's.
            parent_at_resolve = _slot_object(state, parent_key)
            previous_at_resolve = _slot_object(state, previous_parent) if previous_parent else None
            parent_sid = await _recorded_sid_of(state, parent_key)
            previous_parent_sid = (
                await _recorded_sid_of(state, previous_parent) if previous_parent else ""
            )
            # RE-AUTHORIZED here, and this is not the same question the pre-lock call
            # answered. That call decided on a state read before the wait, and the wait is
            # unbounded: the verb ahead in the queue awaits its own append, and anything the
            # gate reads can move meanwhile -- the target can be stopped or closed, the
            # caller's grant can be withdrawn, either side can gain a channel link. Writing
            # on the earlier answer would record a move nobody was entitled to at the
            # moment it landed. The pre-lock call is kept because it is the cheap refusal: an
            # unauthorized caller never contends for this lock at all.
            await prewarm_enabled_check()
            slot = authorize_target(
                state,
                caller_session_key=caller_session_key,
                target=target,
                operation=operation,
                precomputed_ownership_fenced=caller_fenced,
            )
            # The verb's OWN caller rule, re-asked here for the reason the fence above is:
            # a rule decided before the lock wait is a rule decided on stale state, and
            # this one can go stale in the way that matters most -- the giver can stop
            # being the target's manager inside the wait, because another handoff landed.
            if gate is not None:
                gate(slot)
            target_sid = _live_sid_of(state, slot.key)
            if not crew_log_emit.enabled() or not target_sid:
                raise SessionControlError(
                    "the session tree is not being recorded on this gateway, so sessions "
                    f"cannot take part in a {noun}",
                    status=409,
                    code="tree_unavailable",
                )
            # REFUSED -- not silently re-resolved -- when the slot under the target key is a
            # different session from the one the pre-lock gate admitted. The gate re-resolves BY
            # KEY and ``_live_sid_of`` reads the mapping BY KEY, so a close plus a same-key
            # reopen inside the awaits above leaves both answering for the replacement: an
            # open session that passes every fence, whose log ``target_sid`` now names. The
            # move would then be appended against a conversation that never took part, and
            # the entry is append-only -- there is no later write that corrects it. The
            # captured object is this process's identity for the admitted session; the
            # replacement is a different object under the same key, so the ``is`` comparison
            # is what the key-only checks cannot do. A caller may retry: a fresh attempt
            # resolves the current session honestly.
            if _target_replaced(state, slot.key, target_at_resolve):
                raise SessionControlError(
                    f"{target!r} was replaced by a new session under the same key, so the "
                    f"{noun} would name a session that never took part -- retry to act "
                    "on the current one",
                    status=409,
                    code="target_replaced",
                )
            # REFRESHED here, synchronously, for the same reason the resolutions happen
            # before the gate: the gate's own warm suspends, and either of these slots can
            # open its next store inside that hop -- leaving the id above naming the store
            # it has just replaced, in an entry nothing later corrects.
            parent_sid = _freshest_sid(state, parent_key, parent_sid, parent_at_resolve)
            previous_parent_sid = (
                _freshest_sid(state, previous_parent, previous_parent_sid, previous_at_resolve)
                if previous_parent
                else ""
            )
            # THE CYCLE CHECK THAT DECIDES, re-read here and synchronously. Every step
            # between the early check above and this line suspends -- the id
            # resolutions, the gate's own warm, the re-authorization -- and a detached
            # commit window belonging to an earlier verb can compensate inside that gap,
            # restoring the very edge the early check was asked about. The verdict formed
            # before it is then false when this append lands, and the log keeps two edges
            # that each passed on their own with a loop between them. Nothing awaits
            # between this refusal and the append below, which is the discipline the sid
            # refresh just above follows for the same reason.
            _known_now, _previous_now, _nodes_now = _slot_tree_parent(slot.key)
            if not _known_now:
                raise SessionControlError(
                    f"the session tree could not be read, so a {noun} cannot be checked "
                    "for a loop -- retry in a moment",
                    status=409,
                    code="tree_not_ready",
                )
            try:
                _view_now = _tree_with_pending_edges(_nodes_now)
            except Exception:
                logger.warning(
                    "session %s: the queued tree edges could not be folded into the "
                    "tree, so the %s cannot be re-checked for a loop before its append",
                    slot.key,
                    noun,
                    exc_info=True,
                )
                raise SessionControlError(
                    f"the queued session-tree writes could not be read, so a {noun} "
                    "cannot be checked for a loop -- retry in a moment",
                    status=409,
                    code="tree_not_ready",
                ) from None
            if is_ancestor(slot.key, parent_key, _view_now):
                raise SessionControlError(
                    f"{target!r} moved above {parent_key!r} while this {noun} was being "
                    "authorized, so applying it now would make a loop -- retry",
                    status=409,
                    code="would_cycle",
                )
            settled, landed = _tree_append_waiter(slot.key)
            # PUBLISHED as a queued edge, synchronously and before the emit, so the next
            # verb through the lock folds this move into its own cycle check. Both verbs
            # that reparent register here: a queued adoption closes a loop exactly as a
            # queued handoff does, and the check has to see every edge the writer holds,
            # not only the ones this verb wrote.
            _record_pending_edge(slot.key, parent_key, landed)
            # The OTHER endpoint's fence, registered with no await between it and the
            # waiter above so the pair goes up together, and released by the window
            # rather than here: the caller's wait is bounded and the window outlives it,
            # so a release tied to this frame would drop the fence at the one moment it
            # is needed. Self-clearing is not available for this one -- it guards no
            # append of its own -- which is why it is held and handed over explicitly.
            release_parent_fence = _hold_tree_fence(parent_key) if fence_parent else None
            crew_log_emit.on_session_adopted(
                target_sid,
                slot=slot.key,
                parent_slot=parent_key,
                parent_sid=parent_sid,
                previous_parent_slot=previous_parent,
                previous_parent_sid=previous_parent_sid,
                on_settled=settled,
            )

            # Inside the lock, so the next verb through it reads a tree that already carries
            # this decision -- the append's completion is also what advances the projection.
            # The commit runs AFTER this verb's own entry is on disk and still inside the
            # lock: a durable write a verb owes is ordered behind the record of it, and no
            # other mover may read the slot between the two. ``target_sid`` goes with it
            # because a verb that appends its OWN entry needs the log this one just wrote
            # to, resolved and refreshed by the code above. A commit that REFUSES puts the
            # parent back, which is why the two are awaited by one helper rather than in
            # sequence here -- see :func:`_landed_then_committed`.
            # Audited from INSIDE the window, not after this await. The caller's wait is
            # bounded and the window outlives it, so a commit that publishes after the
            # caller gave up would otherwise leave the SEL holding only the ``denied``
            # line ``_audit_denials`` wrote for that timeout -- the reach moved and the
            # audit says it did not. *after_commit* is the verb's own "it happened" work,
            # run in the same place and for the same reason.
            def _committed() -> None:
                _audit(
                    caller_session_key=caller_session_key,
                    operation=operation,
                    slot_key=slot.key,
                    outcome="allowed",
                    detail={"previous_parent": previous_parent or "none"},
                )
                if after_commit is not None:
                    after_commit()

            await _landed_then_committed(
                landed,
                state=state,
                slot=slot,
                target_sid=target_sid,
                commit=commit,
                noun=noun,
                parent_key=parent_key,
                parent_sid=parent_sid,
                previous_parent=previous_parent,
                previous_parent_sid=previous_parent_sid,
                on_committed=_committed,
                release_parent_fence=release_parent_fence,
            )
    # Slot keys only, and deliberately no title. A title is conversation content -- it
    # is generated from the session's own messages -- so returning one to an LLM caller
    # is an output boundary that would owe a redaction pass, and nothing here needs it:
    # the tool reply names the session by the key the caller already used.
    return {
        "target": slot.key,
        "parent": parent_key,
        "previous_parent": previous_parent,
        "caller": caller_key,
    }


async def adopt_target(
    state: "DashboardState",
    *,
    caller_session_key: str,
    target: str,
    caller_fenced: bool | None = None,
) -> "dict[str, Any]":
    """Take *target* over, so it hangs under the CALLING session in the tree.

    The takeover verb. The adopter is the caller, resolved from the connection rather
    than named in the arguments: a tool that let a caller nominate the parent would let
    one session rearrange another's tree, and nothing in the record would show which of
    them asked. :func:`handoff_target` is the verb for naming another session, and it
    is a different verb precisely because it has to be authorized differently.

    Adopting a session that ALREADY has a live parent is allowed, because that is the
    case the verb exists for -- one conductor taking over the workers another one
    opened -- and the parent it replaces is recorded on the entry.

    Refused when the target is an ANCESTOR of the caller. The tree would then hold a
    cycle, and a cycle is not a shape the fold can present: it marks every slot on it
    and nests none of them, so the visible result of allowing this would be a whole
    branch silently flattening. The fold guards itself again at fold time for the
    reordering a checkpoint plus a replayed tail can produce; this is the guard that
    refuses the caller rather than absorbing it.

    Refused too when the tree cannot be READ, which is a separate refusal and not a
    special case of the one above. A guard handed no tree admits everything, so an
    unseeded projection would let exactly the adoption this checks for through.

    Every other refusal is :func:`authorize_target`'s -- an unidentifiable or
    unattended caller, a target that is not open, the caller itself, an ephemeral,
    app-scoped, channel-linked or mirrored session on either side, a different
    workspace, and the ownership fence. They are not re-stated here, which is the
    point of routing this verb through the same gate as the others: a session that
    cannot be sent to is not one that can be adopted either.

    An adoption does NOT change who manages the target. The tree says where the
    sidebar puts a session; ``_manager`` says who may drive it, and only
    :func:`handoff_target` writes that -- so a conductor cannot widen its own reach by
    adopting, which is the reach it already had to have to adopt at all.
    """
    moved = await _reparent_target(
        state,
        caller_session_key=caller_session_key,
        target=target,
        parent_key=caller_slot_key(state, caller_session_key),
        operation="adopt",
        noun="adoption",
        caller_fenced=caller_fenced,
    )
    return {key: moved[key] for key in ("target", "parent", "previous_parent")}


async def release_target(
    state: "DashboardState",
    *,
    caller_session_key: str,
    target: str,
    caller_fenced: bool | None = None,
) -> "dict[str, Any]":
    """Let *target* go, so it stands on its own in the tree again.

    Two callers may do it and no others: the target's CURRENT parent, and the target
    itself. The parent because a holder may put down what it holds, and the target
    because a session that has been taken over must not need its holder's cooperation
    to get out -- a conductor that has stopped running would otherwise pin its workers
    under it for good.

    Refused for a target that is already a root. There is nothing to release, and
    writing the entry anyway would put a record of a change into a log where nothing
    changed.

    The self case is the one place this verb departs from
    :func:`authorize_target`'s rules, and it departs from exactly two of them. The
    self-target refusal is waived, since reaching yourself is not reaching a peer. So
    is the ownership fence, for the same reason and no further: that fence bounds a
    caller to the sessions it created, and the session it is itself was never another
    session's to protect. Every other refusal on both sides still applies.
    """
    caller_key = caller_slot_key(state, caller_session_key)
    # Warmed off-loop before the gate, for the reason the adoption warms it.
    await prewarm_enabled_check()
    slot = authorize_target(
        state,
        caller_session_key=caller_session_key,
        target=target,
        operation="release",
        precomputed_ownership_fenced=caller_fenced,
        allow_self=True,
    )
    with _audit_denials(
        caller_session_key=caller_session_key, operation="release", slot_key=slot.key
    ):
        releasing_self = slot.key == caller_key
        from kiro_crew.crew_log import emit as crew_log_emit

        # The TARGET's object, captured before the first suspension, for the reason the
        # adoption captures it: the gate re-resolves and ``_live_sid_of`` reads BY KEY, so
        # a close plus a same-key reopen inside the awaits ahead would leave both naming a
        # replacement session, and the release would be appended against a conversation
        # that was never released -- append-only, so nothing later corrects it. The
        # ``allow_self`` case is covered too: when the caller IS the target, this just
        # captures the caller's own live object, and a same-key reopen there means a
        # different session is now calling, which the final gate should likewise refuse.
        target_at_resolve = _slot_object(state, slot.key)

        target_sid = _live_sid_of(state, slot.key)
        if not crew_log_emit.enabled() or not target_sid:
            # The permanent refusal first, for the reason the adoption checks it first.
            raise SessionControlError(
                "the session tree is not being recorded on this gateway, so sessions "
                "cannot be released",
                status=409,
                code="tree_unavailable",
            )
        async with _tree_mutation_lock():
            _refuse_if_append_pending(slot.key, operation="release")
            tree_known, previous_parent, _ = _slot_tree_parent(slot.key)
            if not tree_known:
                # The same refusal an adoption makes, for the mirror reason: this verb
                # decides WHO may call it from the parent the tree reports, so an unreadable
                # tree cannot tell "you are not the parent" from "I cannot see who is".
                raise SessionControlError(
                    "the session tree is not readable yet on this gateway, so a release "
                    "cannot be authorized -- retry in a moment",
                    status=409,
                    code="tree_not_ready",
                )
            if not previous_parent:
                raise SessionControlError(
                    f"{target!r} has no parent to be released from",
                    status=409,
                    code="already_root",
                )
            if not releasing_self and previous_parent != caller_key:
                raise SessionControlError(
                    f"{target!r} hangs under another session, so only that session or "
                    f"{target!r} itself can release it",
                    status=403,
                    code="not_parent",
                )
            # Resolve the id before the final authorization because nothing may suspend
            # between that authorization and handing the append to the writer. The slot
            # object is captured first, for the reason the adoption captures its two.
            previous_at_resolve = _slot_object(state, previous_parent)
            previous_parent_sid = await _recorded_sid_of(state, previous_parent)
            # RE-AUTHORIZED inside the lock, for the reason the adoption re-authorizes:
            # the wait is unbounded and everything the gate reads can move during it.
            await prewarm_enabled_check()
            slot = authorize_target(
                state,
                caller_session_key=caller_session_key,
                target=target,
                operation="release",
                precomputed_ownership_fenced=caller_fenced,
                allow_self=True,
            )
            releasing_self = slot.key == caller_key
            target_sid = _live_sid_of(state, slot.key)
            if not crew_log_emit.enabled() or not target_sid:
                raise SessionControlError(
                    "the session tree is not being recorded on this gateway, so sessions "
                    "cannot be released",
                    status=409,
                    code="tree_unavailable",
                )
            # REFUSED -- not silently re-resolved -- when the slot under the target key is a
            # different session from the one the pre-lock gate admitted, for the reason the adoption
            # refuses: the gate and ``_live_sid_of`` both resolve BY KEY, so a same-key
            # reopen inside the awaits above leaves them answering for a replacement, and
            # the release would be appended against a session that was never released. The
            # captured object is this process's identity for the admitted session; a caller
            # may retry to act on the current one.
            if _target_replaced(state, slot.key, target_at_resolve):
                raise SessionControlError(
                    f"{target!r} was replaced by a new session under the same key, so the "
                    "release would name a session that was never released -- retry to act "
                    "on the current one",
                    status=409,
                    code="target_replaced",
                )
            # Refreshed here, synchronously, for the reason the adoption refreshes: the
            # gate's warm suspends, and the parent can open its next store inside it.
            previous_parent_sid = _freshest_sid(
                state, previous_parent, previous_parent_sid, previous_at_resolve
            )
            settled, landed = _tree_append_waiter(slot.key)
            crew_log_emit.on_session_released(
                target_sid,
                slot=slot.key,
                previous_parent_slot=previous_parent,
                previous_parent_sid=previous_parent_sid,
                on_settled=settled,
            )
            # Inside the lock, for the reason the adoption awaits inside it.
            await _tree_append_landed(landed, operation="release", target=slot.key)
    _audit(
        caller_session_key=caller_session_key,
        operation="release",
        slot_key=slot.key,
        outcome="allowed",
        detail={"previous_parent": previous_parent, "self": releasing_self},
    )
    # No title, for the reason the adoption returns none.
    return {
        "target": slot.key,
        "previous_parent": previous_parent,
    }


async def _record_handed_over(state: "DashboardState", slot: "_ChatSlot") -> bool:
    """Record *slot* in the handed-over index. Answers whether the index says so now.

    WHETHER a session has ever been handed over is a fact about authorization, so it
    lives where the entries live -- in the fenced crew log
    (:mod:`~kiro_crew.crew_log.handed_over`) -- and not on the slot's own metadata line.
    The line is the transcript's, and the transcript's writers include the session whose
    grant a handoff revoked: a field there is one the losing party can clear, and
    clearing it is worth more than forging a ``created_by``, because it hands a
    DELIBERATELY REVOKED authorization back to a party that already exists and is
    already authenticated rather than inventing a creator out of nothing.

    It is still CHEAP for the slots that were never handed over, which is nearly all of
    them: the index is one small file listing the distinct slots a handoff committed
    for, read once per process and cached, so an unindexed slot is decided by a set
    lookup and reads no history at all. That is the property the metadata marker was
    there for, kept, with the authority moved somewhere its subject cannot write.

    Off the loop, because the write is file IO under a lock. False on every way it can
    fail, and the caller then refuses the handoff rather than publishing it: a grant the
    index does not carry is one the next start will not read, so the recipient would
    hold a reach that vanishes at the next restart while the giver silently gets it
    back.
    """
    key = str(getattr(slot, "key", "") or "")
    if not key:
        return False
    try:
        from kiro_crew.crew_log.handed_over import record_handed_over

        return bool(await asyncio.to_thread(record_handed_over, key))
    except Exception:
        logger.warning(
            "session_handoff: %s could not be recorded in the handoff index",
            key,
            exc_info=True,
        )
        return False


def _was_handed_over(slot: Any) -> "tuple[bool, bool]":
    """``(handed_over, decided)`` for *slot*, from the handed-over index.

    The gate that decides whether this slot's own log has to be read at all. A slot the
    index does not name was never handed over, so its creator holds it and no history
    is read; a slot it names has its log read, which is the path the rebuild already
    walks.

    ``decided`` is False when the index EXISTS and could not be established, and every
    slot is then treated as unread rather than as unhanded -- see
    :mod:`~kiro_crew.crew_log.handed_over` for why the empty set is the unsafe default.

    Kept for the ONE caller that asks about a single slot it already holds
    (:func:`resolve_slot_manager`, from its own thread). A BUILD path never calls this:
    it takes one :func:`~kiro_crew.crew_log.handed_over.handed_over_slots` read off the
    loop and passes that value to every slot it builds, because it has many slots and
    one read answers for all of them.
    """
    key = str(getattr(slot, "key", "") or "")
    if not key:
        return (False, True)
    from kiro_crew.crew_log.handed_over import handed_over_slots

    index = handed_over_slots()
    return (index.names(key), index.decided)


async def resolve_slot_manager(
    state: "DashboardState",
    slot: "_ChatSlot",
    *,
    slot_manager: "Callable[[str, str], tuple[str, bool]] | None" = None,
    slot_manager_tab: "Callable[[str, str], str] | None" = None,
) -> bool:
    """Establish *slot*'s manager from the crew log, re-validated. Answers whether it moved.

        THE ONE WAY the field is ever established on a slot, and every path that builds one
        must call it before publishing: the boot reconstruction, a revive, a resume from
        history. A path that constructs a slot and publishes it without this leaves
        ``_manager`` empty, which reads as "the creator holds it" -- so a handoff SILENTLY
        REVERTS and the giver gets send, read, stop and close back on a session the log
        records as driven by someone else. That is a grant handed back to someone recorded as not
        having it, which is why this is one helper and not a check each path remembers.

        One function rather than a call added per path as each is found: four earlier rounds
        on this line each closed one hole of this shape, and the fifth would have been found
        the same way. A path that forgets this one is a path whose slot comes up unmanaged,
        which the tests below name directly.

    THREE OUTCOMES, and only two of them give the session back to its creator. The
        invariant is that ``_manager`` carries only a statement the log actually makes, and
        EMPTY is a statement -- "the creator holds this session".

        1. An authoritative read finds NO handoff recorded: empty, and the creator holds it.
           The log makes that statement by having nothing to say.
        2. A decided revert -- the ``to_creator`` entry form -- says it explicitly: empty,
           and the creator holds it.
        3. A recorded manager that fails :func:`_manager_still_admissible`, because that
           session is closed or rebound to another principal's store: manager-dependent
           access is DENIED. Not the recipient, which cannot hold it, and NOT the giver
           either. "The recipient is not admissible" and "no handoff happened" are
           different facts, and only the second one means the creator still holds it;
           reverting here would invent a transfer back the log never recorded and hand a
           revoked grant to the session that gave it away.

        An UNDECIDED read is none of the three and establishes nothing, so the slot stays
        unresolved and denied until a read succeeds.

        Denying in case 3 strands nobody. An unfenced caller's reach never came from this
        field, so a person can still adopt, release, stop or close the session; the only
        caller that loses anything is the agent that chose to hand it over. The way back is
        a fresh handoff, which is the statement the log is missing.

        UNRESOLVED is its own state and never collapses into a decided one. An undecided read
        -- a unit that would not open, a store that could not be listed, an id this reader
        cannot account for -- establishes nothing, so the slot keeps
        ``_manager_unresolved`` and session control refuses the verbs whose authorization
        reads the manager until an authoritative read succeeds. Treating that read as "no
        handoff" instead is the one direction that lets a revoked creator back in: empty
        means the creator holds it, which is a CLAIM, and a read that failed cannot make it.
        A decided answer clears the flag, including a decided EMPTY one -- an entry-less log
        is a fact about the slot -- and so does a store that was never written, where no
        handoff can ever have been recorded.

        WHETHER THE LOG CAN BE READ IS THE READER'S ANSWER TO GIVE, not a flag consulted
        here. :func:`~kiro_crew.crew_log.session_tree.slot_manager` already separates "the
        store is not there, so nothing was ever recorded" from "the store is there and
        could not be read", which are the decided and undecided cases this needs. Emission
        being switched off says neither: it stops new entries and leaves every entry
        already on disk exactly where it is. Reading it as "no handoff is recorded" is the
        one inference that turns an operator's config change into a revoked giver getting
        its reach back, and the index is what makes that reachable -- an INDEXED slot is
        one whose handoff committed, so "no records" is contradicted by the index.

        *slot_manager* and *slot_manager_tab* are the boot pass's own resolved readers,
        passed so a pass over many slots imports once. A caller that omits them gets them
        resolved here, which is what a single revive wants.

        AN UNINDEXED SLOT IS DECIDED WITHOUT READING THE LOG. The log is authoritative
        only for a slot the crew log's handed-over index names, which the handoff records
        after its entry lands and before it publishes. Not named means no handoff was ever
        committed, so the creator holds the session -- outcome 1 above, established from a
        cached set rather than from history. That is what keeps a restore from reading
        anything for the sessions that were never handed over, which is nearly all of
        them. The index and not the slot's own line, because the line is the transcript's
        and the party a handoff revokes can write it.

        THE ANSWER IS ONLY EVER PUBLISHED IF IT IS STILL THE NEWEST ONE. Both reads below
        happen off the loop, and a handoff can commit inside either window -- its entry
        lands, its marker lands, its manager is published -- which leaves this function
        holding an answer that was true when it was read and revoked by the time it
        returns. So the revision is taken before the first read and handed back at the
        publish, and a publication whose revision has moved is discarded. Not a retry: a
        read that lost a race has nothing to add, and whatever moved the revision is
        either the authority itself or a later read of the same log.

        It is also asked PER INCARNATION. A slot key is a name, and a closed session's
        name can be taken by a new one; the grant names the tab identity it was written
        for, so a replacement under the same name finds no entry of its own and resolves
        as creator-held rather than inheriting a reach nobody gave it. A slot carrying no
        identity cannot be matched against a record and stays unresolved. The INDEX is
        keyed by slot alone, which is why it only ever says "read this slot's log": the
        entries inside it are what tell the incarnations apart.
    """
    key = str(getattr(slot, "key", "") or "")
    if not key:
        return False
    #: Read BEFORE THE FIRST SUSPENSION, which is the index read below -- not before
    #: the log read further down. Every publication in this function is derived from a
    #: read that happened off the loop, so every one of them hands this back and is
    #: discarded if the fact moved meanwhile. Capturing it after the index read would
    #: leave the two earliest branches unguarded, and those are the ones that publish a
    #: CLEARED statement: a handoff committing during that read would have its manager
    #: and its flags wiped by an older "never handed over" verdict.
    revision = manager_revision(slot)
    handed_over, index_decided = await asyncio.to_thread(_was_handed_over, slot)
    if not index_decided:
        # The index exists and could not be established, so NOTHING is known about any
        # slot -- including this one. Refusing is the only answer available: the empty
        # set reads as "never handed over", which would serve every slot from its
        # creator and is exactly the answer a revoked party wants.
        if not publish_manager(
            slot, manager="", manager_tab="", unresolved=True, expect_revision=revision
        ):
            return _discarded(key, revision)
        logger.warning(
            "manager resolve: the handoff index could not be read, so whether %s was "
            "ever handed over is UNKNOWN and session control refuses the verbs that "
            "depend on its manager until the index can be read",
            key,
        )
        return False
    if not handed_over:
        standing = str(getattr(slot, "_manager", "") or "")
        if not publish_manager(slot, manager="", manager_tab="", expect_revision=revision):
            return _discarded(key, revision)
        return bool(standing)
    #: WHICH INCARNATION of this name is asking. A record written for another one is
    #: not an answer about this session, and an empty identity cannot be matched
    #: against any record, which leaves the slot unresolved rather than guessing.
    tab = str(getattr(slot, "_tab_id", "") or "")
    if slot_manager is None:
        try:
            from kiro_crew.crew_log.session_tree import slot_manager as _reader
            from kiro_crew.crew_log.session_tree import slot_manager_tab as _tab_reader

            slot_manager = _reader
            slot_manager_tab = _tab_reader
        except Exception:
            logger.debug("manager resolve: the crew log is not available", exc_info=True)

    standing = str(getattr(slot, "_manager", "") or "")
    manager, decided = "", False
    if slot_manager is not None:
        try:
            manager, decided = await asyncio.to_thread(slot_manager, key, tab)
        except Exception:
            logger.debug("manager resolve: %s could not be read", key, exc_info=True)
            manager, decided = "", False
    if not decided:
        # UNRESOLVED, and left that way. The read established nothing, so there is no
        # answer to publish -- and "empty" is not the absence of an answer, it is the
        # answer "the creator holds it", which this read cannot support. The slot stays
        # denied for the verbs that depend on the manager until a read succeeds.
        if not publish_manager(
            slot, manager="", manager_tab="", unresolved=True, expect_revision=revision
        ):
            return _discarded(key, revision)
        logger.warning(
            "manager resolve: %s could not be read authoritatively, so it stays "
            "unresolved and session control refuses the verbs that depend on it",
            key,
        )
        return False
    if manager and manager != key:
        authority = manager
    else:
        # A decided revert, no entry at all, or an entry naming the slot itself: the
        # creator holds it. A self-manager is not a shape the verb can write, so a log
        # that states one is not trusted to state anything here.
        authority = ""
    # THE OCCUPANCY THE ENTRY RECORDED, read before ANY publish and before the
    # admissibility branch, because both branches publish and both must publish the
    # recorded value. Without it a restored manager is a bare name, and the key-to-key
    # compare then admits whoever answers to that name now -- the same hole the live
    # publish closes, closed with the same field and checked by the same function.
    #
    # Awaited here and not between two writes, which is the point of a single publish
    # per branch: an occupancy read is store IO, so the slot keeps its previous WHOLE
    # statement across the suspension instead of reading as resolved with no key yet
    # written. That partial state was found four times on this line, once per write
    # site, before the sites became one.
    occupancy = ""
    if authority and slot_manager_tab is not None:
        try:
            occupancy = str(await asyncio.to_thread(slot_manager_tab, key, tab) or "")
        except Exception:
            logger.debug("manager resolve: %s occupancy could not be read", key, exc_info=True)
            occupancy = ""
    # No manager published, or no occupancy reader on this seam, leaves this empty --
    # UNKNOWN occupancy, which refuses rather than matching whoever holds the name.
    if authority:
        refusal = _manager_still_admissible(state, slot, authority)
        if refusal is not None:
            # DENIED, and specifically not reverted to the creator. The invariant is
            # that ``_manager`` only ever carries a statement the log actually makes,
            # and empty IS a statement -- "the creator holds this session". The log
            # makes that statement in exactly two ways: it records no handoff at all,
            # or it records an explicit ``to_creator`` transfer back. A recorded
            # manager that cannot be published makes NEITHER. Reverting to the creator
            # there invents the one statement that hands a revoked grant back to the
            # session that gave it away, which is the direction this verb exists to
            # prevent.
            #
            # Denying is safe because an unfenced caller's reach never came from this
            # field: a person can still adopt, release, stop or close the session. The
            # only caller that loses anything is the agent that chose to hand it over.
            logger.warning(
                "manager resolve: the crew log records %s as the manager of %s and that "
                "grant is not admissible now (%s), so manager-dependent access to %s is "
                "DENIED until a fresh handoff records who holds it -- the creator does "
                "not get it back, because the log never recorded a transfer back",
                authority,
                key,
                refusal,
                key,
            )
            # The field carries WHAT THE LOG RECORDS, which is this inadmissible
            # manager; the flag carries that the record cannot authorize. Publishing it
            # invents nothing in either direction -- every reader that authorizes goes
            # through ``require_manager_of`` and is refused on the flag, while the
            # handoff that fixes this can state truthfully what it is replacing.
            #
            # ``unresolved`` is False here: this read succeeded. The two flags are
            # mutually exclusive by construction -- one says the value was never read,
            # the other says it was read and may not be acted on -- and both make every
            # authorization reader refuse, which is why the refusal checks both.
            #
            # The occupancy is the RECORDED one, like the key beside it. Stamping the
            # live occupant of that key here instead would make this refusal path write
            # a stranger's identity into the record: a fenced replacement sitting on a
            # retired manager's key gets its own tab published, a failed recovery
            # handoff copies that into its correction, and the log then names the
            # replacement -- which needs only a store rebind to become control nobody
            # handed over. A path that REFUSES must never rewrite what it refuses.
            if not publish_manager(
                slot,
                manager=authority,
                manager_tab=occupancy,
                inadmissible=True,
                expect_revision=revision,
            ):
                return _discarded(key, revision)
            return bool(standing != authority)
    if standing and not authority:
        logger.info(
            "manager resolve: %s came up holding %s and the crew log records no handoff "
            "at all, so that record is retired and the creator holds it",
            key,
            standing,
        )
    # RESOLVED: an authoritative read answered and the answer is publishable, so the
    # slot is servable. Both flags clear, in the same statement as the value they
    # qualify: a decided, admissible answer retires an earlier inadmissible one, which
    # is how a fresh handoff or a ``to_creator`` record brings a denied slot back.
    if not publish_manager(
        slot, manager=authority, manager_tab=occupancy, expect_revision=revision
    ):
        return _discarded(key, revision)
    return standing != authority


def _discarded(key: str, revision: int) -> bool:
    """Log a publication this read lost the race for. Always answers False.

    False is "nothing moved HERE", which is the truth: the fact was stated by someone
    else while this read was off the loop, and that statement stands. Deliberately not
    a retry -- a read that lost to a handoff would re-read the same log the handoff
    already wrote, and one that lost to a later read of that log has nothing to add.
    """
    logger.info(
        "manager resolve: %s was stated by another writer while this read was in "
        "flight (revision moved past %d), so this answer is discarded rather than "
        "published over the newer one",
        key,
        revision,
    )
    return False


def _manager_still_admissible(
    state: "DashboardState", slot: "_ChatSlot", manager_key: str
) -> str | None:
    """Why *manager_key* may NOT be published as *slot*'s manager, or ``None``.

    The admission re-asked at the point a RECORD is applied. The handoff checks the
    recipient immediately before it publishes, and that check is the whole reason a
    member's private surface or another principal's store never gains a session -- but
    it is a check on the state at that moment, and the entry it guards outlives the
    process that wrote it. Between the append and the next start a recipient can be
    rebound to another principal's store, linked to a channel, scoped to an app or
    closed, and nothing in an append-only log can know it.

    So the log stays the authority on WHAT was recorded and this answers whether that
    record is still ALLOWED. Each rule is the one the admission applied, read off the
    slots themselves:

    * a crew member's own DM surface is refused by key, as it is at admission: it is
      the member's operating surface, not a session work runs in;
    * the manager has to still be a live session, because a key names nothing
      otherwise and an absent session cannot drive anything;
    * the two slots must AGREE on their memory store, which is the positive principal
      test -- a handed session and the session driving it run under one principal, and
      a store rebound while the gateway was down is exactly the drift this catches;
    * the manager must not be a session that cannot be reached at all
      (:func:`_live_target_refusal`), the same target-side containment the admission
      applied to the recipient.

    FAIL CLOSED: a reason that cannot be determined is a reason. This is strictly
    stronger than the verb's corrective append, which is a best-effort fast path --
    a correction that never landed, or a crash between the entry and its correction,
    leaves the log stating a refused grant, and this is the one place that grant would
    otherwise take effect.
    """
    from kiro_crew.members import DM_SLOT_KEY_PREFIX

    try:
        # The DM prefix FIRST, because it is the one rule that needs no session to read:
        # it is part of the key, so it answers even for a surface this gateway has not
        # opened yet -- and it answers with the reason that is actually true.
        if manager_key.startswith(DM_SLOT_KEY_PREFIX):
            return "a crew member's own session cannot manage another"
        live = state.get_slot(manager_key)
        if live is None:
            return "no session is open under that key"
        managed_store = str(getattr(slot, "memory_store", "") or "")
        if str(getattr(live, "memory_store", "") or "") != managed_store:
            return "it is bound to a different memory store, so it runs under another principal"
        refusal = _live_target_refusal(live)
        if refusal is not None:
            return refusal[0]
    except Exception:
        logger.debug(
            "manager rebuild: %s could not be re-validated for %s",
            manager_key,
            getattr(slot, "key", ""),
            exc_info=True,
        )
        return "its admissibility could not be determined"
    return None


async def rebuild_managers_from_log(state: "DashboardState") -> int:
    """RESTORE the manager of every live slot that was handed over. Answers how many moved.

    The read half of the handoff's storage model. The crew log is the only store for WHO
    manages a session; the crew log's handed-over index says only WHICH slots a handoff
    was committed for, and that index is what scopes this pass. A slot it does not name was
    never handed over: it is decided at hydrate as held by its creator and
    :func:`resolve_slot_manager` returns for it without reading any history. Only an
    INDEXED slot comes up unresolved, is refused the manager-dependent verbs, and waits
    on its own log -- so the work here is proportional to the sessions that were actually
    handed over, not to every restored tab.

    For a marked slot the log decides, in three directions, and only two of them hand the
    session to its creator: an entry-less log or an explicit ``to_creator`` record does;
    a recorded manager that fails revalidation leaves manager-dependent access DENIED,
    because "the recipient is not admissible" is not "no handoff happened". An UNDECIDED
    read establishes nothing and leaves the slot unresolved; a refusal arms the re-read,
    so it clears without a restart.

    No process-wide window. Every slot that could be guessed wrong is a marked slot, and
    every marked slot is refused on its own until its read lands, so there is no state
    in which an unread value can be answered from ``_created_by``.

    Hopped off the loop one slot at a time -- the restore driver's own shape -- so a user
    with many tabs does not hold the loop long enough to trip the stall watchdog.
    """
    try:
        from kiro_crew.crew_log.session_tree import slot_manager, slot_manager_tab
    except Exception:
        logger.debug("manager rebuild: the crew log is not available", exc_info=True)

        def slot_manager(_slot: str, _tab: str) -> "tuple[str, bool]":  # type: ignore[misc]
            # UNDECIDED, which leaves every marked slot refused until a read lands. A
            # reader that cannot even be imported has not established that nothing was
            # recorded, and the marker says something was.
            return ("", False)

        def slot_manager_tab(_slot: str, _tab: str) -> str:  # type: ignore[misc]
            return ""

    return await _restore_managers(state, slot_manager, slot_manager_tab)


async def _restore_managers(
    state: "DashboardState",
    slot_manager: "Callable[[str, str], tuple[str, bool]]",
    slot_manager_tab: "Callable[[str, str], str]",
) -> int:
    """The per-slot pass itself. Split out so the gate's release is one ``finally``."""
    moved = 0
    for slot in list(getattr(state, "_slots", {}).values()):
        if await resolve_slot_manager(
            state,
            slot,
            slot_manager=slot_manager,
            slot_manager_tab=slot_manager_tab,
        ):
            moved += 1
    if moved:
        logger.info("Restored the manager of %d session(s) from the crew log", moved)
        # Broadcast, because this pass runs OUTSIDE the restore's own push suspension:
        # it is its own task off the boot path, so the coalesced boot broadcast has
        # already gone out by the time a manager moves here.
        try:
            state.push_slots_update()
        except Exception:
            logger.debug("manager rebuild: the sidebar push failed", exc_info=True)
    # The line ABOVE a closed manager, brought back in the SAME pass and inside the same
    # window, so there is no second rebuild and no second gate to keep in step.
    await _rebuild_departures_from_log(state)
    return moved


#: "This cannot be decided without reading the disk", from the synchronous half of the
#: restored-edge admission. Distinct from ``None``, which is a verdict -- the edge is
#: admissible -- so the two cannot be confused by a caller that only tests for falsey.
_UNDECIDED: Any = object()


def managed_store(persisted: "dict[str, Any]") -> str:
    """The principal a closed session's persisted line records, or ``""``."""
    return str(persisted.get("memory_store") or "")


def _restored_edge_refusal_now(
    state: "DashboardState", manager: str, store: str
) -> "str | None | Any":
    """The part of the restored-edge admission that needs no disk, or :data:`_UNDECIDED`.

    Split out because it has to run TWICE and in two different positions: once while the
    edge is being judged, and once more immediately before the edge is written, after
    the last await. Everything it reads -- the live table, a live slot's own store -- can
    change across a suspension, and this rebuild runs as a background task on a gateway
    that is already serving. A judgement made before an await and acted on after it is
    the whole shape of this feature's findings.

    :data:`_UNDECIDED` means the manager is not in the live table, so its principal is
    on disk and only the caller that may suspend can answer.
    """
    from kiro_crew.members import DM_SLOT_KEY_PREFIX

    if manager.startswith(DM_SLOT_KEY_PREFIX):
        return "a crew member's own session cannot manage another"
    live = state.get_slot(manager)
    if live is None:
        return _UNDECIDED
    refusal = _live_target_refusal(live)
    if refusal is not None:
        return refusal[0]
    if str(getattr(live, "memory_store", "") or "") != store:
        return "it is bound to a different memory store, so it runs under another principal"
    return None


async def _restored_edge_refusal(
    state: "DashboardState", key: str, persisted: "dict[str, Any]", manager: str
) -> str | None:
    """Why a rebuilt edge from *key* to *manager* may not be published, or ``None``.

    The admission the boot rebuild was missing, asked for an edge rather than for a
    live slot. Two sessions joined by a handoff run under ONE principal -- that is the
    positive test the verb applies before it appends -- and between the append and this
    restart the recipient can have been rebound to another store. The log cannot know
    that, so the log is the authority on WHAT was recorded and this answers whether the
    record may still take effect.

    The manager's own principal comes from its live slot when it has one and from its
    persisted line when it does not, because a departure edge may legitimately name a
    hop that is itself closed. A principal that cannot be read is a refusal: a fact this
    gateway has lost is not a fact in the caller's favour.
    """
    sync_answer = _restored_edge_refusal_now(state, manager, managed_store(persisted))
    if sync_answer != _UNDECIDED:
        return sync_answer
    if True:
        try:
            above_meta, above_readable = await asyncio.to_thread(
                state.conversation_log.get_metadata_status, _history_key_for(manager)
            )
        except Exception:
            logger.debug("departure rebuild: %s principal unreadable", manager, exc_info=True)
            return "its memory binding could not be read"
        if not above_readable:
            return "its memory binding could not be read"
        if str(above_meta.get("memory_store") or "") != managed_store(persisted):
            return "it is bound to a different memory store, so it runs under another principal"
    return None


async def _rebuild_departures_from_log(state: "DashboardState") -> int:
    """Bring back what each CLOSED manager reported to. Answers how many came back.

    Without this the hand-up is process-local, and a restart undoes exactly the thing
    the feature is for: a conductor closes, its workers name that closed conductor as
    their manager, and falling back to the creation chain strands them -- which is the
    bug this feature exists to fix. A feature that undoes its own purpose on the next
    restart is not one anybody can turn on.

    NO NEW DURABLE WRITE IS NEEDED, which is the whole reason this is small. A handoff
    already publishes ``session/manager_set`` only once that entry is on disk, and the
    entry outlives the session's close -- so what a departed session reported to is
    already recorded, by the base PR, in the form
    :func:`~kiro_crew.crew_log.session_tree.slot_manager` reads. The in-memory record
    was never the storage; it was a cache of a read nobody was doing for closed keys.

    WHICH keys to read is the only real question, because the log is not enumerable by
    slot. It does not have to be: the walk only ever meets a departed hop that some LIVE
    slot's manager edge names, so the keys worth reading are reachable from the live
    table by following those edges upward. That is the same input
    :func:`rebuild_managers_from_log` iterates, one step further.

    Bounded three ways: the depth bound per chain, :data:`_KEY_OCCUPANCY_MAX` overall,
    and one key read at a time hopped off the loop -- the restore driver's own shape, so
    a gateway with many tabs does not hold the loop long enough to trip the stall
    watchdog. An undecided read records NOTHING and leaves the hop unresolved, because a
    read that may have missed an entry must not be written down as a fact.
    """
    if not await drained_to_thread(manager_chain_enabled):
        # OFF: every reader of these records is behind the switch, so this pass is a
        # disk read per closed hop at a boot whose output nothing consults. The window
        # it closes only matters once the line is on, and a gap already refuses.
        _departures_rebuilt[0] = True
        return 0
    note_chain_switch(True)
    try:
        from kiro_crew.crew_log import emit as crew_log_emit
        from kiro_crew.crew_log.schema import CrewLogError
        from kiro_crew.crew_log.session_tree import slot_manager, slot_manager_tab
        from kiro_crew.crew_log.store import session_units_for_slot

        if not crew_log_emit.enabled():
            # Nothing will ever restore a closed hop's line on this gateway, so a gap
            # is FINAL rather than not-read-yet, and the window closes here.
            _departures_rebuilt[0] = True
            return 0
    except Exception:
        logger.debug("departure rebuild: the crew log is not available", exc_info=True)
        _departures_rebuilt[0] = True
        return 0

    pending = [manager_of(slot) for slot in list(getattr(state, "_slots", {}).values())]
    seen: set[str] = set()
    brought_back = 0
    for _ in range(LINEAGE_WALK_MAX_DEPTH):
        if not pending or len(seen) >= _KEY_OCCUPANCY_MAX:
            break
        climbing: list[str] = []
        for key in pending:
            if not key or key in seen or len(seen) >= _KEY_OCCUPANCY_MAX:
                continue
            seen.add(key)
            if state.get_slot(key) is not None:
                # Live: answered from the table, and its own manager edge is the next
                # hop worth following.
                climbing.append(manager_of(state.get_slot(key)))
                continue
            # THE CLOSED SESSION'S OWN PERSISTED LINE, FIRST, because the log reads
            # below are keyed on the session that ran under this name -- the log holds a
            # unit per session id and answers for the one it is asked about. Five things
            # needed this line and each was a separate finding: which session to ask the
            # log about, what a decided-empty answer MEANS, which principal the closed
            # session ran under, whether the recorded grant is still admissible, and
            # whether any of that could be read at all. One read answers all five, and a
            # read that did not land answers none of them -- so nothing is recorded.
            try:
                persisted, persisted_readable = await asyncio.to_thread(
                    state.conversation_log.get_metadata_status, _history_key_for(key)
                )
            except Exception:
                logger.debug("departure rebuild: %s metadata unreadable", key, exc_info=True)
                persisted, persisted_readable = {}, False
            if not persisted_readable:
                # A FORGOTTEN FACT IS NOT A FACT IN YOUR FAVOUR. The line this key
                # reported to cannot be established, so the hop stays unresolved and
                # every walk through it refuses -- which is the reach a gateway has
                # without this feature at all.
                logger.info(
                    "departure rebuild: %s has no readable metadata line, so its line "
                    "is left unresolved and reach across it is refused",
                    key,
                )
                note_hop_left_unreadable(state, key)
                continue
            departed_tab = str(persisted.get("tab_id") or "")
            if not departed_tab:
                # The log cannot be asked about a session this line does not name, and
                # an identity nobody can state matches nobody. Both answers are the
                # same refusal, so the hop is left unresolved.
                logger.info(
                    "departure rebuild: %s names no session on its line, so the log "
                    "cannot be asked about it and reach across it is refused",
                    key,
                )
                note_hop_left_unreadable(state, key)
                continue
            try:
                manager, decided = await asyncio.to_thread(slot_manager, key, departed_tab)
            except (CrewLogError, OSError, ValueError):
                # NARROW, and deliberately. A handler wide enough to catch a TypeError
                # turns a signature that stopped matching into "there is nothing to
                # restore" -- which is this line's own recurring defect, an absence read
                # as a safe answer, and it hid exactly that for a whole round: every
                # call raised, every exception was swallowed, and no departed line was
                # restored in production while the tests stayed green. A read that
                # cannot be MADE is a bug and must reach the log and the operator; only
                # the ways a read can legitimately fail are absorbed here.
                logger.debug("departure rebuild: %s could not be read", key, exc_info=True)
                note_hop_left_unreadable(state, key)
                continue
            if not decided:
                # UNRESOLVED, and said out loud: an undecided read is not the fact that
                # this session reported to nobody, and only the mark keeps the two
                # apart for the handoff's loop guard.
                note_hop_left_unreadable(state, key)
                continue
            try:
                await asyncio.to_thread(session_units_for_slot, key)
                units_known = True
            except (CrewLogError, OSError):
                # A read that FAILED is not the fact that the key had one occupant.
                # Treating it as one would file the entry without a tombstone, and the
                # walk would then resolve an edge naming this key confidently -- which
                # is the direction that grants reach rather than costing it. So an
                # unreadable unit list counts as REUSED: the conservative answer is the
                # one that refuses.
                logger.debug("departure rebuild: %s units unreadable", key, exc_info=True)
                units_known = False
            # MORE THAN ONE unit for a key means the key has run under several session
            # ids -- which is reuse, the same fact the in-process tombstone carries, and
            # the reason it has to be derived here too: a restart must not forget that a
            # key is ambiguous and start resolving its edges confidently again.
            # UNIT COUNT IS NOT REUSE. A slot gets a new unit on a reset, an agent or
            # model switch and a provider swap -- the reader says so itself -- while its
            # ``_tab_id`` does not change. Counting units therefore tombstoned every
            # closed conductor that had ever been reset, and the lead's reach over its
            # workers came back from a restart refused as ambiguous.
            #
            # What a restart needs is the identity, and the persisted line carries it.
            # A later DIFFERENT occupant of the key is still caught: taking the key
            # writes a tombstone from this record's occupant, and an edge naming the key
            # is compared against it. Only an unreadable list is treated as reuse, for
            # the reason it always was -- a read that did not land is not the fact that
            # the key had one occupant.
            reused = not units_known
            # The OCCUPANCY the log recorded with that manager, read through the base
            # PR's own companion reader so this pass and the live check cannot disagree
            # about where the value comes from. Empty is unverifiable and refuses, which
            # is what the live check does with it too.
            try:
                recorded_tab = await asyncio.to_thread(slot_manager_tab, key, departed_tab)
            except (CrewLogError, OSError, ValueError):
                logger.debug("departure rebuild: %s manager tab unreadable", key, exc_info=True)
                recorded_tab = ""
            # WHETHER THE LOG NAMED A MANAGER AT ALL, read before the substitution
            # below replaces an empty answer with the creator. The two cases carry
            # different identities: a published handoff names an occupant, and a
            # creation edge names none and is not judged on identity -- the base PR
            # states that is the same key-to-key question ``_created_by`` has always
            # had. Collapsed, a restored creation edge would read as a handoff whose
            # occupancy is unknown and refuse the most ordinary line there is.
            handed = bool(manager)
            if not manager:
                # DECIDED AND EMPTY is a statement, not a blank. ``slot_manager``
                # answering ``("", True)`` means the log was read and records no
                # handoff, which is "the session's CREATOR manages it" -- the same
                # thing an empty ``_manager`` means on a live slot. Recorded as
                # ``""`` it would mean something else entirely, "reports to nobody",
                # and the walk would stop at this hop: the most ordinary shape there
                # is -- a lead opens a conductor, the conductor opens a worker, nobody
                # hands anything over, the conductor closes -- would lose the lead its
                # reach at every restart, which is the behaviour this pass exists to
                # restore.
                #
                # So the creator is read from the same persisted line, and a line that
                # names no creator really is a session with nobody above it.
                manager = str(persisted.get("created_by") or "")
            if manager:
                # THE ADMISSION, RE-ASKED, which is the one check this pass was
                # publishing grants without. The verb applies it to a live recipient
                # immediately before it appends, and the base PR re-applies it when it
                # restores a LIVE slot's manager -- but a departure edge grants reach
                # over everything below a closed hop just the same, and between the
                # append and this restart the recipient can have been rebound to
                # another principal's store, closed or scoped away.
                #
                # The PRINCIPAL half of that admission, and deliberately not the
                # whole of it: ``_manager_still_admissible`` also requires the manager
                # to be a live session, which is right for a field that has to name
                # something drivable now and wrong for a step in a line, because the
                # hop above a departed hop may be departed too -- the chained departure
                # this feature exists to read through. Borrowing the rule whole would
                # refuse exactly the case the pass is for, so what is reused is the
                # test that decides a principal and the one that refuses a member's own
                # surface, and the liveness condition is left where it belongs.
                refusal = await _restored_edge_refusal(state, key, persisted, manager)
                if refusal is not None:
                    logger.info(
                        "departure rebuild: the line above %s names %s, which is not "
                        "admissible (%s), so nothing is recorded and reach across it "
                        "is refused",
                        key,
                        manager,
                        refusal,
                    )
                    # The log DOES name a hop above this one; what is refused is
                    # publishing it. So this is not a line that ended, and a writer
                    # must not read it as one.
                    note_hop_left_unreadable(state, key)
                    continue
            # EVERY VOLATILE CONDITION, RE-ASKED HERE, with no suspension between this
            # block and the write below it. This is the position that matters and the
            # earlier placement was the bug: each check was correct where it stood and
            # stood before a ``to_thread`` read, while the rebuild runs as a background
            # task on a gateway that is already serving. Between the read and the write
            # a person can resume this very key, and the recipient this edge names can
            # be reopened under a different principal.
            #
            # Taking a key an occupant already holds does not merely add a stale entry:
            # it moves the key's occupancy off that occupant, which silently disables
            # every departure the occupant later records (each checks its own generation
            # against the key's) and replaces the mark its construction wrote with an
            # entry naming whatever manager the LOG remembers -- an edge nothing can
            # retract. And a recipient that came back on another store is exactly the
            # grant the admission above refuses; answered before it was reopened, that
            # refusal describes a session that no longer exists.
            if state.get_slot(key) is not None or key in _key_occupancy:
                logger.info(
                    "departure rebuild: %s was taken while its line was being read, so "
                    "the live occupant keeps the key and nothing is recorded for it",
                    key,
                )
                continue
            if manager:
                settled = _restored_edge_refusal_now(state, manager, managed_store(persisted))
                if settled is not _UNDECIDED and settled is not None:
                    logger.info(
                        "departure rebuild: the line above %s names %s, which stopped "
                        "being admissible while its line was read (%s), so nothing is "
                        "recorded and reach across it is refused",
                        key,
                        manager,
                        settled,
                    )
                    note_hop_left_unreadable(state, key)
                    continue
            generation = claim_occupancy_for_rebuilt_key(key)
            # The rebuilt entry carries the identity its own line records and nothing
            # it cannot establish. A key the log shows running under exactly ONE session
            # names that session, so the hop can be told apart from a later occupant of
            # the same key and a resume under the same id is recognised; a key that ran
            # under SEVERAL is refused outright with an empty identity, because the units
            # say the name changed hands and nothing says which occupant an edge meant.
            #
            # ``manager_tab`` is what the departed session recorded about ITS manager,
            # so the hop above this one is judged by the same rule a live edge is.
            # THE PERSISTED TAB, not a unit id. A unit is named by an ACP session id,
            # and every comparison this record feeds is against ``_tab_id`` values --
            # so recording a unit id made every rebuilt handed-over hop read as a
            # different session and refuse. The identity has to come from the same
            # space the live check reads.
            identity = "" if reused else departed_tab
            if _record_departure(
                key,
                generation,
                manager,
                pending=False,
                tab=identity,
                occupant=identity,
                # ``recorded_tab`` even when it is EMPTY, because empty is what a
                # failed occupancy read produces and ``None`` means something else
                # entirely: a creation edge, which carries no occupant and is not
                # judged on identity. Collapsing the two let a restored handoff whose
                # occupancy could not be read skip both the identity check and the
                # store check -- the same twin this PR just fixed in
                # :func:`_edge_identity`, left standing on the restore path.
                manager_tab=recorded_tab if handed else None,
                # The PRINCIPAL off the same line, so a restored handed-over edge is
                # refused when the manager has since been rebound to another store --
                # the comparison the live edge makes and the one the rebuild was
                # publishing grants without.
                store=managed_store(persisted),
            ):
                brought_back += 1
            if manager:
                climbing.append(manager)
        pending = climbing
    # DONE, whatever it managed to restore. From here a gap is a line that ended, which
    # is the answer a key this process never saw has to get -- a tab closed before a
    # restart is exactly that shape, and refusing it would freeze every handoff under it.
    _departures_rebuilt[0] = True
    if brought_back:
        logger.info(
            "Restored the line above %d closed session(s) from the crew log",
            brought_back,
        )
    return brought_back


async def handoff_target(
    state: "DashboardState",
    *,
    caller_session_key: str,
    target: str,
    to: str,
    caller_fenced: bool | None = None,
) -> "dict[str, Any]":
    """Hand *target* over to *to*, which manages it from then on.

    The transfer verb, and the only writer of ``_manager``. A session is created by one
    caller and may have to be run by another -- a lead that opens a conductor and three
    workers has to be able to put those workers under that conductor -- and today it
    cannot: the ownership fence admits the CREATOR, so the conductor is refused
    ``not_creator`` on every worker it was meant to run. This verb moves that answer.

    Who may call it: the target's CURRENT manager, which is its creator until a handoff
    says otherwise. That is the direction a transfer runs in -- the manager gives a
    report away, and neither the report nor the receiving manager can take one -- and it
    is enforced by the ownership fence itself rather than re-checked here, because
    ``handoff`` is one of the operations the fence reads the current manager for (see
    :data:`HANDOFF_WIDENED_OPERATIONS`). An owner driving the dashboard is not fenced at
    all and may hand over any session in the workspace, which is the same reach the
    owner already has for every other verb.

    What the receiver must be: a session this caller ALSO controls, open, in the same
    workspace, and not one of the slots that cannot stand in for a manager -- a cron
    tab, a channel-linked session, an app-scoped or ephemeral session, a mirrored one
    (all refused by :func:`authorize_target`, which this verb runs against the receiver
    exactly as it runs against the target), or a crew member's DM. The member DM is
    refused here rather than there because it is not a containment failure on the
    target side at all: a member's DM slot is the member's own operating surface, bound
    to the member's private memory store, and a session handed to it would be driven
    under a principal the giver is not.

    What moves: the reach :data:`HANDOFF_WIDENED_OPERATIONS` names -- send, read, stop,
    close, adopt, release and a further handoff -- plus the sidebar edge, so the person
    sees the session under the manager that now runs it. What does NOT move is
    ``_created_by``: the record of who OPENED the session is never rewritten, the memory
    delegation gate and the per-creator slot cap keep reading it, and a manager that was
    GIVEN a worker therefore cannot mint sessions against the giver's memory binding.

    Refused as ``already_manager`` when the receiver is already the target's manager,
    which also covers handing a session to itself: the verb records a change, and an
    entry that records a change where none happened is a worse answer than a refusal.
    """
    caller_key = caller_slot_key(state, caller_session_key)
    recipient = (to or "").strip()
    if not recipient:
        raise SessionControlError("a recipient session is required", code="recipient_required")
    # One read for the whole verb: the recipient is authorized twice and the loop guard
    # runs twice, and a switch that answered differently between them would admit a
    # recipient under one rule and write the field under another.
    #
    # OFF the loop. The read loads and validates the config file on a cache miss, and
    # this verb reaches it BEFORE its own prewarm, so an inline read blocks every other
    # session's turn whenever a config edit landed in the request.
    chain_enabled = note_chain_switch(await asyncio.to_thread(manager_chain_enabled))

    def _receiver(*, skip_enabled_check: bool) -> "_ChatSlot":
        """The receiver, authorized on the SAME terms as the target, plus a principal test.

        Run twice -- once before the lock and once inside it -- because the receiver can
        close, gain a channel link or be rebound to another memory store during the lock
        wait, and a handoff is only as bounded as its later read.

        The principal test is POSITIVE, and that is the whole of why it reads the stores
        rather than asking whether the recipient is a crew member. A memory store is a
        separate silo, and the question a handoff has to answer is whether the recipient
        runs under the same principal as the giver -- so it is answered by the two slots
        AGREEING on their binding, which is a fact about the slots. Asking
        ``_member_caller`` instead would answer from the mutable config record, where
        every unreadable or incomplete answer is False: that direction withholds member
        status, which is right for an admission and inverted for a refusal, so a V2
        store whose record lost ``memory_version`` would be admitted as a plain session
        and the member would gain the handed session's reach once the record came back.
        A session the giver created inherits the giver's binding, so the sessions this
        verb exists for pass by construction.

        The member DM prefix is still refused on its own: a DM slot is the member's own
        operating surface rather than a session anybody runs work in, and the prefix is
        part of the slot's key, so that refusal needs no record at all.
        """
        from kiro_crew.members import DM_SLOT_KEY_PREFIX

        got = authorize_target(
            state,
            caller_session_key=caller_session_key,
            target=recipient,
            operation="handoff",
            precomputed_ownership_fenced=caller_fenced,
            # Resolved once by this verb, so the in-lock re-run cannot answer a
            # different question from the one that admitted the recipient, and so the
            # synchronous re-run loads no config.
            precomputed_manager_chain=chain_enabled,
            skip_enabled_check=skip_enabled_check,
        )
        if got.key.startswith(DM_SLOT_KEY_PREFIX):
            raise SessionControlError(
                "a crew member's own session cannot be handed another session to manage",
                code="member_recipient",
            )
        got_store = str(getattr(got, "memory_store", "") or "")
        giver = state.get_slot(caller_key)
        giver_store = str(getattr(giver, "memory_store", "") or "") if giver is not None else ""
        if got_store != giver_store:
            raise SessionControlError(
                f"{name_for_person(state, recipient)!r} is bound to a different "
                "memory store, so it runs under another principal and cannot be handed "
                "a session to manage",
                code="recipient_other_store",
            )
        # The TARGET's binding too, which is the admission a RESTART re-applies. The two
        # comparisons are different questions and the second is the one that has to
        # outlive the process: the rebuild re-validates a recorded grant by comparing the
        # manager against the session it manages, so a handoff admitted only on the
        # giver's binding can be accepted now and silently revoked at the next start.
        # Three sessions make them disagree -- an owner-rooted lead on the default store
        # dispatches a member-store worker and a default-store conductor, and giver and
        # recipient agree while recipient and worker do not. Refusing up front is the
        # honest answer; accepting a grant the next boot will drop is not.
        #
        # The comparison against the TARGET is not made here, and neither is the deputy
        # refusal: both need the target SLOT, and this function has only the string the
        # caller typed. ``_resolve_slot`` accepts a slot key, a transcript stem and a
        # title, so a key lookup on that string answers None for two of the three forms
        # -- and a check that silently does not run is worse than no check. They live in
        # :func:`_gate` instead, which is handed the RESOLVED slot.
        return got

    def _target_side_refusals(handed: "_ChatSlot", receiver: "_ChatSlot") -> None:
        """The refusals that compare the recipient against the SESSION BEING HANDED.

        Separated from the recipient's own admission because they need a different
        thing: the recipient is named by one string this verb resolves once, while the
        target is named by any of the forms ``_resolve_slot`` accepts -- a slot key, a
        transcript stem, or the title on screen. A check written against the caller's
        string instead of the resolved slot runs on one of those forms and silently
        skips the rest, which is how a refusal that is correct where it runs reaches the
        decision without running.

        Called with the resolved slot from :func:`_gate`, which runs after
        ``authorize_target`` has resolved and authorized the target and before anything
        is appended -- so one copy covers every naming form.
        """
        receiver_store = str(getattr(receiver, "memory_store", "") or "")
        if receiver_store != str(getattr(handed, "memory_store", "") or ""):
            raise SessionControlError(
                f"{name_for_person(state, receiver)!r} and "
                f"{name_for_person(state, handed)!r} are bound to different memory "
                "stores, so they run under different principals and the handoff would "
                "not survive a restart",
                code="recipient_other_store",
            )
        # A FENCED recipient may not be handed an UNFENCED session. The fence decides
        # what a caller may reach, and for the verbs a handoff widens it reads the
        # manager -- so a fenced worker given a person's own tab drives a session the
        # fence does not apply to, and through it reaches every session in the workspace
        # while reading the results back. That is the deputy the fence exists to prevent,
        # assembled out of two admissions that are each correct on their own.
        #
        # Refused HERE rather than by making a managed slot count as fenced, which would
        # change the TARGET's fence status as a side effect of being managed: a person's
        # own tab would silently lose its own reach for having been handed to someone.
        #
        # What this costs is nothing the feature is for. A lead handing agent-created
        # workers to an agent-created conductor is fenced to fenced. An owner handing one
        # of their own tabs to another of their own is unfenced to unfenced. Only the
        # mixed direction is refused, and only that one direction builds a deputy.
        if _caller_is_ownership_fenced(state, receiver.key) and not _caller_is_ownership_fenced(
            state, handed.key
        ):
            raise SessionControlError(
                f"{name_for_person(state, receiver)!r} reaches only the sessions it "
                f"created, and {name_for_person(state, handed)!r} is not one of them "
                "-- handing it over would let it drive a session that reaches further "
                "than it does",
                code="deputy_refused",
            )

    await prewarm_enabled_check()
    # AUDITED, and keyed to the target rather than the recipient. This resolution carries
    # the recipient-side refusals nothing else records -- ``member_recipient``,
    # ``recipient_other_store``, and whatever ``authorize_target`` refuses the recipient
    # on -- and each of them is an attempt to put a session under a principal that may
    # not have it, which is the kind of denial the SEL exists for. The target is what the
    # record is about because it is the session whose control was being moved; the
    # recipient appears in the refusal's own message.
    with _audit_denials(
        caller_session_key=caller_session_key, operation="handoff", slot_key=target
    ):
        receiver = _receiver(skip_enabled_check=False)
    receiver_key = receiver.key
    # The receiver's OBJECT, for the reason the mover captures the target's: a close
    # plus a same-key reopen inside the lock wait would otherwise hand the session to a
    # conversation that never agreed to anything, under a key that still resolves.
    receiver_at_resolve = _slot_object(state, receiver_key)
    #: The principal the recipient was admitted on: the memory store both slots shared
    #: at that moment. Captured, because what the publication below has to re-establish
    #: is that the recipient is still the session the admission judged -- not whatever
    #: the giver happens to be bound to later.
    admitted_store = str(getattr(receiver, "memory_store", "") or "")

    def _recipient_still_admissible(handed: "_ChatSlot") -> bool:
        """Whether the recipient is still the session the gate admitted, for *handed*.

        Read SYNCHRONOUSLY, with no config and no IO, immediately before the reach
        moves -- which is the only place it can be read usefully. Every other check in
        this verb happens before the save, and the save suspends: the recipient's own
        routes take the recipient's slot lock and not this verb's, so inside that
        window its agent can be switched, its memory store rebound to another
        principal's, a channel link attached, or the session closed and reopened under
        the same key. Each of those changes WHO would hold the handed session, so each
        of them has to be re-established here rather than assumed from the gate.

        Attribute reads on captured objects, so it is safe in the save's worker: the
        object identity test is what a key-only test cannot do, the store tests are the
        principal rule, and :func:`_live_target_refusal` is the same target-side
        containment ``authorize_target`` applied at admission.

        Both store comparisons are re-read here, and the one against *handed* is the one
        a RESTART will apply: the rebuild re-validates a recorded grant by comparing the
        manager against the session it manages, so a publication admitted on the giver's
        binding alone would be dropped at the next start. Either slot can be rebound
        inside the append's own suspension, so neither reading survives from the gate.
        """
        from kiro_crew.members import DM_SLOT_KEY_PREFIX

        live = state.get_slot(receiver_key)
        if live is None or live is not receiver_at_resolve:
            return False
        if str(getattr(live, "key", "") or "").startswith(DM_SLOT_KEY_PREFIX):
            return False
        live_store = str(getattr(live, "memory_store", "") or "")
        if live_store != admitted_store:
            return False
        if live_store != str(getattr(handed, "memory_store", "") or ""):
            return False
        return _live_target_refusal(live) is None

    #: The manager the handoff replaced, read under the lock by the commit below and
    #: reported to the caller. A list because the commit is a closure and this is the
    #: one value it has to hand back.
    replaced: list[str] = []

    def _gate(slot: "_ChatSlot") -> None:
        # "Already manages" is asked through :func:`_managed_by_other`, the same
        # predicate the fence asks it with, and with the receiver's IDENTITY beside its
        # key. A key-only answer here closes the one way OUT of a stale record: when the
        # recorded manager is gone and a different session answers to its name, the
        # refusal everywhere else says to hand the session over again -- and this gate
        # would refuse that handoff as redundant, because the name matches. The record
        # names a session that is not the one being handed it, so there is something to
        # hand over, and ``for_write`` is what lets the read say so instead of raising.
        if slot.key == receiver_key or not _managed_by_other(
            state, slot, receiver_key, _tab_identity(state, receiver_key), for_write=True
        ):
            raise SessionControlError(
                f"{name_for_person(state, receiver_key)!r} already manages "
                f"{name_for_person(state, slot)!r}, so there is nothing to hand over",
                code="already_manager",
            )
        # Refused on the MANAGER line specifically: the tree guard inside the mover
        # answers for the sidebar, and an adoption moves a sidebar edge without moving
        # a manager, so the two can disagree about who is above whom.
        # Gated with the reach it protects. A loop matters because every walk over one
        # fails closed, and with the line OFF nothing walks it -- the fence reads the
        # direct manager alone -- so there is no broken state here to prevent, and
        # refusing anyway would reject a handoff the gateway accepts today.
        #
        # ``unread_is_present=True`` is the writer's polarity: a line the walk could not
        # finish leaves a loop possible, and a WRITER cannot act on the fence's
        # assumption. The fence refuses reach on an unread line and is safe to be wrong;
        # this write would PERSIST the loop, and one loop takes the manager away from
        # every session in it at once.
        # NO starting identity, and that is not an omission. The walk starts AT the
        # receiver rather than above it: the edge that would name it is the one this
        # verb is deciding whether to write, and the receiver itself was resolved live
        # by the verb a moment ago. Every hop ABOVE it is named by an edge the walk
        # reads as it goes, so each of those is judged.
        #
        # The TARGET's own identity is passed as the claim even though the compare is
        # redundant here -- the target is the live slot this verb just resolved, so the
        # occupant of that key is this very object. Passed anyway, because an exemption
        # is a list and a list is what the next call site is added without; and a
        # mismatch, were one somehow possible, refuses, which is this writer's polarity.
        line = (
            _walk_manager_line(
                state, receiver_key, slot.key, claimed=_tab_identity(state, slot.key)
            )
            if chain_enabled
            else None
        )
        if line is not None and (line.found or line.unread):
            # REFUSED either way: walking a chain that could not be read to the end and
            # concluding "no cycle" is the thing this fences. The walk reads EVERY hop's
            # unsettled state, not just the two endpoints, because the loop a handoff
            # closes can run entirely through hops that are neither.
            #
            # Both answers come from the SAME walk. Asking a second time to find out why
            # it stopped is how a guard ends up correct in one copy and missing from the
            # other, which is the shape of mistake this feature has already made.
            #
            # The two codes differ only in what the caller should do about it, and only
            # the ways a line goes unread that CLEAR ON THEIR OWN earn a retry: a hop
            # inside its own close, which publishes its manager when that close commits
            # or rolls back, and a hop mid-handoff, which publishes when its append
            # settles. A cycle further up, a line past the depth bound and an evicted
            # record do not change by waiting, so they keep the refusal this verb
            # already shipped.
            if line.waits:
                raise SessionControlError(
                    f"the reporting line above {receiver_key!r} runs through a session "
                    "whose own manager is still being written, so whether this would "
                    "close a loop is not known yet -- retry once that write has landed "
                    f"to hand {slot.key!r} over",
                    status=409,
                    code="manager_line_pending",
                )
            raise SessionControlError(
                f"{receiver_key!r} reports to {slot.key!r}, or its reporting line cannot "
                f"be read to the end, so handing {slot.key!r} to it would risk a loop",
                status=409,
                code="would_cycle",
            )
        again = _receiver(skip_enabled_check=True)
        if again.key != receiver_key or _target_replaced(state, receiver_key, receiver_at_resolve):
            raise SessionControlError(
                f"{name_for_person(state, receiver_key)!r} is not the same session it "
                "was when the handoff started -- retry to hand over to the one open now",
                status=409,
                code="recipient_replaced",
            )
        # The two refusals that compare the recipient against the session being handed,
        # applied to the RESOLVED slot and so to every form the caller may have named it
        # by. This gate is where the verb first holds that slot: ``authorize_target`` has
        # resolved and authorized it, and nothing has been appended yet.
        _target_side_refusals(slot, again)

    async def _commit(slot: "_ChatSlot", target_sid: str) -> None:
        """Record the new manager, then publish it on the slot.

        The commit point is this verb's own ``session/manager_set`` entry, appended and
        awaited here. That entry states who manages the target, it is append-only, and a
        start rebuilds ``_manager`` from the newest one per slot -- so the crew log is
        the authority, and the only store for WHO manages it. Nothing is published until
        the entry is on disk, and then one INDEX ENTRY says the log has
        an answer -- in the crew log's own handed-over index, never on the slot's line
        and never the manager itself, so there is no cached value to fall out of date
        with the log and nothing the revoked party can rewrite.

        ``_manager`` is this process's copy of that entry, and nothing else writes it.
        A writer that reaches the slot while this runs -- a turn's own save, the
        periodic flush -- encodes a line the manager is not on, so there is no path by
        which a value here reaches disk ahead of its record, no pending value for
        another writer to pick up, and no retraction debt to carry.

        The recipient is re-established TWICE, and the second time is the one that
        decides. The gate's verdict is formed before the append, and the append
        suspends: the recipient's own routes take its slot lock and not this verb's, so
        inside that suspension its agent can be switched, its memory store rebound to
        another principal's, a channel link attached, or the session closed and reopened
        under the same key. Each of those changes WHO would hold the handed session, so
        the check that matters is the one immediately before the publish, with nothing
        suspending after it. The earlier one is kept because it is free and spares the
        log a correction in the common case.

        A refusal in that window is CORRECTED on the log, not merely left unpublished.
        The entry is already the authority, so a refused handoff whose entry stands
        would be granted at the next boot by the very record the refusal was supposed
        to prevent.
        """
        previous_manager = require_manager_of(state, slot, for_write=True)
        #: The PUBLISHED manager this handoff replaces, which is what a correction has to
        #: restore. Deliberately not ``previous_manager``: that one reads through to the
        #: creator for a slot never handed over, and a correction naming the creator
        #: would state a handoff TO the creator rather than none at all.
        previously_published = str(getattr(slot, "_manager", "") or "")
        #: The OCCUPANCY that record carried, captured in the same breath as the key it
        #: qualifies. A correction has to restore the whole earlier statement, and
        #: re-reading the occupancy when the correction is written would stamp whoever
        #: answers to the name THEN -- which is the bare-name hole, reintroduced by the
        #: compensation. Read here, under the lock, beside the key.
        previously_published_tab = str(getattr(slot, "_manager_tab", "") or "")
        #: WHICH SESSION is being handed over, on the entry and on its correction. The
        #: slot key is a name and a closed session's name can be taken by a new one, so
        #: an entry qualified by key alone is applied at the next start to whoever holds
        #: the name. Refused rather than written weakly when the target carries no
        #: identity: a record nothing can attribute is one a later occupant reads as its
        #: own, which is the whole defect this field closes.
        handed_tab = str(getattr(slot, "_tab_id", "") or "")
        #: The recipient's OCCUPANCY, read ONCE and used for both the entry and the
        #: publish. This is the one place entitled to read the live occupant of a key,
        #: because this is where the record is MADE -- everywhere else applies a record
        #: and must use the identity that record carries. Reading it twice would let the
        #: two differ, which is the same defect in miniature: the published value has to
        #: BE the recorded value, not another look at the same key.
        receiver_tab = _tab_identity(state, receiver_key)
        if not handed_tab:
            raise SessionControlError(
                f"{name_for_person(state, slot)!r} carries no session identity, so a "
                "handoff of it could not be recorded in a way a restart can attribute "
                "to this session rather than to a later one under the same name",
                status=409,
                code="handoff_not_recorded",
            )
        if not _recipient_still_admissible(slot):
            raise SessionControlError(
                f"{name_for_person(state, receiver_key)!r} changed while the handoff "
                f"was being recorded, so {name_for_person(state, slot)!r} was not given "
                "to it -- retry to hand over to the session open now",
                status=409,
                code="recipient_replaced",
            )

        async def _take_back() -> None:
            """Put the log's newest manager statement back where it was.

            A correction that could not be made durable leaves this process UNABLE TO
            SAY who manages the slot, and it must stop claiming it does. The log may
            now end with the refused grant -- the queued entry can still land, and a
            rebuild folds the newest statement per slot -- while this process goes on
            answering the giver from a field the grant was never published to. That is
            a reach granted here and revoked at the next start, which is the one
            direction this whole verb exists to avoid.

            So the slot goes UNRESOLVED, which is default-deny: every reader that
            authorizes against the manager goes through
            :func:`require_manager_of` and is refused until an authoritative read
            succeeds. ``resolve_slot_manager`` is what clears it, and it reads the log
            -- so once the writer drains, this process agrees with whatever the log
            actually says, grant or correction, instead of waiting for a restart to
            disagree with it.

            Deliberately NOT a retry. A writer backed up enough to time out the grant's
            settlement is the same writer at its ceiling that rejected this correction,
            so the two failures arrive together and a retry spends the caller's time to
            reach the same place. Refusing to answer costs one read; guessing costs the
            giver a reach it will lose.

            EITHER WAY THIS PROCESS STATES SOMETHING, which is what keeps a read that is
            already in flight from landing on top of it. A correction moves the LOG; a
            resolve that read the log before it holds the refused recipient and a
            revision snapshot that nothing has advanced, so its publish would pass the
            check and grant exactly the reach this verb refused. Restating the outcome
            advances the revision and discards that read. It needs no lock over the
            window: every ordering ends on a statement made here, because the
            restatement is unconditional and the resolve's is not. The resolve publishing
            first changes nothing -- this overwrites it; publishing later is discarded.
            """
            if await _revoke_manager_record(
                target_sid=target_sid,
                slot_key=slot.key,
                refused_manager=receiver_key,
                slot_tab=handed_tab,
                previously_published=previously_published,
                previously_published_tab=previously_published_tab,
            ):
                # RESTATED, not merely left alone. The correction moved the log without
                # this process saying anything, and a resolve already in flight read the
                # log BEFORE the correction: its answer is the refused recipient, its
                # revision snapshot is still current because nothing published, and its
                # publish would therefore pass the check and authorize the session this
                # verb just refused. Saying the restored statement out loud is what
                # makes that read stale -- the value is the one the slot already holds,
                # so the only thing this changes is the revision, which is the point.
                #
                # The inadmissible flag is CARRIED rather than cleared: whether the
                # restored grant may authorize is the resolve's question, answered
                # against the live slots, and a correction is not evidence about it.
                # ``unresolved`` is False because the log's content is known here --
                # this process just wrote it -- and an unresolved slot could not have
                # reached this verb at all, since its gate reads the manager.
                publish_manager(
                    slot,
                    manager=previously_published,
                    manager_tab=previously_published_tab,
                    inadmissible=bool(getattr(slot, "_manager_grant_inadmissible", False)),
                )
            else:
                publish_manager(slot, manager="", manager_tab="", unresolved=True)
                logger.error(
                    "session_handoff: %s was NOT given to %s, and the record of that "
                    "handoff could not be taken back -- the log may name %s as its "
                    "manager, so who manages %s is now UNRESOLVED and session control "
                    "refuses it until the log is read again",
                    slot.key,
                    receiver_key,
                    receiver_key,
                    slot.key,
                )

        # THE COMMIT POINT. This entry is what makes the handoff durable and what a
        # start rebuilds the manager from, so nothing is published until it is on disk.
        # The settlement is read in THREE values because the two ways of not being on
        # disk need opposite handling here: a refused append left nothing behind, and an
        # append still queued can land after this verb has refused.
        settlement = await crew_log_emit.awaiting_settlement(
            lambda settled: _emit_manager_record(
                target_sid,
                slot_key=slot.key,
                slot_tab=handed_tab,
                manager=receiver_key,
                manager_tab=receiver_tab,
                previous=previous_manager,
                to_creator=False,
                on_settled=settled,
            ),
            what=f"the handoff of {slot.key} to {receiver_key}",
        )
        if settlement is not True:
            if settlement is None:
                # Still queued, outcome unknown. It may land at any time after this
                # refusal, so the correction goes on the log behind it.
                await _take_back()
            if getattr(slot, "_manager_unresolved", False):
                # The correction did not land either, so "the giver still manages it"
                # is a claim this verb cannot make -- the log may name the recipient.
                # Said plainly, with its own code, because the caller's next move is
                # different: there is nothing to retry until the gateway can read the
                # log again, and until then the session answers no one.
                raise SessionControlError(
                    f"the handoff of {name_for_person(state, slot)!r} could not be "
                    "recorded and could not be taken back, so which session manages it "
                    "is now unknown until the crew log is read again",
                    status=409,
                    code="handoff_unresolved",
                )
            raise SessionControlError(
                f"the handoff of {name_for_person(state, slot)!r} could not be "
                f"recorded, so {name_for_person(state, previous_manager)!r} still "
                "manages it -- retry",
                status=409,
                code="handoff_not_recorded",
            )
        # Re-established after the append's suspension. Free, and it spares the log a
        # correction in the common case -- but it is NOT the check that decides, because
        # the marker write below suspends again.
        if not _recipient_still_admissible(slot):
            await _take_back()
            raise SessionControlError(
                f"{name_for_person(state, receiver_key)!r} changed while the handoff "
                f"was being recorded, so {name_for_person(state, slot)!r} was not given "
                "to it -- retry to hand over to the session open now",
                status=409,
                code="recipient_replaced",
            )
        # THE INDEX ENTRY, after the entry and before the publish. The log entry says
        # who holds the session; the index says the log is worth reading for it at all.
        # Without it a start resolves this slot to its creator without reading anything,
        # so a handoff published here would be gone at the next start and the giver
        # would have its reach back. An index entry that cannot be written refuses the
        # handoff, and the grant is taken back on the log for a slot an EARLIER handoff
        # marked -- where the log is read, and that grant would otherwise be the answer.
        if not await _record_handed_over(state, slot):
            await _take_back()
            raise SessionControlError(
                f"the handoff of {name_for_person(state, slot)!r} could not be recorded "
                f"in the handoff index, so "
                f"{name_for_person(state, previous_manager)!r} still manages it -- retry",
                status=409,
                code="handoff_not_recorded",
            )
        # THE CHECK THAT DECIDES, and it is this one because it is the LAST one before
        # the reach moves with nothing suspending after it. The marker write above takes
        # the transcript's cross-process lock off the loop, so it is a suspension of the
        # same kind as the append -- and inside it the recipient's own routes, which take
        # the recipient's slot lock and not this verb's, can switch its agent, rebind its
        # memory store to another principal's, attach a channel link, or close and reopen
        # the session under the same key. A verdict formed before that write is a verdict
        # about a session that may not be the one about to be handed a worker.
        if not _recipient_still_admissible(slot):
            await _take_back()
            raise SessionControlError(
                f"{name_for_person(state, receiver_key)!r} changed while the handoff "
                f"was being recorded, so {name_for_person(state, slot)!r} was not given "
                "to it -- retry to hand over to the session open now",
                status=409,
                code="recipient_replaced",
            )
        # One statement, no suspension between the check above and it: the key, the
        # occupancy that qualifies it, and the two refusal flags cleared -- a handoff is
        # the statement the log was missing, so a slot denied for an inadmissible grant
        # is servable again the moment one lands.
        publish_manager(slot, manager=receiver_key, manager_tab=receiver_tab)
        replaced.append(previous_manager)
        # Two writes, in this order and no other: the ENTRY (who holds it) and then the
        # INDEX (that the log has an answer). Both land in the fenced crew log; nothing
        # about who manages a session is written to the slot's own line, where the party
        # a handoff revokes could rewrite it.

        # Except for one copy, and it has to be corrected HERE. This window outlives
        # the request, so a close or an idle archive can take this slot out of the live
        # table while the entry above is still being written. That departure records the
        # COMMITTED manager, which at that moment is still the one being replaced -- so
        # publishing the new manager onto a slot nobody can reach any more would leave
        # the walk resolving this key through the manager the handoff just took it from,
        # and refusing the one it was given to, until the entry is evicted.
        #
        # So if the slot has already left the table, the departure entry for THIS
        # occupancy is rewritten with what was just published. Guarded by the same
        # generation check every other writer goes through, so a key the occupancy has
        # moved on from is not resurrected, and a slot still live records nothing here:
        # the walk reads a live slot off the table and never consults the entry.
        if state.get_slot(slot.key) is not slot:
            # ``receiver_key`` rather than re-reading the slot: this commit published
            # that value two lines up, so it is the one manager here that needs no
            # chokepoint -- re-reading it would be asking a question already answered.
            #
            # ``manager_tab`` is carried explicitly, not left at ``_KEEP``: the entry
            # being corrected may be the pending one the pop wrote (identity ``""``) or
            # the previous edge's, and keeping either would pair the newly published
            # manager with the WRONG occupancy -- a later different occupant of this key
            # would then resolve the line through this record, or a creation edge's
            # ``None`` would skip the identity check on the receiver's key entirely. The
            # correct identity is the one just published, which ``_edge_identity`` reads
            # back off the slot's own freshly set ``_manager_tab``.
            _record_departure(
                slot.key,
                _occupancy_generation_of(slot),
                receiver_key,
                pending=False,
                manager_tab=_edge_identity(slot),
                # And this slot's own principal, for the reason the other two writers
                # carry it: the corrected entry is what the walk reads through, so it
                # has to answer the store comparison with a value and not a blank.
                store=str(getattr(slot, "memory_store", "") or ""),
            )

    moved = await _reparent_target(
        state,
        caller_session_key=caller_session_key,
        target=target,
        parent_key=receiver_key,
        operation="handoff",
        noun="handoff",
        caller_fenced=caller_fenced,
        gate=_gate,
        commit=_commit,
        # BOTH endpoints are fenced for a handoff. The commit window outlives the
        # request here -- the manager entry and, on a refusal, its correction are still
        # being written after the caller has been answered -- so the reverse handoff is
        # the move that must not be decided meanwhile, and it names the recipient as its
        # target rather than this one.
        fence_parent=True,
        # The sidebar is told where the window publishes, for the reason the audit is:
        # a handoff the caller stopped waiting for still moved the session, and a
        # broadcast tied to the caller's frame would never reach the tree that changed.
        after_commit=state.push_slots_update,
    )
    return {
        "target": moved["target"],
        "manager": receiver_key,
        # The manager this handoff replaced, read from the TARGET before the move and
        # not inferred from the caller: an owner handing over a worker somebody else
        # created is not the manager it replaced, so naming the caller here would
        # report the wrong session for exactly the caller class that is not fenced.
        "previous_manager": replaced[0],
        # NO ``cached`` key. There is no second copy to report on: the crew log entry is
        # the whole of the durable state, so a handoff that returns at all is a handoff
        # the log carries.
        "previous_parent": moved["previous_parent"],
    }


#: How long a tree verb waits for its own append to be on disk before it refuses. The
#: wait exists because the append IS the operation; the bound exists because the writer
#: retries a refusing filesystem before giving up, and a caller must get an answer either
#: way rather than hanging for as long as the disk stays wedged.
_TREE_APPEND_TIMEOUT = 10.0

#: One mutation lock per EVENT LOOP, not one per process. A lock constructed at import
#: binds to whichever loop first waits on it, and the test suite runs many loops in one
#: process -- a single module-level lock would then raise or, worse, serialize against a
#: loop that has already closed. Keyed by loop identity, created on demand.
_tree_mutation_locks: "dict[int, asyncio.Lock]" = {}


@contextmanager
def _audit_denials(*, caller_session_key: str, operation: str, slot_key: str) -> Iterator[None]:
    """Record a verb's OWN refusals as denied, then re-raise them untouched.

    :func:`_audit` is easy to reach on the success path and easy to miss on every other
    one, because a refusal LEAVES the verb by raising -- so the allowed call at the end
    is simply not executed and the denial goes unrecorded. That is exactly the shape a
    caller probing a permission boundary produces: every attempt refused, and none of
    them appearing anywhere. ``backend-security-controls`` requires a SEL event for each
    permission decision, and a boundary with no trail is the one most worth a trail.

    One helper rather than a handler written into each verb: two would drift, and the
    audited fact is the same in both. It wraps the region AFTER the authorization gate
    has named a target, which is what gives the record a ``slot_key`` to be about -- a
    refusal with no resolved target has nothing to attribute.

    The exception is re-raised unchanged, so this only ever ADDS the record: the status,
    code and message the caller receives stay the verb's own.
    """
    try:
        yield
    except SessionControlError as exc:
        _audit(
            caller_session_key=caller_session_key,
            operation=operation,
            slot_key=slot_key,
            outcome="denied",
            detail={"code": exc.code},
        )
        raise


def _tree_mutation_lock() -> "asyncio.Lock":
    """The lock a tree mutation holds across reading the tree and committing to it.

    The two verbs decide from the tree (who the parent is, whether a takeover would
    close a loop) and then write to it, and between those two steps another verb can
    land its own decision. Unserialized, a release authorized against the old parent
    commits after an adoption and puts the session back under a parent that had already
    handed it over -- both writes valid, the later one the one nobody asked for.

    Held from the tree READ through the awaited append, which together with that await
    is what makes the pair atomic: the append's completion also advances the projection,
    so the next verb through this lock reads the state the previous one committed rather
    than the state it started from.

    Not held across :func:`authorize_target`, which does not read the tree. The
    serialized region is the one that decides on tree state.
    """
    loop = asyncio.get_running_loop()
    lock = _tree_mutation_locks.get(id(loop))
    if lock is None:
        lock = asyncio.Lock()
        _tree_mutation_locks[id(loop)] = lock
    return lock


#: Slots whose tree append has been handed to the writer and has NOT settled. Keyed by
#: slot, one entry per in-flight append, cleared when the future resolves either way.
#:
#: The mutation lock alone does not cover this. A verb that times out waiting for its own
#: append raises out of the lock, which releases it -- and the append is still queued, so
#: the next verb through the lock would decide from a tree the writer is about to change
#: and commit against it. Holding the lock until settlement instead would pin the loop's
#: serialized region to a wedged filesystem, which is what the timeout exists to avoid.
#: Fencing the SLOT is the third option: the loop stays free, and no verb decides on a
#: slot whose state is about to move.
_tree_pending_slots: "dict[str, asyncio.Future[bool]]" = {}

#: The reparent EDGES handed to the writer and not yet settled, as ``slot -> new parent``.
#:
#: What the cycle check has to see. :data:`_tree_pending_slots` says a slot is moving;
#: this says where to, and only the second closes a loop. The fences built on the first
#: refuse a move that shares an ENDPOINT with a queued one, which is the two-node case
#: (``A`` to ``B`` queued, ``B`` to ``A`` next) and nothing larger: with ``D -> A`` and
#: ``B -> C`` already in the tree and ``A -> B`` queued, a handoff of ``C`` to ``D`` names
#: neither ``A`` nor ``B``, passes both fences, and is checked against a tree the queued
#: edge has not reached. All four entries are then valid in an append-only log and the
#: fold has a cycle it can only mark, not undo.
#:
#: Registered when the append is handed over and cleared when it settles -- the same
#: lifetime as the fence, for the same reason: the window that matters is exactly the one
#: where the writer holds an edge the projection cannot show.
_tree_pending_edges: "dict[str, str]" = {}


def _record_pending_edge(slot: str, parent: str, landed: "asyncio.Future[bool]") -> None:
    """Publish ``slot -> parent`` as a queued edge until *landed* settles.

    Self-clearing through the future's own callback, so an append that drops, lands or
    times out all end the same way and nothing has to remember to retract the edge.
    """
    if not slot or not parent:
        return
    _tree_pending_edges[slot] = parent

    def _forget(_f: "asyncio.Future[bool]") -> None:
        if _tree_pending_edges.get(slot) == parent:
            del _tree_pending_edges[slot]

    landed.add_done_callback(_forget)


def _tree_with_pending_edges(nodes: "Mapping[str, Any]") -> "dict[str, Any]":
    """*nodes* overlaid with every queued reparent edge, for the cycle check to walk.

    The ancestor walk follows ``TreeNode.parent_slot`` and steps only onto a slot that
    HAS a node, so a queued edge is expressed by replacing the moving slot's node with
    one naming its new parent -- and by MINTING a node for a slot the settled tree has
    none for, because a session whose first edge is still queued would otherwise end the
    walk right where the loop closes.

    ``cycle`` is cleared on an overlaid node. The flag belongs to the settled fold's
    reading of that slot, and this view asks a different question: whether the move under
    consideration would ADD a loop. Carrying a stale mark would stop the walk early and
    answer "no cycle" for the one case worth refusing.

    Raises on anything it cannot express. The caller turns that into a refusal, because a
    cycle check is only as good as the view it ran against: concluding "no loop" from a
    view that could not be built is the confident wrong answer, not a missing one.
    """
    from kiro_crew.crew_log.session_tree import TreeNode

    combined = dict(nodes)
    for child, parent in list(_tree_pending_edges.items()):
        if not child or not parent:
            raise ValueError(f"a queued edge names no slot: {child!r} -> {parent!r}")
        combined[child] = TreeNode(slot=child, parent_slot=parent, cycle=False)
    return combined


def _tree_append_waiter(slot: str) -> "tuple[Callable[[bool], None], asyncio.Future[bool]]":
    """A ``(on_settled, future)`` pair for one tree append, fencing *slot* until it lands.

    The emitter calls ``on_settled`` from the WRITER's thread, so the result is handed
    back through the loop rather than set directly, and a second call is a no-op.

    Registering the future in :data:`_tree_pending_slots` is what makes the fence
    self-clearing: the entry is removed by a done callback, so it goes when the append
    settles and nothing has to remember to release it. The wait below shields the future
    for the same reason -- a timeout that cancelled it would resolve it, and the fence
    would clear while the append is still queued.
    """
    loop = asyncio.get_running_loop()
    future: "asyncio.Future[bool]" = loop.create_future()

    def _settled(landed: bool) -> None:
        def _set() -> None:
            if not future.done():
                future.set_result(landed)

        loop.call_soon_threadsafe(_set)

    def _unfence(_f: "asyncio.Future[bool]") -> None:
        if _tree_pending_slots.get(slot) is future:
            del _tree_pending_slots[slot]

    _tree_pending_slots[slot] = future
    future.add_done_callback(_unfence)
    return _settled, future


def _refuse_if_append_pending(slot: str, *, operation: str) -> None:
    """Refuse when *slot* has a tree append still in flight. Called inside the lock.

    The tree cannot be reasoned about for a slot whose next state is already queued: a
    decision taken now would be taken against a state the writer is about to replace, and
    it would commit UNDER the queued one rather than after it.

    ``tree_write_pending`` rather than a failure, for the reason the timeout uses that
    code: the earlier append may still land, so the caller is told to read the tree and
    try again rather than that anything went wrong.
    """
    pending = _tree_pending_slots.get(slot)
    if pending is not None and not pending.done():
        raise SessionControlError(
            f"an earlier tree write for {slot!r} is still in flight, so the {operation} "
            "cannot be decided yet -- read the session tree before retrying",
            status=409,
            code="tree_write_pending",
        )


#: How long a commit WINDOW waits for the writer's definitive verdict on its tree
#: append. Longer than :data:`_TREE_APPEND_TIMEOUT` because it bounds something else:
#: that one bounds how long a REQUEST waits, and this one bounds how long the window
#: task -- which outlives the request -- waits before it stops expecting a verdict and
#: compensates instead. A window that gave up on the caller's bound would be the hole
#: the compensation exists to close, so the two cannot be the same number.
_TREE_SETTLE_TIMEOUT = 120.0


async def _tree_append_verdict(
    future: "asyncio.Future[bool]", *, target: str, timeout: float
) -> bool | None:
    """The writer's verdict on a tree append: ``True`` on disk, ``False`` refused, or
    ``None`` for still queued when *timeout* ran out.

    Three-valued because the two ways of not being on disk call for opposite handling.
    A refusal is final and leaves nothing behind, so the operation it was ordered for
    simply does not happen. A still-queued append may land at ANY time afterwards, so
    whoever gives up on it owes the log a correction.

    The slot's fence is dropped HERE on both decided verdicts, rather than being left to
    the future's done callback: that callback runs on a later loop iteration, and a verb
    whose request finishes first would leave the slot fenced for a write that already
    landed. The callback stays as the backstop for a future resolved somewhere other
    than this wait, and it is also what clears the fence on the undecided verdict, where
    the fence is deliberately left standing until the writer does answer.
    """
    try:
        # SHIELDED, so the bound cancels only this wait. Cancelling the future itself
        # would resolve it, the fence's done callback would fire, and the slot would be
        # unfenced at the one moment the fence is needed: the append is still queued and
        # the next verb must not decide against a tree it is about to move.
        landed = await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
    except asyncio.TimeoutError:
        return None
    if _tree_pending_slots.get(target) is future:
        del _tree_pending_slots[target]
    return landed


async def _tree_append_landed(
    future: "asyncio.Future[bool]",
    *,
    operation: str,
    target: str,
    timeout: float = _TREE_APPEND_TIMEOUT,
) -> None:
    """Wait for a tree append to be durable, and REFUSE when it was not.

    What this stops the verb from doing is reporting a takeover that never reached the
    disk: the emitter hands the entry to the writer and returns, so without this wait a
    filesystem that refused the append, or a session with no live log to write to, would
    still have answered the caller "adopted" and audited it as allowed.

    A still-queued append is reported as retryable rather than as a failure, because it
    is neither: the writer may yet land the entry. The distinction matters to a caller
    deciding whether to try again, and both codes say which it is.

    For an append that is the WHOLE operation, which is the only case with nothing to
    hold together: the append either happened or it did not, and a caller told
    "retryable" can read the tree and decide. An append another half is ordered behind
    goes through :func:`_tree_append_verdict` directly, because the undecided verdict
    there has to be compensated rather than reported.
    """
    landed = await _tree_append_verdict(future, target=target, timeout=timeout)
    if landed is None:
        raise SessionControlError(
            f"the {operation} of {target!r} was handed to the crew-log writer but has "
            "not landed yet, so it is not confirmed -- read the session tree before "
            "retrying",
            status=409,
            code="tree_write_pending",
        )
    if not landed:
        raise SessionControlError(
            f"the {operation} of {target!r} could not be written to the crew log, so "
            "the session tree is unchanged",
            status=500,
            code="tree_write_failed",
        )


def _hold_tree_fence(slot: str) -> "Callable[[], None]":
    """Fence *slot* until the returned callable is run. For a window with no append in it.

    :func:`_tree_append_waiter`'s fence is tied to one append and clears when that append
    settles, which is right for the append and not enough for what follows it: the commit
    and, if the commit refuses, the compensating append that puts the parent back. Across
    that window the slot's recorded parent and the field the commit publishes disagree,
    and no other verb may decide against it -- not even though the lock may already be
    gone, because a cancelled caller releases the lock while the shielded work runs on.

    The returned callable resolves the future, which both clears the entry directly and
    fires the done callback as the backstop -- the same two-sided clearing
    :func:`_tree_append_landed` does, and for the same reason: the callback runs on a
    later loop iteration, too late for a request that finishes first.

    Registering this REPLACES whatever fence the slot had, so it is installed with no
    await between it and the unfence it takes over from. A compensating append inside the
    window registers its own fence over this one and clears it on settling; this hold's
    release then finds a different entry and leaves it alone, which is correct -- the
    append that cleared it was the last durable act of the window.
    """
    loop = asyncio.get_running_loop()
    future: "asyncio.Future[bool]" = loop.create_future()

    def _unfence(_f: "asyncio.Future[bool]") -> None:
        if _tree_pending_slots.get(slot) is future:
            del _tree_pending_slots[slot]

    _tree_pending_slots[slot] = future
    future.add_done_callback(_unfence)

    def _release() -> None:
        if not future.done():
            future.set_result(True)
        if _tree_pending_slots.get(slot) is future:
            del _tree_pending_slots[slot]

    return _release


def _emit_manager_record(
    target_sid: str,
    *,
    slot_key: str,
    slot_tab: str,
    manager: str,
    manager_tab: str,
    previous: str,
    to_creator: bool,
    on_settled: "Callable[[bool], None]",
) -> None:
    """THE ONLY CALLER of ``on_manager_set`` outside the emitter. One entry shape.

    A ``session/manager_set`` entry states a key AND the occupancy that qualifies it,
    and the second half is the one a second call site forgets. ``manager_tab`` defaults
    to empty at the emitter -- it has to, for the entry forms that have no manager --
    so omitting it is silent at the write and expensive at the read: the restore brings
    back a bare name, and the key-to-key compare then refuses the manager that is still
    live, or admits whoever answers to that name now. That is what the handoff's
    correction did: it restored the previous manager with no occupancy, on the ordinary
    refused-handoff path.

    Funnelling the two writers through one assembly is what makes the occupancy
    non-optional here even though the emitter keeps it optional. Both of them -- the
    handoff's own entry and the correction that takes it back -- state every field by
    name, and a contract test pins that no third site calls the emitter directly.

    ``slot_tab`` is the TARGET's identity and sits on both statements. It is the half
    the occupancy field cannot cover: ``manager_tab`` says which session was handed the
    work, and this says which session the work IS. Without it the entry names a slot
    key, which is a name a closed session's replacement can take, and a restart applies
    the grant to whoever holds the name. Identity has to be on the record, because a
    check made where the value is COMPARED cannot tell that the value was written for
    somebody else.
    """
    from kiro_crew.crew_log import emit as crew_log_emit

    crew_log_emit.on_manager_set(
        target_sid,
        slot=slot_key,
        slot_tab=slot_tab,
        manager=manager,
        previous=previous,
        to_creator=to_creator,
        manager_tab=manager_tab,
        on_settled=on_settled,
    )


async def _revoke_manager_record(
    *,
    target_sid: str,
    slot_key: str,
    slot_tab: str,
    refused_manager: str,
    previously_published: str,
    previously_published_tab: str,
) -> bool:
    """Durably take back a manager entry whose handoff was refused. Answers whether that
    landed.

    The manager counterpart of :func:`_restore_parent`, and needed for the same reason:
    the log is the AUTHORITY for who drives a session, so a handoff that is refused after
    its entry was queued cannot be unsaid by simply not publishing the field. The entry
    is what a start rebuilds from, so an entry left standing hands the refused session
    its reach at the next boot -- the gateway's own refusal surviving only in memory.

    An erasure is not available and is not needed: the log is append-only and the fold
    takes the NEWEST manager entry per slot, so a correction queued after the refused one
    is what a rebuild reads. The writer is FIFO per unit, so this is ordered behind the
    entry it corrects even when that entry has not settled yet -- which is the case this
    matters most in.

    TWO statements, because "back" is two different facts, exactly as it is for a parent.
    A slot that had a published manager goes back to that session. A slot that had none
    reverts to its CREATOR, which is the statement :func:`~kiro_crew.crew_log.emit
    .on_manager_set`'s ``to_creator`` exists to make -- a correction naming the creator as
    the manager would instead record a handoff TO the creator, leaving the field set
    where it was empty and the next release answering for a handoff that never happened.

    Bounded and non-raising, like the parent compensation: the caller already carries the
    refusal that brought it here. False means the log may still state the refused manager,
    which is the one case a person has to be told about.
    """
    from kiro_crew.crew_log import emit as crew_log_emit

    if previously_published:
        what = f"taking {slot_key} back to {previously_published}"
    else:
        what = f"taking {slot_key} back to its creator"

    def _emit(settled: "Callable[[bool], None]") -> None:
        _emit_manager_record(
            target_sid,
            slot_key=slot_key,
            slot_tab=slot_tab,
            manager=previously_published,
            # The occupancy the record being restored CARRIED, handed in from the
            # commit rather than read here: this runs after the window has given up
            # the lock, so a read now would stamp the current occupant of the name.
            manager_tab=previously_published_tab,
            previous=refused_manager,
            to_creator=not previously_published,
            on_settled=settled,
        )

    try:
        return await crew_log_emit.awaiting_commit(_emit, what=what)
    except Exception:
        logger.warning("manager compensation: %s failed", what, exc_info=True)
        return False


def _restoring_would_cycle(slot_key: str, previous_parent: str) -> bool:
    """Whether putting *slot_key* back under *previous_parent* would close a loop.

    The forward path's cycle check, applied to the rollback. A compensation runs after the
    window has given up its place in the mutation lock, so the tree it is undoing into is
    not the tree it moved out of: the previous parent can have been reparented under this
    very slot meanwhile, each move checked and each fine on its own, and the restoration
    is then the edge that completes the cycle.

    FAIL CLOSED, and here that means answering True: a view this cannot build is a view
    that cannot clear the edge, and the caller's alternative -- releasing the target to
    the root -- loses nesting rather than persisting a loop. That is the opposite
    direction from the forward path, which refuses the operation, because a compensation
    has no refusal available: it is already the undo.
    """
    try:
        from kiro_crew.crew_log.holders import is_ancestor

        tree_known, _parent, nodes = _slot_tree_parent(slot_key)
        if not tree_known:
            return True
        return is_ancestor(slot_key, previous_parent, _tree_with_pending_edges(nodes))
    except Exception:
        logger.debug(
            "tree compensation: the restoration of %s under %s could not be checked for " "a loop",
            slot_key,
            previous_parent,
            exc_info=True,
        )
        return True


async def _restore_parent(
    *,
    target_sid: str,
    slot_key: str,
    parent_key: str,
    parent_sid: str,
    previous_parent: str,
    previous_parent_sid: str,
) -> bool:
    """Durably put *slot_key* back under the parent it had. Answers whether that landed.

    The compensation for a move whose commit refused. The log is append-only, so putting
    a parent back is not an erasure but another edge stated on top: the fold prefers the
    newest entry, so the tree reads the restored parent and the history of both moves
    stays legible.

    TWO entry types, because "back" is two different facts. A target that had a parent is
    adopted BACK by it. A target that had none was a root, and the entry that makes a
    session a root again is the release -- an adoption naming an empty parent is not an
    edge at all and the emitter refuses to write one, which would have left the
    compensation silently unwritten for exactly the root case.

    THREE, counting the case where putting the parent back would close a loop. The tree
    moves on while this window runs: the previous parent can have been reparented UNDER
    the target meanwhile, by moves that were each checked and each fine, and then the
    restoration is the edge that completes a cycle. A rollback is a forward write like
    any other, so it is checked like one -- against the settled tree plus the queued
    edges -- and when it would cycle the target is RELEASED to the root instead. That
    loses the old nesting, which is visible and repairable; persisting a cycle is
    neither, because the fold can only mark it and no verb undoes an append.

    The rollback is ANNOUNCED like a forward append, not only checked like one. It takes
    a settlement-backed waiter, which fences the slot and -- for the adoption, the only
    branch that adds an edge -- publishes ``slot -> previous parent`` in
    :data:`_tree_pending_edges`. Checking the rollback against the queued edges while not
    publishing it among them closes this function's own hazard and leaves the mirror
    image open: the next mover's cycle check would read a tree without the edge this one
    is about to write, and the loop forms from the other side.

    Both the fence and the edge outlive the wait below. They are cleared by the waiter's
    own future, so a wait that runs out leaves them standing until the writer actually
    settles -- which is the point, because an append still queued is exactly the one a
    later decision must see.

    Bounded and non-raising: the caller is already carrying the refusal that brought it
    here, and a compensation that raised its own error would replace the reason the
    operation failed with the reason the undo did. False means the tree still records the
    new parent -- the one case a person has to be told about, which is the caller's to
    report, not this function's to decide. A bound that runs out answers False for the
    same reason the forward path treats it as undecided: nothing is on disk yet.
    """
    from kiro_crew.crew_log import emit as crew_log_emit

    if previous_parent and _restoring_would_cycle(slot_key, previous_parent):
        logger.warning(
            "session %s: putting it back under %s would close a tree cycle, so it is "
            "released to the root instead",
            slot_key,
            previous_parent,
        )
        previous_parent = ""

    if previous_parent:

        def _emit(settled: "Callable[[bool], None]") -> None:
            crew_log_emit.on_session_adopted(
                target_sid,
                slot=slot_key,
                parent_slot=previous_parent,
                parent_sid=previous_parent_sid,
                previous_parent_slot=parent_key,
                previous_parent_sid=parent_sid,
                on_settled=settled,
            )

        what = f"putting {slot_key} back under {previous_parent}"
    else:

        def _emit(settled: "Callable[[bool], None]") -> None:
            crew_log_emit.on_session_released(
                target_sid,
                slot=slot_key,
                previous_parent_slot=parent_key,
                previous_parent_sid=parent_sid,
                on_settled=settled,
            )

        what = f"putting {slot_key} back at the root"

    try:
        # The waiter FIRST, so the fence and the queued edge are registered before the
        # entry is handed over and there is no instant where the writer holds an edge
        # nothing announces. A release adds no edge, so it registers the fence alone.
        settled, landed = _tree_append_waiter(slot_key)
        if previous_parent:
            _record_pending_edge(slot_key, previous_parent, landed)
        _emit(settled)
        # SHIELDED, so the bound ends this wait and not the fence: cancelling the future
        # would resolve it, and the fence and the edge would clear while the append is
        # still queued -- the one moment a later mover must not decide without them.
        return await asyncio.wait_for(
            asyncio.shield(landed), timeout=crew_log_emit.COMMIT_WAIT_SECONDS
        )
    except asyncio.TimeoutError:
        logger.warning(
            "tree compensation: %s is still queued; the slot stays fenced until it " "settles",
            what,
        )
        return False
    except Exception:
        logger.warning("tree compensation: %s failed", what, exc_info=True)
        return False


#: The in-flight commit windows, held strongly so the loop cannot collect a task whose
#: only other reference was a shield the caller has stopped awaiting. Cleared by each
#: task's own done callback. The whole point of the shield is that THIS work outlives the
#: request, so the request's frame cannot be what keeps it alive.
_commit_windows: "set[asyncio.Task[None]]" = set()


async def _landed_then_committed(
    future: "asyncio.Future[bool]",
    *,
    state: "DashboardState",
    slot: "_ChatSlot",
    target_sid: str,
    commit: "Callable[[_ChatSlot, str], Awaitable[None]] | None",
    noun: str,
    parent_key: str,
    parent_sid: str,
    previous_parent: str,
    previous_parent_sid: str,
    on_committed: "Callable[[], None] | None" = None,
    release_parent_fence: "Callable[[], None] | None" = None,
) -> None:
    """Wait for the tree append, then run *commit*, and put the parent back if it refuses.

    The two halves of a move that has to agree. Without this, a refused commit leaves the
    edge it was ordered behind standing on disk: the target hangs under the recipient, the
    giver is still its manager, and the giver's ``release`` answers ``not_parent`` -- so
    the move is neither done nor undoable by either session. The compensation closes that
    by restoring the previous parent before the refusal is reported.

    It runs in its OWN task, awaited through a shield, so the window settles independently
    of the caller. A request that goes away -- a disconnect, a tool timeout, a cancelled
    task group -- cancels the await and not the work: the entry is already queued, the
    commit may already have published, and abandoning the window midway is what would
    leave the halves apart. The slot stays fenced for as long as the task runs, which is
    what keeps the next verb out of a window the lock does not cover: a cancelled caller
    unwinds out of ``async with`` and releases the lock while this task is still going.

    The CALLER's wait is bounded and the window's is not the same bound, which is the
    other half of the same idea. A request cannot be held for as long as a writer may
    take, so it is answered "still pending, retryable" on :data:`_TREE_APPEND_TIMEOUT`
    while the task goes on waiting for the real verdict up to
    :data:`_TREE_SETTLE_TIMEOUT`. A window that ended on the caller's bound would end
    with the edge still queued and the commit never attempted, which is the split this
    whole function exists to prevent.

    *on_committed* is the verb's "it happened" side effect -- the SEL's ``allowed`` line
    and the sidebar broadcast. It runs HERE rather than in the caller's frame, because
    the caller's frame may be gone: a commit that publishes after the request timed out
    would otherwise leave the audit holding only that timeout's denial. Run exactly once,
    on the path where the operation really took effect.

    *release_parent_fence* drops the fence held on the OTHER endpoint, and the window
    owns that too, for the same reason it owns the audit: the fence has to outlive the
    request and go when the work does.

    *commit* of ``None`` is :func:`adopt_target`: the append IS the whole operation, there
    is nothing to disagree with it, and the wait stays the plain one.
    """
    if commit is None:
        try:
            await _tree_append_landed(future, operation=noun, target=slot.key)
        finally:
            if release_parent_fence is not None:
                release_parent_fence()
        if on_committed is not None:
            on_committed()
        return

    async def _compensate(why: str) -> None:
        restored = await _restore_parent(
            target_sid=target_sid,
            slot_key=slot.key,
            parent_key=parent_key,
            parent_sid=parent_sid,
            previous_parent=previous_parent,
            previous_parent_sid=previous_parent_sid,
        )
        if not restored:
            logger.error(
                "session %s: the %s %s, and the edge could not be put back -- the tree "
                "may record %s as its parent while its manager did not move",
                slot.key,
                noun,
                why,
                parent_key,
            )

    async def _window() -> None:
        landed = await _tree_append_verdict(future, target=slot.key, timeout=_TREE_SETTLE_TIMEOUT)
        if landed is None:
            # Undecided: the append is STILL QUEUED and may land after this window has
            # given up, with the commit never attempted. The correction is queued now,
            # behind it, because the queue is FIFO per unit -- so whatever order the two
            # land in, the newest statement on the log is the one that puts the parent
            # back, and a start cannot read the edge as a move that happened.
            await _compensate("was still queued when its window stopped waiting")
            raise SessionControlError(
                f"the {noun} of {name_for_person(state, slot)!r} was handed to the "
                "crew-log writer but has "
                "not landed yet, so it was not applied -- read the session tree before "
                "retrying",
                status=409,
                code="tree_write_pending",
            )
        if not landed:
            # Refused for good. Nothing reached the disk, so there is nothing to put
            # back and the commit is correctly never attempted.
            raise SessionControlError(
                f"the {noun} of {name_for_person(state, slot)!r} could not be written "
                "to the crew log, so "
                "the session tree is unchanged",
                status=500,
                code="tree_write_failed",
            )
        # NO await between the unfence the verdict just did and the hold below, so the
        # slot is never unfenced at a point another verb could be scheduled onto it.
        release_fence = _hold_tree_fence(slot.key)
        try:
            await commit(slot, target_sid)
        except BaseException:
            await _compensate("was refused after its tree edge landed")
            raise
        finally:
            release_fence()
        if on_committed is not None:
            on_committed()

    async def _fenced_window() -> None:
        try:
            await _window()
        finally:
            if release_parent_fence is not None:
                release_parent_fence()

    task = asyncio.ensure_future(_fenced_window())
    _commit_windows.add(task)

    def _finished(done: "asyncio.Task[None]") -> None:
        _commit_windows.discard(done)
        # RETRIEVED unconditionally, because the caller may not be there to take it: an
        # abandoned shield leaves the task's exception unread and the loop then reports it
        # as never retrieved at collection time, which is noise about a refusal this
        # function already handled.
        if not done.cancelled():
            done.exception()

    task.add_done_callback(_finished)
    try:
        # SHIELDED and bounded: the bound ends the CALLER's wait, and the shield is what
        # keeps it from ending the window with it.
        await asyncio.wait_for(asyncio.shield(task), timeout=_TREE_APPEND_TIMEOUT)
    except asyncio.TimeoutError:
        raise SessionControlError(
            f"the {noun} of {name_for_person(state, slot)!r} is still being written "
            "and is not confirmed "
            "yet -- read the session tree before retrying",
            status=409,
            code="tree_write_pending",
        ) from None


def _audit(
    *,
    caller_session_key: str,
    operation: str,
    slot_key: str,
    outcome: str,
    detail: dict[str, Any] | None = None,
) -> None:
    """Record one completed session-control operation in the SEL.

    Logged as a tool invocation rather than an API access because that is what
    it is from the caller's side, and because it carries the per-call detail
    (how the message landed, whether a stop escalated) that makes the audit
    line answer "what actually happened to the other session".

    Dispatched OFF the loop when one is running, mirroring
    ``update_metadata_off_loop``. ``log_tool_invocation`` only enqueues, but the
    FIRST ``sel()`` of a process CONSTRUCTS the log -- trust-dir creation and
    key validation, blocking file IO -- and this can genuinely
    be that first call: ``sel_audit_middleware`` logs AFTER ``await handler(...)``,
    so on a fresh gateway the first authenticated request constructs the log
    inside whatever handler runs first. Offloading here covers every call site
    without adding a step to the boot path -- which the boot-path rule forbids and
    a background prewarm would only race rather than close.
    """

    def _do() -> None:
        sel().log_tool_invocation(
            session_key=caller_session_key,
            agent="",
            source="mcp",
            tool_name=f"session_{operation}",
            tool_kind="command",
            outcome=outcome,
            resources=f"target={slot_key}",
            metadata=dict(detail or {}, target=slot_key),
        )

    _sel_off_loop(_do, "session-control audit")


def _sel_off_loop(write: "Callable[[], None]", what: str) -> None:
    """Run one SEL write off the event loop, best-effort.

    Shared by every session-control SEL write so the property holds in one place
    instead of per call site -- the denial audit was the THIRD site of this class
    to be found separately, having been missed while the other two were fixed.

    Two failure modes, both handled: a loop-blocking construct (a ``sel()`` that
    creates the trust dir and validates keys — blocking file IO), and a construct
    that RAISES, which unguarded turns a 403 into a
    500 -- losing the refusal in order to report it. An audit that cannot be
    written must never change what the caller is told.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is None:
        try:
            write()
        except Exception:  # noqa: BLE001 - an audit failure must not fail the op
            logger.warning("%s failed inline", what, exc_info=True)
        return

    def _report(fut: "asyncio.Future[None]") -> None:
        exc = fut.exception()
        if exc is not None:
            logger.warning("%s failed off-loop: %r", what, exc)

    loop.run_in_executor(None, write).add_done_callback(_report)


async def stop_target(
    state: "DashboardState",
    *,
    caller_session_key: str,
    target: str,
    caller_fenced: bool | None = None,
) -> dict[str, Any]:
    """Stop *target*'s in-flight turn, via the same path as the Stop button.

    A stop cancels cooperatively. The button escalates to a hard kill when a
    second press lands while the first is still pending; this verb deliberately
    does not do that for a repeat it cannot tell apart from a RETRY, because a
    client that got no response inside its request timeout re-sends the same
    request, and the kill path discards the target's queue and pending steers. So
    within ``stop_retry.WINDOW_SECS`` of this caller's first stop of this target, a
    repeat returns the existing "stop already in progress" no-op instead. A stop
    arriving after that window still escalates, so a genuine second decision keeps
    the capability — only a blind retry cannot reach it.

    Withholding the escalation never costs the caller the stop it asked for: a
    repeat that finds the target running again soft-stops it as a first call would.

    Still no force flag: escalation is decided by the target's own stop state and
    the window above, never by anything the caller can ask for, so advertising one
    would promise a hard kill a first call cannot deliver.

    ``caller_fenced`` is the ownership-fence verdict the HTTP gate carries for a
    caller it admitted as a crew member (``True``); ``None`` for every other
    caller. Forwarded to :func:`authorize_target` as
    ``precomputed_ownership_fenced`` — see there for why the verified admission
    travels with the request instead of being re-derived from config at the fence.
    The same parameter, with the same meaning, is on :func:`close_target`,
    :func:`send_to_target` and :func:`read_messages`.
    """
    # Prewarmed BEFORE `authorize_target`, and that ordering is load-bearing.
    # `stop_slot_turn`'s IDLE branch logs to the SEL with no await before it, so on
    # a fresh gateway a first `session_stop` against an idle slot would CONSTRUCT
    # the log on the loop -- trust-dir creation and key validation, blocking file
    # IO. Constructing it off-loop first makes that call a cheap cache hit.
    # Per-request, not a boot step: prewarming at startup is what
    # `no-new-work-on-gateway-boot-path` forbids, and a background task would only
    # narrow the race rather than close it.
    #
    # It must sit ABOVE the gate because `await` is a suspension point: between
    # `authorize_target` and `stop_slot_turn` the loop must not yield, or a user
    # action landing in that window (linking the target to a channel) makes the
    # decision stale and the `mirrored_target` refusal is bypassed -- the turn gets
    # cancelled on a session that became channel-backed after the check passed.
    # Nothing may suspend between this gate and the act it authorizes.
    #
    # Best-effort on purpose: construction can raise (a trust root too short to
    # sign the chain), and this is a latency guard, not an authorization one --
    # failing it must not turn a stop into a 500.
    try:
        await asyncio.to_thread(sel)
    except Exception:  # noqa: BLE001 - a prewarm failure must not fail the stop
        logger.warning("session-control SEL prewarm failed", exc_info=True)

    # The config warm goes HERE, not in the handler: the SEL prewarm above is an
    # `await`, and so is reading the request body, so a warm done before either of
    # them can be invalidated by a config edit landing in the gap -- leaving
    # `authorize_target`'s synchronous `session_control_enabled` to re-read and
    # validate the file on the loop, which is the whole thing the warm exists to
    # avoid. This is the last suspension before the gate.
    await prewarm_enabled_check()

    slot = authorize_target(
        state,
        caller_session_key=caller_session_key,
        target=target,
        operation="stop",
        precomputed_ownership_fenced=caller_fenced,
    )
    # Both calls below are SYNCHRONOUS, which is what lets them sit here at all:
    # the rule the comment above states is that nothing may SUSPEND between the
    # gate and the act, and neither of these does.
    #
    # `caller_slot_key` repeats the slot walk `authorize_target` just did rather
    # than changing what that function returns for all three verbs. The walk is
    # bounded by `MAX_LIVE_SLOTS` and touches no filesystem, and with no
    # suspension between them the two resolutions cannot disagree — a rebind
    # landing in that window is impossible, not merely unlikely.
    caller_key = caller_slot_key(state, caller_session_key)
    may_escalate = allow_escalation(caller_key, slot.key)
    # Deferred: ``chat_handlers`` imports ``dashboard.chat`` transitively, which
    # reaches back into the gateway at import time — a module-scope import here
    # closes that cycle through ``handlers.session_control`` -> ``server``.
    from kiro_crew.dashboard.chat_handlers import stop_slot_turn

    result = await stop_slot_turn(state, slot, source="session_control", escalate=may_escalate)
    _audit(
        caller_session_key=caller_session_key,
        operation="stop",
        slot_key=slot.key,
        outcome="allowed",
        detail={
            "result": result.get("info", "stopping"),
            # Recorded on the ALLOWED line, not only inside `stop_slot_turn`'s
            # own audit: this is the layer that made the retry judgement, so the
            # session-control trail has to show it was made.
            "escalation_withheld": not may_escalate,
        },
    )
    return {"ok": True, "target": slot.key, **result}


async def retry_target(
    state: "DashboardState",
    *,
    caller_session_key: str,
    target: str,
    caller_fenced: bool | None = None,
) -> dict[str, Any]:
    """Re-run *target*'s failed last turn, via the same path as the Resume button.

    Reuses ``chat_handlers.continue_slot_turn``, the mechanism behind
    ``POST /api/chat/slots/{slot}/continue``: the recovery continuation is queued
    at the head and dispatched, so the transcript gains the same recovery row a
    Resume press leaves and no second copy of the user's message. Every refusal
    Continue makes applies unchanged, including ``session_start_repeat`` after
    two identical failed starts.

    One refusal is this verb's own: ``turn_not_failed`` when
    ``is_turn_interrupted`` says the last turn finished or was ended with Stop.
    Continue carries on from a finished turn; a retry must not, or one session
    could make another produce a fresh answer on top of a good one.

    Not readiness-gated, for the reason ``api_chat_slot_continue`` gives: it
    mutates nothing durable before dispatch, so the turn's own start is the
    authority and a signed-out install reports in the transcript.

    ``caller_fenced`` has the meaning :func:`stop_target` documents.
    """
    slot_key, _body = await _continue_turn_of_target(
        state,
        caller_session_key=caller_session_key,
        target=target,
        caller_fenced=caller_fenced,
        operation="retry",
        require_interrupted=True,
    )
    return {"ok": True, "target": slot_key}


async def continue_target(
    state: "DashboardState",
    *,
    caller_session_key: str,
    target: str,
    caller_fenced: bool | None = None,
) -> dict[str, Any]:
    """Hand *target*'s thread back to its agent, as the Continue button does.

    The same call as :func:`retry_target` without ``require_interrupted``: a
    force-quit turn runs no ``finally`` and leaves no error row, so it reads as
    finished, and only this verb can pick it up. ``continue_slot_turn`` chooses
    the body exactly as it does for the button, and the result reports which one
    it queued in ``body``: ``"resumed"`` (the interrupted pick-up) or
    ``"continued"`` (carry on from a turn that ended cleanly). No prompt text is
    accepted; the body is always one of the two fixed continuation messages.

    Not readiness-gated, for the reason :func:`retry_target` gives.
    ``caller_fenced`` has the meaning :func:`stop_target` documents.
    """
    slot_key, body = await _continue_turn_of_target(
        state,
        caller_session_key=caller_session_key,
        target=target,
        caller_fenced=caller_fenced,
        operation="continue",
        require_interrupted=False,
    )
    return {"ok": True, "target": slot_key, "body": body}


async def _continue_turn_of_target(
    state: "DashboardState",
    *,
    caller_session_key: str,
    target: str,
    caller_fenced: bool | None,
    operation: str,
    require_interrupted: bool,
) -> tuple[str, str]:
    """Authorize *target* and queue a continuation on it; the core of retry and continue.

    Returns the target's slot key and the body ``continue_slot_turn`` queued
    (``"resumed"`` or ``"continued"``). *operation* names the verb on every audit
    line and on the gate's refusals.
    """
    # Both verbs re-read the target's last turn: a target restored as a
    # sidebar row loads its window first, ahead of the prewarm below.
    await load_target_window(state, target)
    # Same prewarm ordering as `close_target`: the SEL and config reads must be
    # warm before the synchronous gate, and the fence verdict is resolved once so
    # the re-check under the slot lock never loads config on the loop.
    try:
        await asyncio.to_thread(sel)
    except Exception:  # noqa: BLE001 - a prewarm failure must not fail the call
        logger.warning("session-control SEL prewarm failed", exc_info=True)
    await prewarm_enabled_check()

    caller_key = caller_slot_key(state, caller_session_key)
    if caller_fenced is None:
        caller_fenced = bool(caller_key) and _caller_is_ownership_fenced(state, caller_key)

    slot = authorize_target(
        state,
        caller_session_key=caller_session_key,
        target=target,
        operation=operation,
        precomputed_ownership_fenced=caller_fenced,
    )
    slot_key = slot.key
    # Deferred for the same import cycle `stop_target` documents.
    from kiro_crew.dashboard.chat_handlers import SlotContinueRefusal, continue_slot_turn

    def _reassert_reachable() -> None:
        # The slot lock `continue_slot_turn` takes, and its sub-agent probe, are
        # awaits: a target that was reachable at the gate above can become
        # channel-linked or mirrored, or be replaced under the same key, before
        # the continuation is queued. Re-run the same gate here, synchronously
        # and with no config read, as `close_target` does at its point of no
        # return.
        try:
            live = authorize_target(
                state,
                caller_session_key=caller_session_key,
                target=slot_key,
                operation=operation,
                skip_enabled_check=True,
                precomputed_ownership_fenced=caller_fenced,
            )
        except SessionControlError as exc:
            raise SlotContinueRefusal(exc.message, exc.code, status=exc.status) from exc
        if live is not slot:
            raise SlotContinueRefusal(
                f"the target session was replaced during the {operation}", "target_replaced"
            )

    with _audit_denials(
        caller_session_key=caller_session_key, operation=operation, slot_key=slot_key
    ):
        # The same predicate the Continue endpoint refuses on, so the button and
        # the verbs cannot drift: a relay archive would run the crew's old turn on
        # this machine.
        if is_relay_archive(slot):
            raise SessionControlError(RELAY_ARCHIVE_ERROR, code=RELAY_ARCHIVE_CODE, status=409)
        try:
            body = await continue_slot_turn(
                state,
                slot,
                # An agent asked for this, not the person at the keyboard: the
                # turn must not gain the authenticated-human flag.
                directive_user_origin=False,
                via="session_control",
                require_interrupted=require_interrupted,
                before_dispatch=_reassert_reachable,
                # FALSE, and not this helper's fence verdict. The stamp records
                # whether the MANAGER fence applied to this admission, which is
                # what licenses the drain to re-compare the sender against the
                # manager -- and NEITHER verb this helper serves is in
                # :data:`HANDOFF_WIDENED_OPERATIONS`, so both gates authorized on
                # ``_created_by`` alone and never read the manager at all.
                # Claiming otherwise lets the drain drop an accepted continuation
                # as ``sender_no_longer_manages`` against a predicate that was
                # never applied to it, after the verb has already answered ok --
                # the caller told the retry ran while nothing ran. The
                # target-side containment re-check is unaffected: it travels in
                # its own keys and is re-asserted whatever this says.
                extra_meta=send_origin_meta(state, caller_key, manager_fenced=False),
            )
        except SlotContinueRefusal as exc:
            raise SessionControlError(exc.message, status=exc.status, code=exc.code) from exc
    _audit(
        caller_session_key=caller_session_key,
        operation=operation,
        slot_key=slot_key,
        outcome="allowed",
        detail={"body": body},
    )
    return slot_key, body


async def end_wait_target(
    state: "DashboardState",
    *,
    caller_session_key: str,
    target: str,
    caller_fenced: bool | None = None,
) -> dict[str, Any]:
    """Wake *target* from the ``wait`` tool early, keeping its turn.

    The same mechanism as the dashboard's End-wait button
    (``api_chat_slot_end_wait``): the request is parked on the slot as
    ``_end_wait_request`` and the sleeping tool collects it on its next keepalive
    ping, then returns a normal tool result. Nothing is cancelled and no work is
    discarded, which is what separates this verb from ``session_stop``.

    The caller does not name a ``wait_id``. The button needs one because a stale
    tab can still show an old countdown; here the id is read from the slot at the
    moment of the request, so the request can only ever name the sleep that is in
    flight now. ``_end_wait_by`` records who asked, so the keepalive reply can tell
    the woken session that another session ended its wait rather than the user.

    A target that is not sleeping is not an error: the reply carries ``info``
    instead, so a caller does not retry something that has nothing to act on.
    """
    # Same ordering as `stop_target`: both prewarms are awaits, so they sit
    # above the gate and nothing suspends between the gate and the write.
    try:
        await asyncio.to_thread(sel)
    except Exception:  # noqa: BLE001 - a prewarm failure must not fail the call
        logger.warning("session-control SEL prewarm failed", exc_info=True)
    await prewarm_enabled_check()

    slot = authorize_target(
        state,
        caller_session_key=caller_session_key,
        target=target,
        operation="end_wait",
        precomputed_ownership_fenced=caller_fenced,
    )
    caller_key = caller_slot_key(state, caller_session_key)
    if _created_by_other(slot, caller_key):
        # Narrower than the other verbs on purpose: an owner session is not
        # creator-fenced by `authorize_target`, but waking a sleep moves a turn
        # forward on someone else's schedule, and the caller that armed the
        # worker's wait is the one that knows when it is safe to end it.
        raise _deny_factory(
            caller_session_key=caller_session_key, operation="end_wait", target=target
        )("session_end_wait reaches only sessions you created", "not_creator")
    if getattr(slot, "_wait_contested", False):
        # Two sleeps share this slot's session key (see _service_wait_ping's
        # ambiguous-identity guard). There is no way to aim at one of them, and
        # the button is hidden for the same reason.
        info = "two waits share this session, so neither can be ended early"
        result = {"ended": False, "info": info}
        audit_result = "contested"
    else:
        current = getattr(slot, "_wait_state", None) or {}
        wait_id = str(current.get("wait_id") or "")
        if not wait_id:
            result = {"ended": False, "info": "not sleeping in the wait tool"}
            audit_result = "not_waiting"
        else:
            slot._end_wait_request = wait_id
            slot._end_wait_by = caller_key
            result = {"ended": True, "wait_id": wait_id}
            audit_result = "requested"
    _audit(
        caller_session_key=caller_session_key,
        operation="end_wait",
        slot_key=slot.key,
        outcome="allowed",
        detail={"result": audit_result},
    )
    return {"ok": True, "target": slot.key, **result}


async def set_model_target(
    state: "DashboardState",
    *,
    caller_session_key: str,
    target: str,
    model: str | None = None,
    reasoning_effort: str | None = None,
    caller_fenced: bool | None = None,
) -> dict[str, Any]:
    """Record *model* and/or *reasoning_effort* as *target*'s pending pick.

    ``None`` for either means "keep what the target has"; at least one must be
    given. The pick is applied when the target's next turn starts.

    Nothing about the target changes now. The pick is committed by
    :func:`apply_pending_model_pick` at the start of the target's next turn,
    which re-runs this same gate and writes ``slot.model`` in one synchronous
    step. Committing here instead would need the live model switch, whose
    provider awaits sit after the last gate: a channel link or mirror landing
    in that window would let the change reach a session the caller may no
    longer touch.

    Only an IDLE session takes a pick. A target with a turn or attached
    sub-agents in flight is refused with ``target_busy``; a caller that wants
    to force it stops the target first (``session_stop``) and retries. A later
    pick replaces an earlier one that has not been applied yet.

    Two picks the picker allows are refused here. "Auto (Jev)" arms per-turn
    routing, which the model route keeps owner-only. A crew-bound (remote)
    target runs its turns on the peer, where this pick would never be applied.

    A reasoning effort must be one of the five canonical levels (``low``,
    ``medium``, ``high``, ``xhigh``, ``max``), which every backend accepts;
    any other level, including ``""``, is refused here and is set from the
    effort dropdown instead. It is committed with the model in the same step.
    It is NOT pushed live the way that route pushes it: the
    turn-start reset makes the cold start apply it through the same
    ``reasoning_effort_override`` the route's reset fallback relies on, with no
    provider await after the gate.

    ``caller_fenced`` has the meaning :func:`stop_target` documents.
    """
    # Deferred for the same import cycle `stop_target` documents.
    from kiro_crew.dashboard.chat_handlers import (
        _is_jev_route_pick,
        _model_rejected_reason,
        _normalize_model,
        _subagents_attached_response,
        _switch_target_busy,
    )
    from kiro_crew.dashboard.chat_runner import _JEV_ROUTE_AUTO_MODELS

    # Validated before any gate: a malformed pick needs no target lookup, and
    # refusing it first keeps a bad argument from reading as an access decision.
    if model is None and reasoning_effort is None:
        raise SessionControlError(
            "pass a model, a reasoning_effort, or both", code="bad_request", status=400
        )
    if reasoning_effort is not None:
        if not is_valid_effort(reasoning_effort):
            # Only the five standard levels: each folds onto every backend's
            # cold start, so a level checked here cannot go stale before the
            # target's next turn applies it. A level only one harness advertises
            # (Pi's ``minimal``, Claude's ``default``) is set from that session's
            # effort dropdown, which reads its live list.
            #
            # "" (back to the model default) is refused too: on a kiro-family
            # backend the default only takes once the workspace effort overlay
            # is cleared, which only the dropdown route's live clear_effort does.
            # Committed here, the cold start would recover the old level from
            # the overlay while every surface said "default".
            raise SessionControlError(
                f"reasoning_effort must be one of: {', '.join(EFFORT_LEVELS)}. "
                "Other levels, and returning a session to its model's default, are "
                "set from its effort dropdown.",
                code="effort_rejected",
                status=400,
            )
    model_name: str | None = None
    if model is not None:
        # Stripped once, so every check and the stored pick see the same spelling.
        model = model.strip()
        # "auto" is refused with the Jev sentinel: with the Jev preview on, a slot
        # on "auto" hands each turn's model choice to Jev routing, which only the
        # owner may arm. The picker's display label and any case of the sentinel
        # are refused the same way, so no spelling of the Jev entry is stored.
        # An empty name is the absence of a pick, not a model.
        folded = model.lower()
        if (
            _is_jev_route_pick(model)
            or folded in _JEV_ROUTE_AUTO_MODELS
            or folded in _JEV_ROUTE_SPELLINGS
        ):
            raise SessionControlError(
                "Auto and Auto (Jev) can only be picked by the owner from the model picker",
                code="model_owner_only",
                status=403,
            )
        if not model:
            raise SessionControlError("model is required", code="model_rejected", status=400)
        if redact(model) != model:
            # The pick is stored on the slot and broadcast to every dashboard, so a
            # credential-shaped argument is refused rather than persisted.
            raise SessionControlError(
                "model looks like it contains a credential; model not changed",
                code="model_rejected",
                status=400,
            )
        model_name = _normalize_model(model)
    try:
        agent_cfg = (await asyncio.to_thread(KiroCrewConfig.load)).agent
        provider = agent_cfg.provider
        member_backend, default_backend = agent_cfg.member_acp_backend, agent_cfg.acp_backend
    except Exception:  # pragma: no cover - config load is resilient
        provider, member_backend, default_backend = "", "", ""

    # Same prewarm ordering and fence-verdict handling as `close_target`: the
    # verdict is stored with the pick so the turn-start re-check reads no config.
    try:
        await asyncio.to_thread(sel)
    except Exception:  # noqa: BLE001 - a prewarm failure must not fail the switch
        logger.warning("session-control SEL prewarm failed", exc_info=True)
    await prewarm_enabled_check()
    caller_key = caller_slot_key(state, caller_session_key)
    if caller_fenced is None:
        caller_fenced = bool(caller_key) and _caller_is_ownership_fenced(state, caller_key)

    slot = authorize_target(
        state,
        caller_session_key=caller_session_key,
        target=target,
        operation="set_model",
        precomputed_ownership_fenced=caller_fenced,
    )
    slot_key = slot.key
    # The model check and the alias correction both key on the TARGET's backend,
    # resolved through the same member-aware gate the provider factory uses: a
    # member DM routes to agent.member_acp_backend, which can differ from the
    # configured default, and a Claude backend takes canonical keys as wire ids.
    backend = select_provider_backend(effective_session_key(slot), member_backend, default_backend)
    target_provider = (
        provider if is_claude_code(provider) else capabilities_for(backend).provider_seam
    )
    rejected = (
        _model_rejected_reason(model_name, provider=target_provider)
        if model_name is not None
        else ""
    )
    if rejected:
        raise SessionControlError(rejected, code="model_rejected", status=400)
    if (
        model_name is not None
        and not is_claude_code(target_provider)
        and capabilities_for(backend).model_id_namespace == MODEL_NAMESPACE_ACP
    ):
        # On a kiro-cli backend an alias such as "sonnet" is not a wire id: stored
        # as-is it would be withheld at session start and the target would stay
        # on the default. Other backends keep their own id namespace untouched.
        model_name = model_registry.acp_id_correction(model_name) or model_name

    with _audit_denials(
        caller_session_key=caller_session_key, operation="set_model", slot_key=slot_key
    ):
        if is_relay_archive(slot):
            raise SessionControlError(RELAY_ARCHIVE_ERROR, code=RELAY_ARCHIVE_CODE, status=409)
        # No restore-allowlist marker is written for the level: the five
        # canonical levels are always restorable (`_REASONING_EFFORT_FALLBACK`),
        # so the marker the effort dropdown writes for other levels is a no-op.
        session_key = effective_session_key(slot)
        if _switch_target_busy(state, slot, session_key, state.sessions.get_provider(session_key)):
            raise _target_busy_error()
        children = await _subagents_attached_response(state, slot, session_key, "set_model")
        if children is not None:
            raise _target_busy_error()
        # The await above can let a turn start, or the target be replaced,
        # linked or mirrored; re-run the idle check and the gate synchronously
        # right before storing the pick.
        session_key = effective_session_key(slot)
        if _switch_target_busy(state, slot, session_key, state.sessions.get_provider(session_key)):
            raise _target_busy_error()
        # A model-picker switch in flight holds _model_pick_lock across its
        # provisional pick-generation bump and a possible rollback. Capturing
        # the generation inside that window would store a value the rollback
        # then invalidates, and the next turn would silently drop the pick.
        # Nothing awaits between here and the store below, so this check
        # holds until the pick is recorded.
        if slot._model_pick_lock.locked():
            raise _target_busy_error()
        # The effort dropdown holds the per-session switch lock across its
        # commit, its live push and any rollback. A level captured inside that
        # window would record a generation the rollback then bumps, and the
        # next turn would silently drop the pick, so refuse it the same way.
        if reasoning_effort is not None and slot_switch_session_lock(session_key).locked():
            raise _target_busy_error()
        live = authorize_target(
            state,
            caller_session_key=caller_session_key,
            target=slot_key,
            operation="set_model",
            skip_enabled_check=True,
            precomputed_ownership_fenced=caller_fenced,
        )
        if live is not slot:
            raise SessionControlError(
                "the target session was replaced; model not changed",
                code="target_replaced",
                status=409,
            )
        caller_tab_id = _caller_tab_id(state, caller_session_key)
        if not caller_tab_id:
            raise SessionControlError(
                "this session has no tab identity to hold a pending pick; model not changed",
                code="caller_unidentified",
                status=403,
            )
        slot._pending_model_pick = PendingModelPick(
            model=model_name,
            caller_session_key=caller_session_key,
            caller_tab_id=caller_tab_id,
            caller_fenced=caller_fenced,
            pick_gen=slot._model_pick_gen,
            reasoning_effort=reasoning_effort,
            effort_gen=slot._effort_pick_gen,
            pair_id_backend=backend in ACP_BACKENDS_MODEL_EFFORT_PAIR_IDS,
        )

    requested = _pick_fields(model_name, reasoning_effort)
    _audit(
        caller_session_key=caller_session_key,
        operation="set_model",
        slot_key=slot_key,
        outcome="allowed",
        detail={**_pick_audit_fields(model_name, reasoning_effort), "stage": "pending"},
    )
    return {"ok": True, "target": slot_key, **requested, "pending": True}


def _pick_fields(model: str | None, reasoning_effort: str | None) -> dict[str, str]:
    """The pick's requested fields, omitting the ones the caller left alone."""
    fields: dict[str, str] = {}
    if model is not None:
        fields["model"] = model
    if reasoning_effort is not None:
        fields["reasoning_effort"] = reasoning_effort
    return fields


def _pick_audit_fields(model: str | None, reasoning_effort: str | None) -> dict[str, str]:
    """:func:`_pick_fields` spelled for the audit trail, where "" reads as a default."""
    fields: dict[str, str] = {}
    if model is not None:
        fields["model"] = model or "auto"
    if reasoning_effort is not None:
        fields["reasoning_effort"] = reasoning_effort or "default"
    return fields


#: Stored cap on a renamed title, the same ``[:200]`` the sidebar rename route and
#: :func:`create_session` keep. The MCP schema's own bound on the argument is the
#: generic ``MAX_SHORT_STRING`` (500), so :func:`_clean_rename_title` is what
#: refuses a title between 201 and 500 characters, for MCP and direct callers alike.
_MAX_RENAME_TITLE_CHARS = 200

#: Every Unicode text-direction control, refused in a renamed title: the
#: embeddings, overrides and isolates (U+202A..U+202E, U+2066..U+2069) and the
#: directional marks (U+200E, U+200F, U+061C).
_BIDI_CONTROLS = frozenset(
    "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\u200e\u200f\u061c"
)

#: Characters that render as a blank glyph yet are neither whitespace nor
#: invisible to :func:`kiro_crew.terminal_safe._is_invisible`, which already
#: covers format characters, variation selectors, the Hangul and Khmer fillers
#: and the tag block. The blank Braille pattern is the one such character: a
#: printable symbol that draws nothing.
_BLANK_GLYPHS = frozenset("\u2800")

#: Format characters (category ``Cf``) a renamed title may contain: the zero-width
#: non-joiner and joiner, which emoji sequences and joined scripts such as Persian
#: and Devanagari need. The tag characters U+E0020..U+E007F are allowed
#: separately, by range; the tag block's other format character, U+E0001
#: LANGUAGE TAG, is refused. Its unassigned code points are not ``Cf`` and are
#: not refused by this rule.
_ALLOWED_FORMAT_CHARS = frozenset("\u200c\u200d")


def _is_allowed_format_char(ch: str) -> bool:
    """True for a format character a renamed title keeps: a joiner, or a tag
    character (U+E0020..U+E007F), which spells the subdivision of a flag emoji
    such as England's. Every other ``Cf`` character is refused."""
    return ch in _ALLOWED_FORMAT_CHARS or "\U000e0020" <= ch <= "\U000e007f"


def _clean_rename_title(title: str) -> str:
    """Validate *title* for :func:`rename_target` and return the stored form.

    The rules are the sidebar's: surrounding whitespace is stripped, an empty
    result is refused, and nothing over 200 characters is stored. Two refusals
    go further than the sidebar, because the sidebar's text box can produce
    neither case and a model can: a title over the cap is refused instead of cut
    down, and a control character (a newline, a tab, an escape) is refused
    because a sidebar title is one line.

    The stored form is the scrubbed title, which can differ from what the caller
    sent: ``handlers._shared._scrub_text`` replaces credential-shaped text with a
    longer marker, also catching a secret split by an invisible character. A
    title that grows past the cap that way is refused, never cut, so no stored
    title holds a torn marker. The verb's reply carries this stored form, so the
    caller reads back what the sidebar shows.
    """
    stripped = title.strip()
    if not stripped:
        raise SessionControlError("title is required", code="invalid_title", status=400)
    if len(stripped) > _MAX_RENAME_TITLE_CHARS:
        raise SessionControlError(
            f"title is longer than {_MAX_RENAME_TITLE_CHARS} characters; session not renamed",
            code="invalid_title",
            status=400,
        )
    if any(unicodedata.category(ch) == "Cc" or ch in "\u2028\u2029" for ch in stripped):
        raise SessionControlError(
            "title contains a control character; session not renamed",
            code="invalid_title",
            status=400,
        )
    # A direction control makes the sidebar render the label in an order other
    # than the one it was written in.
    if any(ch in _BIDI_CONTROLS for ch in stripped):
        raise SessionControlError(
            "title contains a text-direction control; session not renamed",
            code="invalid_title",
            status=400,
        )
    # Any other format character (a zero-width space, a word joiner, a byte-order
    # mark, a soft hyphen) is invisible inside a visible title, so it can make a
    # near-duplicate of another session's label. Only the format characters that
    # emoji and joined scripts need are kept: see :func:`_is_allowed_format_char`.
    if any(unicodedata.category(ch) == "Cf" and not _is_allowed_format_char(ch) for ch in stripped):
        raise SessionControlError(
            "title contains an invisible format character; session not renamed",
            code="invalid_title",
            status=400,
        )
    # A title needs at least one character that shows something. Invisible
    # characters (a zero-width space or joiner, a variation selector, a Hangul
    # filler) survive `str.strip()`, so a title made only of them, or of blank
    # glyphs, would store an invisible, final name the automatic titler can
    # never replace. Such characters are kept inside a visible title: a joiner
    # is how an emoji sequence is written.
    if all(ch.isspace() or _is_invisible(ch) or ch in _BLANK_GLYPHS for ch in stripped):
        raise SessionControlError("title is required", code="invalid_title", status=400)
    # A blank glyph is refused anywhere in the title, not only in an all-blank
    # one. `terminal_safe.normalize_for_scanning` does not remove it (it is a
    # printable symbol, not an invisible character), so one placed inside a
    # secret ("AKIA\u2800...") would split it past the redactors that
    # `_scrub_text` runs below.
    if any(ch in _BLANK_GLYPHS for ch in stripped):
        raise SessionControlError(
            "title contains a blank glyph (U+2800); session not renamed",
            code="invalid_title",
            status=400,
        )
    # Sanitized with the stored-field scrub the dashboard handlers use: the title
    # is broadcast to every dashboard and persisted with the transcript. The
    # redactors match literal patterns, so an invisible character placed inside
    # a secret ("AKIA\u200d...", "AKIA\ufe0f...") would split it past the match;
    # `_scrub_text` also scans a normalized copy and keeps it when that copy
    # reveals a secret, and otherwise leaves invisible characters in place, so
    # an emoji sequence is unchanged. Its control-character step is inert here,
    # because a control character was refused above.
    scrubbed = _scrub_text(stripped)
    # Redaction markers are longer than the text they replace. A title they push
    # past the cap is refused like any other over-long title rather than cut,
    # so a stored title never holds a torn marker. The message names no part of
    # the title, so the redacted text is not echoed back.
    if len(scrubbed) > _MAX_RENAME_TITLE_CHARS:
        raise SessionControlError(
            f"title is longer than {_MAX_RENAME_TITLE_CHARS} characters once "
            "credential-shaped text in it is redacted; session not renamed",
            code="invalid_title",
            status=400,
        )
    return scrubbed


async def rename_target(
    state: "DashboardState",
    *,
    caller_session_key: str,
    target: str,
    title: str,
    caller_fenced: bool | None = None,
) -> dict[str, Any]:
    """Rename *target*'s sidebar title, the way the sidebar's own rename does.

    The reach is :func:`stop_target`'s plus the caller itself. For a target that
    is not the caller, every rule in :func:`authorize_target` applies, including
    the ownership fence that keeps a crew member, a cron run, an owner-DM channel
    session and any agent-created session to the sessions it created. For the
    caller itself, ``allow_self`` waives the self-target refusal and, with it,
    that ownership fence, since a session naming itself acts on nothing but its
    own sidebar label. It waives nothing else: an ephemeral, app-scoped or
    channel-linked caller is still refused, and so is a cron run's own session,
    because the unattended-target refusal is checked before ``allow_self``.

    The rename is final (``_TITLE_ORIGIN_USER``), so the automatic titler stops
    refreshing the target, exactly as after a manual rename. It changes nothing
    else about the target and does not wait for a running turn.

    ``caller_fenced`` has the meaning :func:`stop_target` documents.
    """
    # Validated before any gate, as `set_model_target` does: a malformed title
    # needs no target lookup, and refusing it first keeps a bad argument from
    # reading as an access decision.
    clean_title = _clean_rename_title(title)
    # Same prewarm ordering as `stop_target`: nothing may suspend between the
    # gate and the write it authorizes.
    try:
        await asyncio.to_thread(sel)
    except Exception:  # noqa: BLE001 - a prewarm failure must not fail the rename
        logger.warning("session-control SEL prewarm failed", exc_info=True)
    await prewarm_enabled_check()

    slot = authorize_target(
        state,
        caller_session_key=caller_session_key,
        target=target,
        operation="rename",
        precomputed_ownership_fenced=caller_fenced,
        allow_self=True,
    )
    # `apply_manual_title` assigns every field before its first await, so the
    # title is on the slot before this coroutine can suspend.
    if not await apply_manual_title(state, slot, clean_title):
        _audit(
            caller_session_key=caller_session_key,
            operation="rename",
            slot_key=slot.key,
            outcome="error",
            detail={"chars": len(clean_title), "reason": "title_persist_failed"},
        )
        raise SessionControlError(
            f"{slot.key!r} now shows the new title, but it could not be saved yet; "
            "the session's next save writes it, and until then a restart brings back "
            "the previous title. Renaming again with the same title is safe",
            status=500,
            code="title_persist_failed",
        )
    _audit(
        caller_session_key=caller_session_key,
        operation="rename",
        slot_key=slot.key,
        outcome="allowed",
        detail={"chars": len(clean_title)},
    )
    # The slot's title, not this call's: a rename that overtook this one while it
    # saved has already replaced it, and the caller acts on what the reply names.
    return {"ok": True, "target": slot.key, "title": slot.title}


def _admit_generate_title(slot_key: str) -> None:
    """Spend *slot_key*'s one title generation for this window, or refuse."""
    if not allow_create(GENERATE_TITLE, slot_key):
        raise SessionControlError(
            f"{slot_key!r} had its title generated within the last "
            f"{int(WINDOW_SECS)} seconds; try again later",
            status=429,
            code="title_rate_limited",
        )


async def generate_title_target(
    state: "DashboardState",
    *,
    caller_session_key: str,
    target: str,
    caller_fenced: bool | None = None,
) -> dict[str, Any]:
    """Regenerate *target*'s sidebar title, the way "Regenerate title" does.

    Runs the sidebar's own code (:func:`chat_title.generate_manual_title` and
    :func:`chat_title.apply_regenerated_title`), so the result is the one a
    person clicking it would get: one titling-model call, awaited before this
    returns, and a new name that replaces the current one (a hand-set title
    included) and stays "auto" so the background refresh may revise it.

    Reach is narrower than :func:`stop_target`'s. After every
    :func:`authorize_target` rule, the target must be a session the caller
    created (``_created_by``), for EVERY caller class: the verb spends a model
    call to relabel a session, and a session the person opened is theirs to
    name. The caller itself is refused too.

    One admitted call per target per ``create_rate_limit.WINDOW_SECS``, stamped
    before the model call, so a patrol loop or a concurrent retry cannot spend
    twice. A call refused before the stamp spends nothing.

    The model call suspends, so the whole gate runs again, synchronously, after
    it answers and before anything changes: a target that was closed, replaced
    under the same key, linked to a channel or otherwise put out of reach in the
    meantime keeps its title (``target_replaced`` or the gate's own refusal).
    From there the title is assigned with no suspension in between, the same
    guarantee :func:`rename_target` gives; the save that follows only makes an
    already-authorized change durable.

    ``caller_fenced`` has the meaning :func:`stop_target` documents.
    """
    # Same prewarm ordering as `stop_target`: nothing may suspend between the
    # gate and the write it authorizes.
    try:
        await asyncio.to_thread(sel)
    except Exception:  # noqa: BLE001 - a prewarm failure must not fail the call
        logger.warning("session-control SEL prewarm failed", exc_info=True)
    await prewarm_enabled_check()

    operation = "generate_title"
    slot = authorize_target(
        state,
        caller_session_key=caller_session_key,
        target=target,
        operation=operation,
        precomputed_ownership_fenced=caller_fenced,
    )
    caller_key = caller_slot_key(state, caller_session_key)
    with _audit_denials(
        caller_session_key=caller_session_key, operation=operation, slot_key=slot.key
    ):
        if _created_by_other(slot, caller_key):
            raise SessionControlError(
                "session_generate_title only reaches sessions this session created",
                status=403,
                code="not_creator",
            )
        _admit_generate_title(slot.key)

    epoch = slot._title_epoch
    title = await generate_manual_title(state, slot)

    # Re-gate after the suspension. The creator rule was settled above and
    # ``_created_by`` belongs to this slot object, so the fence is passed as
    # settled and the switch read skipped: neither reads config on the loop.
    with _audit_denials(
        caller_session_key=caller_session_key, operation=operation, slot_key=slot.key
    ):
        if state._slots.get(slot.key) is not slot or slot.is_closing:
            raise SessionControlError(
                f"{slot.key!r} was closed while its title was being generated; "
                "nothing was changed",
                status=409,
                code="target_replaced",
            )
    authorize_target(
        state,
        caller_session_key=caller_session_key,
        target=slot.key,
        operation=operation,
        skip_enabled_check=True,
        precomputed_ownership_fenced=True,
    )

    # A rename that landed during the model call outranks a generated name, as
    # in the sidebar; an empty result means the model had nothing better.
    if not title or slot._title_epoch != epoch:
        _audit(
            caller_session_key=caller_session_key,
            operation=operation,
            slot_key=slot.key,
            outcome="allowed",
            detail={"applied": False},
        )
        return {"ok": True, "target": slot.key, "title": slot.title, "changed": False}

    previous = slot.title

    # `apply_regenerated_title` assigns every field before its first await, so
    # the title is on the slot before this coroutine can suspend.
    # `still_current` is re-checked under the history write lock, so a slot
    # closed and replaced under the same key while the write was queued never
    # stamps its title over the successor's saved one.
    def _still_current() -> bool:
        return state._slots.get(slot.key) is slot

    if not await apply_regenerated_title(state, slot, title, still_current=_still_current):
        if not _still_current():
            _audit(
                caller_session_key=caller_session_key,
                operation=operation,
                slot_key=slot.key,
                outcome="denied",
                detail={"applied": False, "reason": "target_replaced"},
            )
            raise SessionControlError(
                f"{slot.key!r} was closed while its title was being saved; " "nothing was changed",
                status=409,
                code="target_replaced",
            )
        _audit(
            caller_session_key=caller_session_key,
            operation=operation,
            slot_key=slot.key,
            outcome="error",
            detail={"applied": True, "reason": "title_persist_failed"},
        )
        raise SessionControlError(
            f"{slot.key!r} now shows the new title, but it could not be saved yet; "
            "the session's next save writes it, and until then a restart brings back "
            "the previous title",
            status=500,
            code="title_persist_failed",
        )
    _audit(
        caller_session_key=caller_session_key,
        operation=operation,
        slot_key=slot.key,
        outcome="allowed",
        detail={"applied": True},
    )
    # The slot's title, not this call's: a rename that overtook this one while it
    # saved has already replaced it, and the caller acts on what the reply names.
    return {
        "ok": True,
        "target": slot.key,
        "title": slot.title,
        "changed": slot.title != previous,
    }


@dataclass(frozen=True)
class PendingModelPick:
    """A ``session_set_model`` pick waiting for the target's next turn.

    Carries the caller's key AND tab identity, its ownership-fence verdict at
    call time, and the target's pick generation then, so a model the user picks
    in the meantime wins over this one. The tab identity is what ties the pick
    to the calling session: a slot key can be handed to a new occupant after
    the caller closes, and that occupant must not inherit the pick.

    ``model`` or ``reasoning_effort`` is ``None`` when the caller left that
    setting alone. ``effort_gen`` is the target's effort write generation at
    call time: any write since (the effort dropdown, a fork, a restore) means
    a newer choice landed, and it wins over this one even when it returned to
    the level the caller saw. ``pair_id_backend`` records that the target's
    backend spells effort into the model id (``gpt-6-astra[max]``), so a
    committed level must also fold a legacy suffix off the pin, as the effort
    dropdown route does.
    """

    model: str | None
    caller_session_key: str
    caller_tab_id: str
    caller_fenced: bool
    pick_gen: int
    reasoning_effort: str | None = None
    effort_gen: int = 0
    pair_id_backend: bool = False


def _pending_pick_view(pick: PendingModelPick | None) -> dict[str, str]:
    """The ``pending_*`` fields ``session_read_message`` reports for *pick*.

    Redacted: the owner's picker stores whatever string it is given.
    """
    if pick is None:
        return {}
    view: dict[str, str] = {}
    if pick.model is not None:
        view["pending_model"] = redact(pick.model)
    if pick.reasoning_effort is not None:
        view["pending_reasoning_effort"] = pick.reasoning_effort
    return view


def _caller_tab_id(state: "DashboardState", caller_session_key: str) -> str:
    """The calling slot's ``_tab_id``, or ``""`` when it has none or is gone."""
    caller_key = caller_slot_key(state, caller_session_key)
    caller = state._slots.get(caller_key) if caller_key else None
    return str(getattr(caller, "_tab_id", "") or "") if caller is not None else ""


def _another_alias_is_mid_turn(state: "DashboardState", slot: "_ChatSlot") -> bool:
    """Whether a slot OTHER than *slot* runs a turn on *slot*'s session.

    The alias half of ``_switch_target_busy``: *slot*'s own turn is the one
    starting, so its ``running`` flag is not a signal here. A registered
    provider with an active turn, or another running slot whose turn key
    names the same session, means the session is not *slot*'s to reset now.
    """
    # circular import: chat_handlers imports this module at module level.
    from kiro_crew.dashboard.chat_handlers import _cancel_target
    from kiro_crew.messaging.link import canonical_key

    session_key = effective_session_key(slot)
    provider = state.sessions.get_provider(session_key)
    has_active_turn = getattr(provider, "has_active_turn", None)
    if callable(has_active_turn) and has_active_turn():
        return True
    target = canonical_key(session_key)
    return any(
        other is not slot and other.running and canonical_key(_cancel_target(other)) == target
        for other in list(state._slots.values())
    )


def legacy_effort_base(model: str | None, *, pair_id_backend: bool) -> str:
    """The base id a pin folds to when an effort level is committed, or ``""``.

    On a backend that spells effort into the model id (codex:
    ``gpt-6-astra[max]``), a pin still carrying a ``[level]`` suffix would keep
    claiming the old level after a new one is committed, so the commit folds the
    pin to its base id. Any other backend, or a pin with no suffix, folds nothing.

    The effort dropdown route and :func:`apply_pending_model_pick` both commit a
    level, and both ask this, so a change to the fold reaches both paths.
    """
    if not pair_id_backend:
        return ""
    base, level = model_registry.split_effort_suffix(model or "")
    return base if level else ""


def effort_commit_needs_reset(provider: object) -> bool:
    """Whether committing a new effort level must reset the live session.

    A live ACP session whose model takes no effort level keeps the level as a
    stored value only: it applies once the user picks a capable model, and a
    reset would restart the session for nothing. Any other live session, or no
    live session, needs the reset (or the live push the dropdown tries first)
    for the level to take.

    Shared by the effort dropdown route and :func:`apply_pending_model_pick`
    for the same reason as :func:`legacy_effort_base`.
    """
    # Duck-typed so this module stays above the agent-SDK boundary. Every
    # LLMProvider has ``supports_effort``, and the base one returns a placeholder
    # False, so only a provider that sets ``reports_effort_support`` (the ACP
    # provider) can skip the reset. Read off the class, as isinstance would, so
    # any other provider, an unspecced test double or no session keeps it.
    if getattr(provider.__class__, "reports_effort_support", False) is not True:
        return True
    return provider.supports_effort() is not False  # type: ignore[attr-defined]


def apply_pending_model_pick(state: "DashboardState", slot: "_ChatSlot") -> bool:
    """Commit *slot*'s pending ``session_set_model`` pick, if it is still allowed.

    Called at the start of the slot's turn, before a session is acquired, after
    :func:`prewarm_enabled_check` so the fence re-read below is a cache hit.
    SYNCHRONOUS on purpose: the gate and the writes to ``slot.model`` and
    ``slot.reasoning_effort`` run with no suspension between them, so nothing
    can link or mirror the target after it was authorized and before either
    changed. A pick the gate now refuses is dropped and audited, and the turn
    runs on the model and effort it already had. So is each half of a pick the
    user has overtaken with a newer choice from that half's own control.

    The ownership fence only tightens: a caller fenced at call time stays
    fenced, and one that was not is re-checked now, since it may have become a
    fenced crew member while the pick waited.

    Returns True when the live session must be reset before this turn uses it:
    ``slot.model`` changed (or a fallback serves the session), a codex pin's
    legacy ``[level]`` suffix was folded away, or ``slot.reasoning_effort``
    changed.
    """
    pick = slot._pending_model_pick
    if pick is None:
        return False
    if slot._model_pick_lock.locked():
        # A model-picker switch is mid-transaction: its pick generation is
        # provisional and may still roll back. Leave the pick pending and run
        # this turn on the current model; the next turn start decides against
        # the settled generation (a committed picker choice then supersedes the
        # pick, a rolled-back one lets it apply). Nothing is decided yet, so
        # nothing is audited here.
        logger.info("session-control set_model: pick on %s deferred, picker in flight", slot.key)
        return False
    if _another_alias_is_mid_turn(state, slot):
        # Another slot drives the same session and has a turn in flight (or is
        # cold-starting it). A commit now would reset nothing the turn can see:
        # get_or_create attaches to that session on its old model. Leave the
        # pick pending; a later turn start applies it once the session is idle.
        logger.info("session-control set_model: pick on %s deferred, alias mid-turn", slot.key)
        return False
    slot._pending_model_pick = None
    # Each half of the pick yields separately to a newer choice the user made
    # from its own control, so a dropdown effort pick does not throw away the
    # caller's model, nor a picker model pick the caller's effort.
    model = pick.model
    effort = pick.reasoning_effort
    if model is not None and slot._model_pick_gen != pick.pick_gen:
        model = None
        _audit(
            caller_session_key=pick.caller_session_key,
            operation="set_model",
            slot_key=slot.key,
            outcome="denied",
            detail={"code": "superseded_by_newer_pick", "stage": "turn_start"},
        )
    # On a pair-id backend (codex) the model picker IS the effort control: a
    # user picking ``gpt-6-astra[max]`` bumps only the model generation, so a
    # newer model pick supersedes the effort half as well.
    if effort is not None and (
        slot._effort_pick_gen != pick.effort_gen
        or (pick.pair_id_backend and slot._model_pick_gen != pick.pick_gen)
    ):
        effort = None
        _audit(
            caller_session_key=pick.caller_session_key,
            operation="set_model",
            slot_key=slot.key,
            outcome="denied",
            detail={"code": "superseded_by_newer_effort", "stage": "turn_start"},
        )
    if model is None and effort is None:
        return False
    if _caller_tab_id(state, pick.caller_session_key) != pick.caller_tab_id:
        _audit(
            caller_session_key=pick.caller_session_key,
            operation="set_model",
            slot_key=slot.key,
            outcome="denied",
            detail={"code": "caller_replaced", "stage": "turn_start"},
        )
        return False
    caller_key = caller_slot_key(state, pick.caller_session_key)
    fenced = pick.caller_fenced or (
        bool(caller_key) and _caller_is_ownership_fenced(state, caller_key)
    )
    # The enabled check runs here too: an operator who disables session control
    # after a pick was queued must stop it from applying. `_run_chat` calls
    # `prewarm_enabled_check` immediately before this, with nothing suspending
    # in between, so the check reads the warmed config and does no file IO.
    try:
        live = authorize_target(
            state,
            caller_session_key=pick.caller_session_key,
            target=slot.key,
            operation="set_model",
            precomputed_ownership_fenced=fenced,
        )
    except SessionControlError as exc:
        _audit(
            caller_session_key=pick.caller_session_key,
            operation="set_model",
            slot_key=slot.key,
            outcome="denied",
            detail={"code": exc.code, "stage": "turn_start"},
        )
        return False
    if live is not slot:
        _audit(
            caller_session_key=pick.caller_session_key,
            operation="set_model",
            slot_key=slot.key,
            outcome="denied",
            detail={"code": "target_replaced", "stage": "turn_start"},
        )
        return False
    changed = False
    if model is not None:
        # A fallback serving the session means the wire model differs from the
        # pin, so an equal pin still needs the session reset, as the picker
        # treats it.
        changed = (
            (slot.model or "") != model
            or bool(slot._active_fallback_model)
            or bool(slot._refusal_fallback_primary)
        )
        slot.model = model
        # A concrete pick answers the routing question, as it does from the picker.
        slot.jev_route = False
        # Recorded as an explicit pick so the model-fallback restore never undoes it.
        slot._model_pick_gen += 1
    if effort is not None:
        base = legacy_effort_base(slot.model, pair_id_backend=pick.pair_id_backend)
        if base:
            slot.model = base
            changed = True
        if (slot.reasoning_effort or "") != effort and effort_commit_needs_reset(
            state.sessions.get_provider(effective_session_key(slot))
        ):
            # A live session keeps the level it started with; the reset the
            # caller arms on True makes the cold start read this one. A model
            # that takes no level keeps it stored only, as the dropdown does.
            changed = True
        slot.reasoning_effort = effort
    _audit(
        caller_session_key=pick.caller_session_key,
        operation="set_model",
        slot_key=slot.key,
        outcome="allowed",
        detail={**_pick_audit_fields(model, effort), "stage": "turn_start"},
    )
    return changed


#: Lowercased spellings of the picker's Jev entry that ``session_set_model``
#: refuses alongside the exact sentinel: its display label and any case of the
#: sentinel id. Owner-only, like the sentinel itself.
_JEV_ROUTE_SPELLINGS = frozenset({"auto (jev)", "auto:jev"})


def _target_busy_error() -> SessionControlError:
    """The refusal ``set_model_target`` gives a target with work in flight."""
    return SessionControlError(
        "session busy, model not changed: it has a turn or sub-agents in flight. "
        "Stop it with session_stop and retry once it is idle.",
        code="target_busy",
        status=409,
    )


def _reload_busy_error() -> SessionControlError:
    """The refusal ``reload_target`` gives a target with work in flight or queued."""
    return SessionControlError(
        "session busy, not reloaded: it has a turn, queued messages or sub-agents in "
        "flight. Wait until it is idle and retry.",
        code="target_busy",
        status=409,
    )


def _reload_route_refusal(status: int, body: dict[str, Any]) -> SessionControlError:
    """Map a ``reload_slot_session`` refusal to what ``session_reload`` reports.

    The two busy codes collapse into ``target_busy`` so a caller sees one code for
    "not now"; ``slot_not_found`` means the slot was replaced while the request
    queued on a lock, which for this verb is a replaced target.
    """
    code = str(body.get("code") or "")
    if code in ("turn_in_flight", "slot_subagents_running"):
        return _reload_busy_error()
    if code == "slot_not_found":
        return SessionControlError(
            "the target session was replaced; not reloaded",
            code="target_replaced",
            status=409,
        )
    return SessionControlError(
        str(body.get("error") or "reload refused"), code=code or "reload_failed", status=status
    )


async def reload_target(
    state: "DashboardState",
    *,
    caller_session_key: str,
    target: str,
    caller_fenced: bool | None = None,
) -> dict[str, Any]:
    """Relaunch *target*'s agent process, as the tab menu's Reload session does.

    The teardown is ``chat_handlers.reload_slot_session``, the same code the
    dashboard route runs, so the lock order, the rebind and replacement
    re-checks and the ``skip_if_busy`` reset are shared rather than copied. The
    transcript is not rewritten: the session is reset and one reload notice is
    appended, naming the calling session.

    Narrower than the other verbs in three ways:

    * Only a session the caller CREATED, for every caller class. The dashboard
      owner's own sessions are reachable by ``session_stop`` and the rest, but
      a reload relaunches a process the person may be in the middle of using,
      so this verb stays on sessions the caller dispatched itself.
    * Never the caller itself (``authorize_target``'s default self refusal). A
      caller is mid-turn by definition, and the teardown refuses a session with
      a turn in flight, so a self-reload could only ever fail.
    * Only an IDLE target: ``_switch_target_busy`` (which also sees a turn that
      is still cold-starting), a non-empty queue, or attached sub-agents refuse
      it with ``target_busy`` before anything is torn down, and the same probe
      runs again inside the teardown's locks.

    ``caller_fenced`` has the meaning :func:`stop_target` documents.
    """
    # Deferred for the same import cycle `stop_target` documents.
    from kiro_crew.dashboard.chat_handlers import (
        _SESSION_RELOAD_NOTICE,
        _subagents_attached_response,
        _switch_target_busy,
        reload_slot_session,
    )

    # Same prewarm ordering as `stop_target`.
    try:
        await asyncio.to_thread(sel)
    except Exception:  # noqa: BLE001 - a prewarm failure must not fail the reload
        logger.warning("session-control SEL prewarm failed", exc_info=True)
    await prewarm_enabled_check()
    caller_key = caller_slot_key(state, caller_session_key)
    if caller_fenced is None:
        caller_fenced = bool(caller_key) and _caller_is_ownership_fenced(state, caller_key)

    def _gate(target_name: str, *, first: bool = False) -> "_ChatSlot":
        # Synchronous, so it can run inside the teardown's locks with no
        # suspension between the decision and the act it authorizes. Only the
        # first pass reads the enabled switch, as in `close_target`'s re-check.
        found = authorize_target(
            state,
            caller_session_key=caller_session_key,
            target=target_name,
            operation="reload",
            skip_enabled_check=not first,
            precomputed_ownership_fenced=caller_fenced,
        )
        with _audit_denials(
            caller_session_key=caller_session_key, operation="reload", slot_key=found.key
        ):
            if not caller_key or _created_by_other(found, caller_key):
                raise SessionControlError(
                    "a session can only reload sessions it created itself",
                    code="not_creator",
                    status=403,
                )
            if is_relay_archive(found):
                raise SessionControlError(RELAY_ARCHIVE_ERROR, code=RELAY_ARCHIVE_CODE, status=409)
        return found

    def _busy(target_slot: "_ChatSlot", session_key: str) -> bool:
        provider = state.sessions.get_provider(session_key)
        return bool(target_slot._queue) or _switch_target_busy(
            state, target_slot, session_key, provider
        )

    slot = _gate(target, first=True)
    slot_key = slot.key

    with _audit_denials(
        caller_session_key=caller_session_key, operation="reload", slot_key=slot_key
    ):
        # Refused up front, before any lock is taken, so a busy target costs the
        # caller one probe and nothing queues behind the target's own switches.
        session_key = effective_session_key(slot)
        if _busy(slot, session_key):
            raise _reload_busy_error()
        if await _subagents_attached_response(state, slot, session_key, "reload") is not None:
            raise _reload_busy_error()

    changed_after_reset: list[SessionControlError] = []

    def _still_ours(after_reset: bool) -> bool:
        # Re-runs the whole gate after every await inside the teardown: a
        # replaced slot, a new channel link or mirror, a moved workspace or a
        # changed creator raises its own refusal out of the lock stack.
        # ``after_reset`` comes from reload_slot_session itself, so the phase
        # is never inferred from which callback ran last.
        if state._slots.get(slot_key) is not slot:
            if after_reset:
                changed_after_reset.append(
                    SessionControlError(
                        "the target session was replaced", code="slot_replaced", status=409
                    )
                )
            return False
        try:
            return _gate(slot_key) is slot
        except SessionControlError as exc:
            if not after_reset:
                raise
            # The reset already ran. Letting the refusal escape would tell the
            # caller "nothing happened" about a process that is gone, and would
            # skip the audit below; record it and answer as a changed target.
            changed_after_reset.append(exc)
            return False

    notice = f"{_SESSION_RELOAD_NOTICE} Requested by session `{caller_key}`."
    try:
        response = await reload_slot_session(
            state,
            slot,
            slot_key,
            still_ours=_still_ours,
            denied=lambda _session_key: None,
            busy=lambda session_key: _busy(slot, session_key),
            notice=notice,
        )
    except SessionControlError:
        raise
    except Exception as exc:
        # The shared teardown absorbs a raise after the session pop (the
        # reload happened, degraded) and re-raises only one from before it,
        # when the old process is still the registered one. Report that as a
        # failed reload and audit it, rather than escaping as a bare 500.
        logger.exception("session_reload of %s failed before the reset", slot_key)
        _audit(
            caller_session_key=caller_session_key,
            operation="reload",
            slot_key=slot_key,
            outcome="denied",
            detail={"code": "reload_failed", "error": type(exc).__name__},
        )
        raise SessionControlError(
            "the reload failed before the target's agent process was reset; "
            "nothing was torn down",
            code="reload_failed",
            status=500,
        ) from exc
    if changed_after_reset:
        _audit(
            caller_session_key=caller_session_key,
            operation="reload",
            slot_key=slot_key,
            outcome="denied",
            detail={
                "code": "target_changed_during_reload",
                "gate_code": changed_after_reset[0].code,
            },
        )
        raise SessionControlError(
            "the target's agent process was reset, but the session changed during "
            f"the reload ({changed_after_reset[0].code}); no reload notice was added",
            code="target_changed_during_reload",
            status=409,
        )
    if response.status != 200:
        body = json.loads(response.text or "{}")
        error = _reload_route_refusal(response.status, body)
        _audit(
            caller_session_key=caller_session_key,
            operation="reload",
            slot_key=slot_key,
            outcome="denied",
            detail={"code": error.code},
        )
        raise error

    body = json.loads(response.text or "{}")
    warning = body.get("warning")
    detail: dict[str, Any] = {"reloaded_by": caller_key}
    if warning:
        detail["warning"] = warning
    _audit(
        caller_session_key=caller_session_key,
        operation="reload",
        slot_key=slot_key,
        outcome="allowed",
        detail=detail,
    )
    out: dict[str, Any] = {"ok": True, "target": slot_key}
    if warning:
        out["warning"] = warning
    return out


async def close_target(
    state: "DashboardState",
    *,
    caller_session_key: str,
    target: str,
    caller_fenced: bool | None = None,
) -> dict[str, Any]:
    """Close *target*, the same archival the tab ✕ performs.

    Non-destructive: the conversation is saved to history and can be reopened
    later — closing dismisses the LIVE tab, it does not delete the transcript.
    An in-flight turn is cancelled first (its work is discarded), so this is a
    strictly heavier act than :func:`stop_target`; the description tells the
    caller to read the session before closing it.

    Reuses the dashboard's own close path (:func:`chat_handlers.close_slot`), so
    a controlled close and a human ✕ share the identical nudge-retirement and
    app-notification ordering that keeps a dismissed tab from being resurrected.
    Its four failure modes surface as their own ``SessionControlError`` codes
    rather than a generic 500, so a caller can tell "the app refused the
    dismissal" from "history could not be saved".
    """
    # Same prewarm ordering as `stop_target`, for the same reasons: the SEL write
    # inside `authorize_target`'s deny path must be a cache hit, and the config
    # warm must be the LAST suspension before the synchronous gate.
    try:
        await asyncio.to_thread(sel)
    except Exception:  # noqa: BLE001 - a prewarm failure must not fail the close
        logger.warning("session-control SEL prewarm failed", exc_info=True)
    await prewarm_enabled_check()

    caller_key = caller_slot_key(state, caller_session_key)
    # Resolve the ownership-fence verdict ONCE, up front, while the config cache
    # is warm from ``prewarm_enabled_check`` above and we are not yet inside
    # ``close_slot``'s no-suspension window. BOTH the initial gate and the
    # synchronous re-check reuse it via ``precomputed_ownership_fenced`` rather
    # than recomputing — the fence can read config on a cache miss
    # (``_member_caller`` → ``_store_is_member_owned`` → ``KiroCrewConfig.load()``),
    # which is exactly the blocking-IO-on-the-event-loop the close critical
    # section must not do. A verdict the HTTP gate already carried (a caller it
    # admitted as a crew member) is honoured as-is; otherwise it is computed here.
    # An empty ``caller_key`` (unidentifiable caller) makes it ``False`` and the
    # gate below still raises ``caller_unidentified`` before the fence is consulted.
    if caller_fenced is None:
        caller_fenced = bool(caller_key) and _caller_is_ownership_fenced(state, caller_key)
    # The manager-line switch, resolved HERE for the reason the fence verdict above is:
    # the synchronous re-check runs inside ``close_slot``'s no-suspension window, and
    # reading it there would load config on the loop. One read, both gates, so the
    # re-check cannot answer a different question from the one that admitted the close.
    #
    # In a THREAD, like the prewarm above it, because the read validates the config file
    # on a cache miss and doing that inline blocks the loop for every other session.
    chain_enabled = note_chain_switch(await asyncio.to_thread(manager_chain_enabled))

    slot = authorize_target(
        state,
        caller_session_key=caller_session_key,
        target=target,
        operation="close",
        precomputed_ownership_fenced=caller_fenced,
        precomputed_manager_chain=chain_enabled,
        allow_self=True,
    )
    slot_key = slot.key
    # Deferred for the same import cycle `stop_target` documents.
    from kiro_crew.dashboard.chat_handlers import SlotCloseError, close_slot

    def _reassert_closeable() -> None:
        # Re-run the SAME target gate at close_slot's point of no return —
        # SYNCHRONOUSLY, so there is NO event-loop suspension between it and the
        # pop and nothing can change between the final authorization and the
        # archival. The initial gate above ran before close_slot's awaits
        # (nudge-loop retirement takes the AutoNudge lock; the app hook awaits
        # external work), and a target that was unmirrored/unlinked then can gain
        # a channel mirror or link in that window — archiving a now-channel-backed
        # session the caller was never allowed to reach.
        #
        # `skip_enabled_check=True` omits the ONE part of authorize_target that
        # can touch the disk (`session_control_enabled()`'s config read): the
        # feature was already confirmed enabled above, whether it was switched off
        # mid-close is not a containment boundary, and skipping it is what lets
        # this run with no await — an async prewarm-then-check would put an await
        # back before the pop and reopen the very window this closes. Every
        # containment and identity refusal still runs.
        #
        # `precomputed_ownership_fenced=caller_fenced` closes the SECOND disk
        # touch: the ownership fence's own config read (member-store lookup),
        # resolved once above and reused here so this callback never loads config
        # on the loop.
        try:
            live = authorize_target(
                state,
                caller_session_key=caller_session_key,
                target=slot_key,
                operation="close",
                skip_enabled_check=True,
                precomputed_ownership_fenced=caller_fenced,
                precomputed_manager_chain=chain_enabled,
                allow_self=True,
            )
        except SessionControlError as exc:
            # A stale-authorization refusal (mirrored/linked/workspace/caller-gone)
            # becomes a SlotCloseError carrying that same status, so it round-trips
            # to the caller as the specific 403 rather than a generic close failure.
            raise SlotCloseError(exc.message, code=exc.code, status=exc.status) from exc
        if live is not slot:
            # The key was re-minted onto a DIFFERENT session while close_slot
            # awaited (a concurrent close+reopen). authorize_target resolves by
            # key, so it would authorize the replacement — but close_slot pops
            # `name` and tears down / saves the ORIGINAL slot it holds. Comparing
            # identity (not mere presence) is the same guard `create_session` uses
            # for its re-minted-key window; abort so the replacement lives.
            raise SlotCloseError(
                "the target session was replaced during the close",
                code="target_replaced",
                status=409,
            )

    try:
        await close_slot(state, slot, slot_key, pre_pop_check=_reassert_closeable)
    except SlotCloseError as exc:
        # The close path already rolled back every partial step and logged the
        # cause; re-raise it as the surface's own error so the caller sees the
        # specific reason (write-in-flight/nudge/app/history) rather than a bare
        # failure. Audited as a denied operation so the trail shows the close was
        # attempted and did not take.
        _audit(
            caller_session_key=caller_session_key,
            operation="close",
            slot_key=slot_key,
            outcome="denied",
            detail={"code": exc.code},
        )
        raise SessionControlError(exc.message, status=exc.status, code=exc.code) from exc
    _audit(
        caller_session_key=caller_session_key,
        operation="close",
        slot_key=slot_key,
        outcome="allowed",
    )
    return {"ok": True, "target": slot_key}


#: Cap on one delivered message. Aliased to ``validation.MAX_LONG_STRING`` rather
#: than restated as its own number: a seed prompt is that shape, the MCP schema
#: layer already rejects on that constant, and two spellings of one 50k limit
#: would drift apart the first time either moved.
def _scan_archived_candidates(
    log: Any, key_candidate: str, wanted_title: str
) -> list[tuple[str, str, dict[str, Any]]]:
    """The DISK half of archived-target resolution: runs in a worker thread.

    Returns every on-disk dashboard session that *key_candidate* names or whose
    title equals *wanted_title*, as ``(slot_key, history_key, metadata)``. It
    reads the history store only -- metadata lines and the session catalog --
    and never touches the live slot table, which belongs to the event loop and
    is consulted by the caller after this returns.
    """
    found: dict[str, tuple[str, dict[str, Any]]] = {}

    def _fold(key: str) -> str:
        # ``_normalize_slot_key`` strips ONE transport prefix per call, so a
        # doubled ``dashboard:dashboard:member-x`` folds to ``dashboard_member-x``:
        # a key the ``member-``/``cron-`` prefix guards do not match, whose
        # history key is nevertheless the real ``dashboard:member-x`` transcript
        # the resume core would publish. Fold to the fixed point so the guards
        # test the key that is actually revived.
        for _ in range(8):
            folded = _normalize_slot_key(key)
            if folded == key:
                return key
            key = folded
        return key

    def _consider(slot_key: str) -> None:
        if not slot_key or slot_key in found:
            return
        history_key = _history_key_for(slot_key)
        meta, readable = log.get_metadata_status(history_key)
        if readable and meta:
            found[slot_key] = (history_key, dict(meta))

    # Every candidate is taken from the CATALOG's spelling of the filename stem,
    # never echoed from the caller: on a case-insensitive filesystem (APFS, NTFS)
    # ``Chat-7`` opens ``dashboard_chat-7.jsonl`` just as well, so a metadata
    # probe on the caller's spelling would succeed while the live-slot dedup
    # (exact ``_slots`` lookup) and the resume core's own live check (exact
    # session-key compare) both miss the ``chat-7`` that is already open -- and
    # a second slot would be published over the same transcript file. Resolving
    # through the catalog keys ``found`` by the stem the store reports, so the
    # key handed back is the one the live table and the resume compare against.
    rows = [
        row for row in log.list_sessions() if str(row.get("key") or "").startswith("dashboard_")
    ]
    if key_candidate:
        wanted_stem = _fold(key_candidate)
        if not wanted_stem.startswith("dashboard_"):
            wanted_stem = f"dashboard_{wanted_stem}"
        for row in rows:
            stem = str(row.get("key") or "")
            if stem.casefold() == wanted_stem.casefold():
                _consider(_fold(stem))
    if wanted_title:
        for row in rows:
            if (str(row.get("title") or "")).strip().casefold() == wanted_title:
                _consider(_fold(str(row.get("key") or "")))
    return [(k, hk, meta) for k, (hk, meta) in found.items()]


async def _resolve_archived_target(
    state: "DashboardState",
    log: Any,
    target: str,
    deny: "Callable[..., SessionControlError]",
    live_refusal: "Callable[[str], Awaitable[SessionControlError]]",
) -> tuple[str, str, dict[str, Any]]:
    """Find the ARCHIVED dashboard session *target* names.

    Accepts what a caller actually holds -- a slot key (``chat-7-...``), the
    ``dashboard:<slot>`` session key, the ``dashboard_<slot>`` filename stem
    ``list_sessions`` reports, or an exact case-insensitive title -- and returns
    ``(slot_key, history_key, metadata)`` for exactly one on-disk session that
    has no live slot. Mirrors :func:`_resolve_slot`'s doctrine: every form is
    resolved before anything is returned, and two DIFFERENT sessions matching
    across forms is refused as ambiguous rather than silently preferring one.

    The disk scan (metadata lines, the session catalog) runs off the loop in
    :func:`_scan_archived_candidates`; the live-slot table is read here, on the
    loop, both before (a live match is refused with its key) and after the scan
    (a candidate that went live during it is dropped rather than revived twice).

    A target that IS live is handed to *live_refusal* with its live key, and
    that builder decides what the caller learns: :func:`revive_session` authorizes
    the live slot as any live target first, so a protected session answers with
    that refusal and reveals neither its existence nor its key, while one the
    caller may reach is named so the caller can just address it.
    """
    # Circular at runtime: ``members`` imports this module.
    from kiro_crew import members as members_mod

    try:
        live = _resolve_slot(state, target)
    except SessionControlError as exc:
        raise deny(exc.message, exc.code, status=exc.status) from exc
    if live is not None:
        raise await live_refusal(live.key)

    # The key forms fold to one slot key; a bare title never does (it is not a
    # dashboard prefix and ``_normalize_slot_key`` would mangle spaces), so only
    # a target that looks like a key is tried as one.
    stripped = target.strip()
    key_candidate = stripped if stripped and not any(ch.isspace() for ch in stripped) else ""
    candidates = await asyncio.to_thread(
        _scan_archived_candidates, log, key_candidate, stripped.casefold()
    )
    found = [(k, hk, meta) for k, hk, meta in candidates if state.get_slot(k) is None]
    if candidates and not found:
        # Every on-disk match went live during the scan (a human resume click
        # racing this call): answer as the pre-scan probe would have -- through
        # the same authorize-then-name builder -- rather than "not found" for a
        # session the caller can see.
        raise await live_refusal(candidates[0][0])

    if len(found) > 1:
        raise deny(
            f"{len(found)} archived sessions match {target!r} (as a session key, "
            "transcript name, or title) -- address it by its session key instead",
            "ambiguous_target",
            status=409,
        )
    if not found:
        raise deny(f"no archived session matches {target!r}", "target_not_found", status=404)
    slot_key, history_key, meta = found[0]
    if slot_key.casefold().startswith(members_mod.DM_SLOT_KEY_PREFIX):
        # A crew member's DM thread is opened only through its own roster route,
        # which re-checks the member binding; session control has no business
        # resurrecting one under a caller's name.
        raise deny("member threads are not addressable", "member_thread_target")
    return slot_key, history_key, meta


def _archived_session_is_mirrored(state: "DashboardState", history_key: str) -> bool:
    """:func:`_has_channel_mirror` for a session that has no live slot yet.

    Fail-closed like the refusal paths: a store that cannot answer counts as
    mirrored, so an unreadable link never opens the boundary.
    """
    probed = _probe_channel_mirror_for_key(state, history_key)
    return True if probed is None else bool(probed)


def _archived_session_is_channel_linked(state: "DashboardState", history_key: str) -> bool:
    """Whether the GATEWAY's session store records a channel link for an archived
    session -- the inbound origin (``get_origin_link``, the conversation the
    channel dispatcher recorded as this session's own) or the Slack thread binding
    (``get_slack_link``).

    The metadata line's ``linked_session_key`` / ``channel_origin`` say the same
    thing, but that line is a file an agent's tools can edit; the session store is
    gateway-owned, so this is the corroborating read the channel-link boundary
    rests on for a revive. Fail-closed: a store that cannot answer counts as
    linked. A state with no session store answers "not linked", the same posture
    as the mirror probe, since there is no record to consult.
    """
    sessions = getattr(state, "sessions", None)
    if sessions is None:
        return False
    try:
        get_origin = getattr(sessions, "get_origin_link", None)
        if get_origin is not None and get_origin(history_key):
            return True
        get_slack = getattr(sessions, "get_slack_link", None)
        if get_slack is not None:
            thread_ts, _channel = get_slack(history_key)
            if thread_ts:
                return True
    except Exception:
        logger.debug("channel-link probe failed", exc_info=True)
        return True
    return False


def _archived_link_and_mirror(state: "DashboardState", history_key: str) -> tuple[bool, bool]:
    """``(channel_linked, mirrored)`` for an archived session -- the WORKER-THREAD half.

    Both probes read the gateway's session store through its guarded getters,
    and the store's off-loop writer holds that same lock across a temp-file write
    and rename, so a probe on the event loop can stall every session behind one
    disk write. :func:`revive_session` therefore runs the pair through
    ``asyncio.to_thread``, the way :func:`_scan_archived_candidates` already runs
    the catalog scan. Each half keeps its own fail-closed reading.
    """
    return (
        _archived_session_is_channel_linked(state, history_key),
        _archived_session_is_mirrored(state, history_key),
    )


def _revived_slot_refusal(slot: "_ChatSlot", caller_slot: "_ChatSlot") -> tuple[str, str] | None:
    """The target-side refusal, if any, that the HYDRATED slot now earns.

    :func:`revive_session` answers the target-side boundaries from the archived
    metadata line before the resume, but the resume core re-reads that line after
    its threaded transcript read and hydrates the slot from the fresh copy. The
    same fields are therefore read back off the live slot -- the reads
    :func:`authorize_target` makes on a live target, same order, same wording and
    codes -- so a line rewritten inside the resume's window cannot publish a
    slot the pre-resume answer would have refused. Returns ``(reason, code)`` or
    ``None`` when every boundary still holds.
    """
    if (refusal := _live_target_refusal(slot)) is not None:
        return refusal
    if (getattr(slot, "workspace", "default") or "default") != (
        getattr(caller_slot, "workspace", "default") or "default"
    ):
        return ("target session belongs to a different workspace", "workspace_mismatch")
    return None


async def revive_session(
    state: "DashboardState",
    *,
    caller_session_key: str,
    target: str,
    folder_id: str = "",
    caller_fenced: bool | None = None,
) -> dict[str, Any]:
    """Bring the ARCHIVED session *target* back into the live sidebar.

    The mirror of :func:`close_target`: a close archives a tab to history; this
    pulls one history session back up into a live slot, the same thing the
    History tab's resume click does. It reuses the dashboard's own resume core
    (:func:`chat_handlers.resume_slot_from_history`), so a controlled revive
    and a human click share one materialisation, one set of member-pin and
    delete/recreate barriers and one set of refusal codes.

    Authorization is :func:`authorize_target`'s, applied to the session's
    PERSISTED metadata because there is no live slot yet to read: the same
    caller-side refusals (via the shared helpers), then the target-side ones --
    unattended, ephemeral, app-scoped, channel-linked, a different workspace --
    read from the metadata line, and the same ownership fence: an ownership-
    fenced caller (an agent-created session, a crew member, a scheduled run)
    may revive only a session whose ``created_by`` is itself AND whose crew-log
    lineage names it as parent -- the metadata line is agent-editable, the
    lineage record is not, so the fence fails closed when the record cannot be
    read.

    ``folder_id`` files the revived slot the way ``create_session`` does, after
    the revive has landed; a folder that does not exist refuses the whole call
    BEFORE anything is revived, so a refusal leaves history untouched. The
    revived slot keeps its own creator: reviving is not creating, so ownership
    is never transferred to the caller.
    """
    # Circular at runtime: ``chat_handlers`` imports this module.
    from kiro_crew.dashboard.chat_handlers import ResumeRefusal, resume_slot_from_history

    try:
        await asyncio.to_thread(sel)
    except Exception:  # noqa: BLE001 - a prewarm failure must not fail the revive
        logger.warning("session-control SEL prewarm failed", exc_info=True)
    await prewarm_enabled_check()

    deny = _deny_factory(caller_session_key=caller_session_key, operation="revive", target=target)

    caller_key = refuse_caller_identity(
        state, caller_session_key=caller_session_key, deny=deny, skip_enabled_check=False
    )

    log = state.conversation_log
    if log is None:
        raise deny("session history is unavailable", "history_unavailable")

    # The whole caller side BEFORE the target is resolved: a caller refused for
    # its own identity or containment must learn nothing from the attempt, and
    # resolving first would make the refusal an existence oracle (404 for an
    # absent archived session, 403 for a present one). Off the loop because the
    # caller-side set ends in the session-store mirror probe, whose guarded
    # getter shares the lock the store's writer holds across its disk write.
    caller_slot = await asyncio.to_thread(
        refuse_caller_surface, state, caller_key=caller_key, deny=deny
    )
    # The fence verdict, resolved ONCE and before any target is named: the
    # live-target builder below needs it on the loop, and evaluating it inline
    # there would be a config read (disk on a cache miss) in a synchronous span.
    ownership_fenced = (
        caller_fenced
        if caller_fenced is not None
        else await asyncio.to_thread(_caller_is_ownership_fenced, state, caller_key)
    )

    async def _live_refusal(live_key: str) -> SessionControlError:
        # A target that turns out to be LIVE (before the scan, during it, or
        # by the time the resume ran) is a live slot hydrated from a line this
        # call never checked, so it is authorized exactly as a live target is
        # before anything about it is said: a slot the caller may not reach
        # answers with THAT refusal and reveals neither its existence nor its
        # key; one it may reach is named so the caller can address it.
        #
        # Split the way the resume hook and ``final_check`` are split. The
        # checks answerable from live slot fields run ON the loop, so nothing
        # can mutate the slot between the read and the answer; the two
        # session-store probes (the caller's owner-DM exemption and the target's
        # outbound mirror) go to a worker, because their guarded getters share
        # the lock the store's writer holds across its disk write; and the
        # store-free set runs once more on the loop after those awaits, right
        # before the key is named -- a workspace commit or a link rebind landing
        # inside the worker hop would otherwise be answered from stale fields.
        def _store_free() -> "_ChatSlot":
            live = state.get_slot(live_key)
            if live is None:
                raise deny(f"no open session matches {target!r}", "target_not_found", status=404)
            if live.key == caller_key:
                raise deny("a session cannot control itself", "self_target")
            live_caller = _check_caller_slot_fields(state, caller_key, deny)
            if (refusal := _revived_slot_refusal(live, live_caller)) is not None:
                raise deny(refusal[0], refusal[1])
            if ownership_fenced and _created_by_other(live, caller_key):
                raise deny(
                    _not_creator_reason(state, caller_key, live_caller, caller_fenced),
                    "not_creator",
                )
            return live

        live = _store_free()
        await asyncio.to_thread(refuse_caller_surface, state, caller_key=caller_key, deny=deny)
        if await asyncio.to_thread(_has_channel_mirror, state, live):
            raise deny("sessions mirrored to a channel are not addressable", "mirrored_target")
        _store_free()
        return deny(
            f"{target!r} is already open as `{live_key}`; address it directly",
            "target_already_live",
            status=409,
        )

    slot_key, history_key, meta = await _resolve_archived_target(
        state, log, target, deny, _live_refusal
    )

    # Target-side containment, read from the metadata line the resume would
    # rehydrate the slot from -- the same fields ``authorize_target`` reads off
    # a live slot, in the same order, with the same codes.
    if slot_key.casefold().startswith(UNATTENDED_SLOT_PREFIXES):
        raise deny("unattended sessions (scheduled runs) cannot be controlled", "unattended_target")
    if str(meta.get("memory_mode") or "persistent") != "persistent":
        raise deny("incognito and temporary sessions are not addressable", "ephemeral_target")
    if meta.get("app"):
        raise deny("app-scoped sessions are not addressable", "app_scoped_target")
    # The session-store probes (inbound link, Slack thread binding, outbound
    # mirror) go through the store's guarded getters, whose lock an off-loop
    # writer holds across its disk write, so they run in a worker thread like the
    # catalog scan does rather than on the loop.
    linked, mirrored = await asyncio.to_thread(_archived_link_and_mirror, state, history_key)
    if meta.get("linked_session_key") or meta.get("channel_origin") or linked:
        # Both readings: the metadata line AND the gateway-owned session store,
        # so a channel link scrubbed from the editable line is still seen.
        raise deny("channel-linked sessions are not addressable", "linked_session_target")
    if mirrored:
        # The outbound-mirror boundary ``_has_channel_mirror`` holds for a live
        # slot, read on the history key because the link lives in the session
        # store, not in the transcript: an archived dashboard session can still
        # be republishing to a channel the moment it is reopened. Unknown reads
        # as mirrored, the refusal paths' fail-closed direction.
        raise deny("sessions mirrored to a channel are not addressable", "mirrored_target")

    target_workspace = str(meta.get("workspace") or "default")
    if target_workspace != (getattr(caller_slot, "workspace", "default") or "default"):
        raise deny("target session belongs to a different workspace", "workspace_mismatch")
    if ownership_fenced:
        if str(meta.get("created_by") or "") != caller_key:
            raise deny(
                _not_creator_reason(state, caller_key, caller_slot, caller_fenced), "not_creator"
            )
        # The metadata line is a file an agent's file tools can edit
        # (``chat_persistence`` says so where it restores ``created_by``), so for a
        # FENCED caller the claim above is only a claim: a caller that rewrote
        # ``created_by`` in an archived transcript would otherwise revive a session
        # it never made and then hold it. The ownership is therefore corroborated
        # against the one record those tools cannot reach -- the crew log's
        # session-tree lineage, written by this gateway at the child's first turn
        # from the in-process ``_lineage_minted`` witness and never from disk
        # metadata. Fail closed when that record is not readable (crew log off,
        # projection unseeded or incomplete) or names a different parent: an
        # unverifiable ownership is refused, not assumed. A caller that is NOT
        # fenced (the person's own tab) is not gated on ownership at all, so the
        # human recovery this tool exists for is unaffected.
        tree_known, lineage_parent, _ = _slot_tree_parent(slot_key)
        if not tree_known or lineage_parent != caller_key:
            raise deny(
                "archived session ownership could not be verified from gateway records "
                "(a fenced caller needs the crew log's session-tree lineage, which is "
                "missing when the log is off, its projection is unseeded, or the "
                "session predates it)",
                "ownership_unverified",
            )

    if folder_id:

        def _exists(folders: list[dict[str, Any]]) -> bool:
            return any(str(f.get("id") or "") == folder_id for f in _safe_folder_tree(folders))

        if not await state.read_folders(_exists):
            raise deny("folder not found", "folder_not_found", status=400)

    # A revive materialises the same resource a create does -- a live slot with
    # a hydrated transcript -- so it spends the same budgets: the per-caller
    # creation window and the per-creator and global slot caps.
    if not allow_create(SESSION_CREATE, caller_key):
        raise deny(
            "too many sessions opened recently; retry shortly", "create_rate_limited", status=429
        )
    if state.live_slot_count() >= MAX_LIVE_SLOTS:
        raise deny(f"slot cap reached ({MAX_LIVE_SLOTS})", "slot_cap_reached", status=429)
    if state.creator_slot_count(caller_key) >= MAX_SLOTS_PER_CREATOR:
        raise deny(
            f"per-caller slot cap reached ({MAX_SLOTS_PER_CREATOR})",
            "creator_slot_cap_reached",
            status=429,
        )

    def _plain(reason: str, code: str, status: int = 403) -> SessionControlError:
        # A non-auditing denier for the hook: its refusal is audited ONCE, by the
        # ``deny`` that raises it after the resume returns.
        return SessionControlError(reason, status=status, code=code)

    async def _containment(built: "_ChatSlot") -> ResumeRefusal | None:
        # The LAST gate, on the slot the resume hydrated and before it is
        # published (the core holds it retracted and under construction while
        # this awaits). Every answer read before the resume came from a metadata
        # snapshot taken before the transcript read and from a caller slot that
        # can change at any moment, so each is re-asserted here on live state:
        # the caller's own eligibility, the four target-side fields off the built
        # slot, the workspace, the gateway store's link and mirror records for the
        # target, the mirror on the built slot itself, and the two ceilings. A
        # refusal discards the built slot inside the core; nothing was published.
        try:
            live_caller = await asyncio.to_thread(
                refuse_caller_surface, state, caller_key=caller_key, deny=_plain
            )
        except SessionControlError as exc:
            return ResumeRefusal(exc.message, exc.code, exc.status)
        refusal = _revived_slot_refusal(built, live_caller)
        if refusal is not None:
            return ResumeRefusal(refusal[0], refusal[1], 403)
        linked, mirrored = await asyncio.to_thread(_archived_link_and_mirror, state, history_key)
        if linked:
            return ResumeRefusal(
                "channel-linked sessions are not addressable", "linked_session_target", 403
            )
        if mirrored or await asyncio.to_thread(_has_channel_mirror, state, built):
            return ResumeRefusal(
                "sessions mirrored to a channel are not addressable", "mirrored_target", 403
            )
        claimed_creator = str(meta.get("created_by") or "")
        if not getattr(built, "_created_by", "") and claimed_creator:
            # Creator attribution restored the way a restart's rehydrate restores
            # it (see ``_rehydrate_slot_from_history``): the ownership fence reads
            # ``_created_by``, and a fenced caller that revived its own worker
            # must still own it, or its next ``session_send`` is refused
            # ``not_creator``. Restored ONLY when the gateway-authored lineage
            # corroborates the claim: ``created_by`` is read from an agent-editable
            # file, and a stamp taken on the line's word alone would let a fenced
            # agent that wrote its own key into an archived transcript be handed
            # session-control reach over the operator's conversation the moment
            # an unfenced tab revived it (the fenced branch above checks the
            # lineage only for a fenced REVIVER, not for the claimed creator). An
            # unknown tree or a different parent leaves the field blank, which
            # matches no caller. Attribution ONLY -- ``_lineage_minted`` stays
            # False, and the reviver is never written here: reviving is not
            # creating.
            tree_known, lineage_parent, _ = _slot_tree_parent(slot_key)
            if tree_known and lineage_parent == claimed_creator:
                built._created_by = claimed_creator
        # Charge the reviver for this slot under the per-caller cap (see
        # ``creator_slot_count``) without touching its ownership. Set before the
        # ceilings are read so the count includes this slot once published.
        built._revived_by = caller_key
        # The ceilings. ``live_slot_count`` counts allocated-but-unpublished slots
        # too, so the built slot IS in it (retracted from the table, still under
        # construction) and the compare is strictly-over, the same exclusive
        # ceiling every other allocation path applies; ``creator_slot_count``
        # walks the table only, so the built slot is not in it and the compare
        # is inclusive.
        if state.live_slot_count() > MAX_LIVE_SLOTS:
            return ResumeRefusal(f"slot cap reached ({MAX_LIVE_SLOTS})", "slot_cap_reached", 429)
        if state.creator_slot_count(caller_key) >= MAX_SLOTS_PER_CREATOR:
            return ResumeRefusal(
                f"per-caller slot cap reached ({MAX_SLOTS_PER_CREATOR})",
                "creator_slot_cap_reached",
                429,
            )
        return None

    def _final_check(built: "_ChatSlot") -> ResumeRefusal | None:
        # SYNCHRONOUS last word, run by the core after its last await and right
        # before the publish: everything the hook answered without a store read
        # is re-asserted here on live state, so nothing can run between these
        # reads and the slot becoming reachable. The store-backed probes (link,
        # mirror) are the hook's; their guarded getters may not run on the loop.
        try:
            live_caller = _check_caller_slot_fields(state, caller_key, _plain)
        except SessionControlError as exc:
            return ResumeRefusal(exc.message, exc.code, exc.status)
        refusal = _revived_slot_refusal(built, live_caller)
        if refusal is not None:
            return ResumeRefusal(refusal[0], refusal[1], 403)
        if state.live_slot_count() > MAX_LIVE_SLOTS:
            return ResumeRefusal(f"slot cap reached ({MAX_LIVE_SLOTS})", "slot_cap_reached", 429)
        if state.creator_slot_count(caller_key) >= MAX_SLOTS_PER_CREATOR:
            return ResumeRefusal(
                f"per-caller slot cap reached ({MAX_SLOTS_PER_CREATOR})",
                "creator_slot_cap_reached",
                429,
            )
        return None

    outcome = await resume_slot_from_history(
        state,
        name=slot_key,
        history_key=history_key,
        caller_label=f"session:{caller_key}",
        containment=_containment,
        final_check=_final_check,
    )
    if outcome.refusal is not None:
        raise deny(outcome.refusal.error, outcome.refusal.code, status=outcome.refusal.status)
    slot = outcome.slot
    assert slot is not None

    if outcome.already_live:
        # Lost a race with a human click or another reviver between the
        # resolution above and the resume: same authorize-then-name answer as
        # the two pre-resume live branches, through the one builder.
        raise await _live_refusal(slot.key)

    def _audit_allowed(filed_now: bool) -> None:
        # The revive is committed the moment the core published the slot, so
        # the allowed record is owed whatever happens to the filing after it:
        # a cancelled request that skipped this row would leave a durable revive
        # with no SEL record, and audit rows are not reconstructible.
        _audit(
            caller_session_key=caller_key,
            operation="revive",
            slot_key=slot.key,
            outcome="allowed",
            detail={
                "messages": str(outcome.total),
                "folder_id": slot.folder_id or "",
                "filed": "true" if filed_now else "false",
                "was_closed": "true" if meta.get("closed") else "false",
            },
        )

    filed = False
    if folder_id and folder_id != slot.folder_id:
        # The revive is already COMMITTED -- the slot is live in the sidebar --
        # so a failure to file it must not propagate as "could not revive": the
        # caller would retry and be told the target is live. Same posture as
        # ``create_session``'s post-commit folder un-hide.
        #
        # Under the state-wide slot-metadata txn lock, the span the folder
        # endpoint (``api_chat_slot_folder``) and the sidebar filing writers
        # serialize their own mutate/save/rollback under: the un-hide and the
        # save below are awaits on an already-published slot, so a folder PATCH
        # could commit inside them and an unconditional ``previous`` restore
        # would then erase an acknowledged placement. Under the lock a rollback
        # can only undo this call's own write; the value compare stays as the
        # defense against the non-endpoint writers that do not take the lock.
        # The lock acquisition itself is an await (``LoopBoundLock`` under
        # contention), so the cancellation guard sits OUTSIDE ``async with``: a
        # cancellation delivered while waiting for the lock has written nothing
        # to roll back but must still leave the allowed audit row for a revive
        # that has already committed. The audit is emitted exactly once: here on
        # the exceptional path, or by the ``_audit_allowed(filed)`` call after
        # the block on the normal one (an ``Exception`` inside is swallowed into
        # the normal path, so it does not reach this arm).
        try:
            async with _slot_meta_txn_lock(state):
                previous = slot.folder_id
                previous_changed = slot._folder_changed
                # Re-read under the lock: filed there meanwhile by another writer
                # means nothing to do, and nothing this call may later roll back.
                if folder_id != previous:
                    slot.folder_id = folder_id
                    slot._folder_changed = True

                    def _roll_back() -> None:
                        if slot.folder_id == folder_id:
                            slot.folder_id = previous
                            slot._folder_changed = previous_changed

                    try:
                        if not await _unhide_folder(state, folder_id):
                            _roll_back()
                        elif await save_slot_off_loop(
                            state,
                            slot,
                            force=True,
                            # Strict, not best-effort: the default swallows a raised
                            # save and answers True (marking the slot dirty for the
                            # periodic flush), which would report ``filed: true`` for
                            # a placement that is not on disk and is lost if the
                            # gateway restarts before that flush. A raise takes the
                            # rollback arm below and reports ``filed: false``.
                            best_effort=False,
                            expected_history_key=slot_history_key(slot),
                        ):
                            note_folder_filed(state, folder_id)
                            filed = True
                        else:
                            _roll_back()
                            slot._dirty = True
                    except Exception:
                        logger.warning(
                            "revive_session: %s revived but filing it into %s failed",
                            slot.key,
                            folder_id,
                            exc_info=True,
                        )
                        _roll_back()
                        slot._dirty = True
                    except BaseException:
                        # A cancellation (gateway shutdown, the MCP client's timeout
                        # closing the socket) inside the two awaits above is not an
                        # ``Exception``: without this arm it would skip the rollback,
                        # leaving the provisional placement live and dirty. Undo this
                        # call's write and let it propagate to the audit arm below.
                        _roll_back()
                        slot._dirty = True
                        state.push_slots_update()
                        raise
        except BaseException:
            _audit_allowed(False)
            raise
        state.push_slots_update()

    _audit_allowed(filed)
    return {
        "ok": True,
        "target": slot.key,
        "title": slot.title or slot.key,
        "messages": outcome.total,
        "folder_id": slot.folder_id or "",
        "filed": filed,
    }


MAX_SEND_MESSAGE_CHARS = MAX_LONG_STRING

#: Provenance prefix on every delivered message. The target's transcript renders
#: the message as a user row, and without this line it is indistinguishable from
#: something the person typed — the same reason auto-nudge tags its injected
#: turns ``[auto-nudge cycle N]``. The model in the target session sees it too,
#: so it can weigh the instruction as coming from a peer session, not its user.
_SEND_PROVENANCE = "[sent by session {caller} via {via}]\n\n"

#: What the envelope names for a broadcast. The verb is part of the message's
#: meaning, not decoration: "the base moved, rebase before you push" addressed to
#: one worker is a fact about that worker's branch, and the same words addressed to
#: eight are a fact about the base. A worker that cannot tell which it received
#: cannot judge whether a sibling is about to make the same change.
BROADCAST_VIA = "session_broadcast"


@dataclass
class _DeliveryProgress:
    """Delivery-call-owned facts that survive cancellation into the caller."""

    steer_await_entered: bool = False


# Shielded steer deliveries whose awaiting frame was cancelled before they
# finished. A shielded task is referenced only by the shield wrapper our frame
# just dropped, so without a strong reference here the event loop may collect it
# MID-RPC -- which is the very interruption the shield exists to prevent. Entries
# remove themselves in the done callback, so the set holds at most the deliveries
# currently outliving their caller.
_ORPHANED_STEER_DELIVERIES: set["asyncio.Future[Any]"] = set()


def _retain_orphaned_steer_delivery(
    task: "asyncio.Future[Any]",
    slot_key: str,
    *,
    unavailable_outcome: str | None = None,
    on_unavailable: "Callable[[], Awaitable[None]] | None" = None,
) -> None:
    """Keep a shielded steer delivery alive after its caller stopped waiting.

    The caller has already reported this delivery as cancelled-with-unknown-outcome,
    so nothing downstream reads the result. Two things still have to happen. The
    delivery coroutine's OWN reconciliation -- popping the per-text steer maps and
    recording the transcript row -- which it does itself once it is not interrupted.
    And the caller's post-outcome handling for ``unavailable_outcome``, which a
    cancelled frame cannot run: that outcome means the text was never handed over,
    so without ``on_unavailable`` queueing it the instruction is lost rather than
    merely delayed. Logged either way, because this is the only trace that a
    delivery completed after its caller moved on.
    """
    _ORPHANED_STEER_DELIVERIES.add(task)

    def _done(finished: "asyncio.Future[Any]") -> None:
        _ORPHANED_STEER_DELIVERIES.discard(finished)
        if finished.cancelled():
            # Only a loop shutdown reaches here: the shield absorbed the caller's
            # cancellation, so nothing else cancels this task.
            logger.warning(
                "session_send: shielded steer delivery to %s was cancelled outright; "
                "its steer bookkeeping may not have reconciled",
                slot_key,
            )
            return
        exc = finished.exception()
        if exc is not None:
            logger.warning(
                "session_send: shielded steer delivery to %s finished with %r after "
                "its caller stopped waiting",
                slot_key,
                exc,
            )
            return
        outcome = finished.result()
        logger.info(
            "session_send: shielded steer delivery to %s finished with outcome=%s "
            "after its caller stopped waiting",
            slot_key,
            outcome,
        )
        if on_unavailable is None or outcome != unavailable_outcome:
            return
        # Scheduled rather than awaited: a done callback runs on the loop and cannot
        # await. Retained in the same set for the same reason the delivery is -- a
        # task referenced only by a local would be collectible mid-queue.
        fallback = asyncio.ensure_future(on_unavailable())
        _ORPHANED_STEER_DELIVERIES.add(fallback)

        def _fallback_done(done: "asyncio.Future[Any]") -> None:
            _ORPHANED_STEER_DELIVERIES.discard(done)
            if done.cancelled():
                logger.warning(
                    "session_send: the queue fallback for the orphaned steer to %s was "
                    "cancelled; that message is not queued",
                    slot_key,
                )
                return
            exc = done.exception()
            if exc is not None:
                logger.warning(
                    "session_send: the queue fallback for the orphaned steer to %s "
                    "failed with %r; that message is not queued",
                    slot_key,
                    exc,
                )

        fallback.add_done_callback(_fallback_done)

    task.add_done_callback(_done)


async def send_to_target(
    state: "DashboardState",
    *,
    caller_session_key: str,
    target: str,
    message: str,
    steer: bool = False,
    caller_fenced: bool | None = None,
    via: str = "session_send",
    _delivery_progress: "_DeliveryProgress | None" = None,
) -> dict[str, Any]:
    """Deliver *message* to *target* as its next agent turn.

    ``via`` names the VERB in the provenance envelope the target reads. It exists
    because :func:`broadcast_to_targets` delivers through this function, and a
    worker told something every sibling was also told must be able to see that:
    the same sentence means one thing addressed to one session and another
    addressed to eight. It never affects authorization or delivery.

    The delivery path is the same queue-vs-run decision the dashboard composer
    uses (``enqueue_or_run_prompt``): an idle target starts a turn immediately,
    a busy one queues the message for its next turn. Both outcomes are reported
    distinctly — ``started`` says which happened — because "it ran" and "it will
    run later" must not look the same to a caller coordinating several sessions.
    A queued delivery is re-validated at the drain: the entry
    carries the containment that held here, and a constraint newly held at
    delivery time drops it with a visible notice instead of executing it under
    the weaker authorization that admitted it. The entry also carries THIS
    caller's slot, so that drop appends a notice to the caller's own transcript
    as well: ``started: False`` says the message will run later, and a caller
    told only that would otherwise wait for a reply that can never come.

    ``steer`` asks for a THIRD outcome on a busy target: the message cuts into
    the turn already running (``steer_into_running_turn``) instead of waiting for
    it, so a caller watching a worker go the wrong way can say so while the work
    is still in flight rather than after it lands. Reported as ``steered``, its
    own field, for the same reason ``started`` is one.

    Two properties make that arm safe rather than a hole in the queued arm's
    checks. The delivery happening NOW removes the queue's waiting window and
    leaves a narrower one: the steer RPC suspends on ``stdin.drain()``.

    * Containment is re-validated at the queue drain because a queued prompt runs
      later than the moment it was authorized, so the target can gain a channel
      link or an outbound mirror while it waits. Nothing suspends between
      ``authorize_target`` and the steer call, so the text is committed against the
      containment the gate cleared — but the RPC's own suspension is a window, and
      each outcome closes it differently. The fallback re-runs the gate before
      taking the queue arm and refuses a target that resolves to a different slot
      OBJECT. A requeued steer becomes an ordinary queued entry and meets the drain
      unexempted. A SUCCESSFUL steer cannot be recalled, so the admission
      containment is compared against the post-RPC containment and the turn is
      stopped when a constraint newly holds — the reply path resolves its mirror
      live, so a mirror linked mid-turn is a real audience.
    * Provenance is carried by the text, not by the delivery mode: both arms hand
      over the same redacted, envelope-prefixed prompt, so neither the target
      agent nor a person reading the transcript can read an injected message as
      something the human typed. It also travels to the requeue as
      ``user_origin=False``, so a peer's steer does not inherit the composer's
      exemption from the drain's LINKED drop.

    A steer that cannot be injected falls back to the queue — never dropped, and
    the caller is told it queued rather than steered.

    The turn is NOT charged against the background-turn cap, and deliberately so:
    that cap only binds unattended (app-owned) slots, and every target this
    function can authorize is attended — see the comment at the delivery call.
    """
    # A broadcast passes its own instance so the timeout handler can read facts
    # this delivery recorded before cancellation erased its stack. A plain send
    # needs the same local state even though no outer timeout reads it.
    delivery_progress = (
        _delivery_progress if _delivery_progress is not None else _DeliveryProgress()
    )

    # The delivery appends to the target's rows: a target restored as a
    # sidebar row loads its window first, ahead of the prewarm below.
    await load_target_window(state, target)
    # Same prewarm ordering as `stop_target`, for the same reasons: the SEL
    # write inside `authorize_target`'s deny path must be a cache hit, and the
    # config warm must be the LAST suspension before the synchronous gate.
    try:
        await asyncio.to_thread(sel)
    except Exception:  # noqa: BLE001 - a prewarm failure must not fail the send
        logger.warning("session-control SEL prewarm failed", exc_info=True)
    await prewarm_enabled_check()

    body = message.strip()
    if not body:
        raise SessionControlError("message is empty", code="message_empty", status=400)
    if len(body) > MAX_SEND_MESSAGE_CHARS:
        raise SessionControlError(
            f"message exceeds {MAX_SEND_MESSAGE_CHARS} characters",
            code="message_too_long",
            status=400,
        )

    #: The manager-line switch as THIS admission sees it, stamped onto the entry below
    #: so the drain re-asserts the policy that admitted it and reads no config itself.
    #:
    #: Resolved HERE, ahead of the authorization, because the read SUSPENDS and a
    #: suspension after the verdict is a window in which the verdict stops being true:
    #: a handoff committing there revokes the sender while this call is parked, and the
    #: target then runs a message its sender was no longer allowed to send. Nothing
    #: between the authorization and the delivery may wait.
    manager_chain_applied: bool = note_chain_switch(await drained_to_thread(manager_chain_enabled))

    slot = authorize_target(
        state,
        caller_session_key=caller_session_key,
        target=target,
        operation="send",
        precomputed_ownership_fenced=caller_fenced,
        # THE SAME SNAPSHOT the stamp below carries. Read twice, the two could disagree
        # -- an operator flipping the switch between them admits a delivery under the
        # line and then stamps it as one-level, and the drain deletes an accepted
        # message whose reporting line never changed. One read, one answer, both uses.
        precomputed_manager_chain=manager_chain_applied,
    )

    # A relay archive is read-only: the delivery below would run the crew's old
    # session on THIS machine, the same thing the send / regenerate / rewind /
    # continue paths refuse.
    if is_relay_archive(slot):
        raise SessionControlError(RELAY_ARCHIVE_ERROR, code=RELAY_ARCHIVE_CODE, status=409)

    # Cancellation can only enter at an await. Route every await below the
    # authorization through this helper so no authorized cancellation loses its
    # target-level permission trail.
    async def _await_authorized_delivery(awaitable: Any) -> Any:
        try:
            return await awaitable
        except asyncio.CancelledError:
            # `cancelled` means authorization was granted, delivery was cancelled
            # mid-flight, and whether the message landed is unknown.
            _audit(
                caller_session_key=caller_session_key,
                operation="send",
                slot_key=slot.key,
                outcome="cancelled",
                detail={
                    "steer_requested": bool(steer),
                    "chars": len(body),
                    "delivery_result": "unknown",
                },
            )
            raise

    # Deferred for the same import cycle `stop_target` documents.
    from kiro_crew.dashboard.chat_runner import _run_chat

    caller_key = caller_slot_key(state, caller_session_key)

    # ONE fence verdict for this call, resolved before the gate and used by the gate
    # AND by every send-origin stamp below. ``None`` means "not resolved yet" -- the
    # HTTP handler's carried verdict is None for any caller that is not a crew member
    # -- and coercing it per stamp with ``bool()`` reads an unresolved verdict as
    # UNFENCED. The stamp is what decides whether the drain may re-compare the manager,
    # so that coercion exempts every agent-created and cron sender from the re-check,
    # and the message it queued then runs after it has handed the session away. The
    # verdict a stamp records has to be the same one authorization used, which means
    # deriving it once.
    if caller_fenced is None:
        caller_fenced = bool(caller_key) and _caller_is_ownership_fenced(state, caller_key)
    #: The RESOLVED verdict, under its own name so the closures below cannot read the
    #: possibly-None argument by accident -- which is the mistake this whole block
    #: exists to undo, and the one mypy catches here rather than at a drain six hours
    #: later.
    manager_fence_applied: bool = caller_fenced
    # Sanitized on the same grounds as the steer path (``chat_delivery`` sanitizes
    # before ``slot.append``): this body comes from ANOTHER session and is persisted
    # into — and broadcast from — the target's transcript, so raw content must never
    # reach that surface. The length gate above deliberately measures the RAW body:
    # redaction can only shrink the text, so validating the raw form is the honest
    # limit and keeps the error keyed to what the caller actually sent.
    prompt = _SEND_PROVENANCE.format(
        caller=caller_key or "unknown", via=via or "session_send"
    ) + sanitize_outbound(body)

    steered = False
    requeued = False
    # Containment names that newly held when a successful steer's turn was stopped;
    # empty on every other path. Read by the audit below.
    steer_audience_changed: list[str] = []
    # True only when the containment stop below raised. The steer was already
    # consumed by the turn at that point, so the failure travels in the audit trail
    # and the caller still sees the delivery it got.
    steer_containment_stop_failed = False
    if steer and slot.running:
        # The mid-turn arm. ONE text is handed to both arms, so what the target
        # reads and what its transcript keeps are the same bytes either way:
        # already redacted, already carrying the provenance envelope, so an
        # injected steer can no more pose as human typing than a queued delivery
        # can. ``chat_delivery`` runs its own ``sanitize_outbound`` before it
        # appends the row, which is a second pass over text that already cleared
        # the same guard.
        #
        # Gated on ``slot.running`` because a steer needs a turn to cut into: the
        # turn publishes the steer-capable client and clears it at teardown, so on
        # an idle slot there is nothing to inject and the queue-or-run arm below is
        # the whole delivery.
        #
        # Deferred import for the cycle `_run_chat` above documents.
        from kiro_crew.dashboard.chat_delivery import (
            STEER_REQUEUED,
            STEER_STEERED,
            STEER_UNAVAILABLE,
            steer_into_running_turn,
        )

        # Captured BEFORE the RPC and synchronous with the gate above -- no await
        # separates `authorize_target` from here -- so this IS the containment the
        # authorization cleared, recorded in the drain's own vocabulary so the
        # comparison below is the same one queued prompts get.
        #
        # The sending slot rides the same dict because the REQUEUE copies it onto
        # the entry verbatim, so a steer that falls back to the queue and is then
        # dropped at the drain reports back to us like any queued delivery. Inert
        # for the two other readers: `newly_held_constraints` reads only the
        # containment key, and so does the audience fence below.
        admission = {
            **containment_meta(state, slot),
            **send_origin_meta(
                state,
                caller_key,
                manager_fenced=manager_fence_applied,
                # The switch as THIS admission saw it, so the drain re-asserts
                # the policy that admitted the entry and reads no config itself.
                chain=manager_chain_applied,
            ),
        }

        # Which turn this steer is going into. `_turn_generation` increments on every
        # task assignment, so it identifies a turn even if a later task object reuses
        # an address. Captured here, synchronously with the gate, because the stop
        # below must not cancel a DIFFERENT turn: if the steered turn ends during the
        # RPC and a queued prompt starts the next one, `slot` still reads as running
        # and an unguarded stop would cancel work that never received this text.
        steered_turn_generation = slot._turn_generation

        # Set BEFORE the RPC, synchronously with the gate above, because this is the
        # half the post-RPC stop cannot cover: `_deliver_cross_surface_reply` runs
        # from the turn's own completion path and resolves the mirror live, so a turn
        # that finishes while this coroutine is suspended would publish to the new
        # audience before any check of ours resumes. Reacting cannot recall a sent
        # reply; withholding until the check has run can.
        audience_fence = uuid.uuid4().hex
        slot._steer_audience_fences[audience_fence] = admission

        # `user_origin=False`: this text was not typed into the target's own
        # surface. The requeue reads it to decide `directive_user_origin`, and a
        # peer must not inherit the composer's exemption from the LINKED drop.
        async def _steer_delivery() -> str:
            return await steer_into_running_turn(
                state, slot, prompt, user_origin=False, admission=admission
            )

        delivery_progress.steer_await_entered = True

        # Shielded, because THIS await is what `broadcast_to_targets` cancels when a
        # target overruns `BROADCAST_TARGET_ALLOWANCE_SECS`, and
        # `steer_into_running_turn` guards its own `client.steer` with
        # `except Exception` -- which does not catch `CancelledError`. Awaited
        # directly, the cancellation unwinds THROUGH the RPC and skips that
        # coroutine's single reconciliation tail, while the bytes may already have
        # reached kiro-cli: the turn then runs text no `slot.append` recorded, and
        # `_steer_delivery_ids` / `_steer_send_ids` / `_steer_user_origin` /
        # `_steer_admissions` are never popped. Nothing else pops them --
        # `_settle_consumed_steers` clears only the attachment and decision-strip
        # maps, and `_requeue_unconsumed_steers` returns early once settling emptied
        # `_pending_steers`. The surviving `_steer_delivery_ids` entry then refuses
        # this exact text on that slot forever (the one-per-text guard reads that
        # dict) and `retained_steer_count` never falls back below
        # `MAX_PENDING_STEERS`, after which the slot refuses every steer.
        #
        # The shield splits the two halves the cancellation conflated: our frame
        # still receives it, so the broadcast reports its timeout row on schedule
        # and the caller still hears "outcome unknown", while the delivery runs to
        # its own end and reconciles itself. Inert on the direct `session_send`
        # path, which has no per-target budget above it.

        # The fallback a CANCELLED frame cannot reach. `STEER_UNAVAILABLE` means no
        # live steer-capable client, an RPC that lost the text, or an identical steer
        # already in flight -- in every case the text was never handed over and the
        # shielded delivery has cleared its own per-text state, so nothing downstream
        # holds it. The arm below queues it instead of dropping it; once the frame is
        # gone that arm cannot run, and shielding the delivery
        # without this would turn an outcome the caller recovers from into a lost
        # instruction. Re-gated on the same terms as that arm, because the RPC
        # suspended and the queue entry records the containment for the drain to
        # re-assert: a snapshot taken now must not certify a link the authorization
        # never saw.
        async def _queue_after_orphaned_unavailable() -> None:
            await prewarm_enabled_check()
            regated = authorize_target(
                state,
                caller_session_key=caller_session_key,
                target=target,
                operation="send",
                precomputed_ownership_fenced=caller_fenced,
            )
            if regated is not slot:
                # Object identity, for the reason the in-frame arm gives: a target
                # closed and resumed under the same key is a different object whose
                # key still compares equal, and queueing onto the detached one loses
                # the text as surely as dropping it.
                logger.warning(
                    "session_send: orphaned steer to %s came back unavailable, but the "
                    "target resolved to a different session; the message is not queued",
                    slot.key,
                )
                return
            slot.enqueue_or_run_prompt(
                prompt,
                _run_chat,
                state,
                extra_meta=send_origin_meta(
                    state,
                    caller_key,
                    manager_fenced=manager_fence_applied,
                    # THE SAME snapshot the admission above used, not a fresh read. These
                    # two stamps are written further down the same verb, so omitting it
                    # read the switch again on the loop -- and an operator flipping it
                    # meanwhile would stamp a policy this admission never applied, which
                    # is the one thing the field exists to record.
                    chain=manager_chain_applied,
                ),
            )
            logger.info(
                "session_send: orphaned steer to %s came back unavailable and was "
                "queued instead of lost",
                slot.key,
            )

        _steer_task = asyncio.ensure_future(_steer_delivery())
        try:
            outcome = await _await_authorized_delivery(asyncio.shield(_steer_task))
        except asyncio.CancelledError:
            _retain_orphaned_steer_delivery(
                _steer_task,
                slot.key,
                unavailable_outcome=STEER_UNAVAILABLE,
                on_unavailable=_queue_after_orphaned_unavailable,
            )
            raise
        steered = outcome == STEER_STEERED
        # The turn ended while the steer RPC was suspended and its teardown moved
        # the text onto the queue: it WILL run, and taking the queue arm below
        # would deliver it a second time. Reported as a queued delivery, which is
        # what it now is.
        requeued = outcome == STEER_REQUEUED
        if requeued and state._slots.get(slot.key) is not slot:
            # The teardown queued this text on THIS slot object, and the key has
            # since stopped resolving to it -- the target was closed and recreated
            # while the RPC was suspended on `stdin.drain()`. The queue that now
            # holds the prompt belongs to a detached object no drain will ever
            # reach, so the text is gone. Refusing is the only honest answer: the
            # `steered` arm already guards this same replacement (it records
            # `slot_replaced` below) and the fallback arm re-runs the gate, so this
            # was the one outcome that reported success for a prompt that had
            # quietly stopped existing.
            #
            # Object identity, not `key != key`: a fresh object under the same key
            # compares equal by key and would pass a key comparison.
            raise SessionControlError(
                "that target was replaced while the steer was in flight and the "
                "message did not survive the hand-over; re-read the session list "
                "and send again",
                code="target_moved",
                status=409,
            )
        if steered:
            # The text is IN the running turn: `STEER_STEERED` means kiro-cli
            # acknowledged consumption, so unlike the requeue arm there is nothing
            # left to remove and unlike the fallback arm there is nothing left to
            # refuse. What can still be prevented is the turn's remaining output
            # reaching an audience this send was never authorized against -- a
            # mirror linked, retargeted, or a channel link bound while the RPC was
            # suspended on `stdin.drain()`. `_deliver_cross_surface_reply` resolves
            # the mirror LIVE at reply-delivery time, so that audience is real and
            # not fixed at turn start.
            #
            # So: compare, and stop the turn when a constraint newly holds. One
            # cooperative cancel, the same thing the Stop button's first press does,
            # with `escalate=False` because this is one automatic decision and not a
            # person who watched a cancel fail to take.
            #
            # This comparison does NOT decide whether the reply may be published. The
            # admission recorded before the RPC stays on the slot for the whole turn,
            # and the publisher re-evaluates it at delivery -- a mirror bound between
            # this moment and the reply would be just as unauthorized, and only the
            # publisher's own moment can see it.
            steer_audience_changed = newly_held_constraints(
                containment_snapshot(state, slot, on_probe_failure=True), admission
            )
            if state._slots.get(slot.key) is not slot:
                # The slot object was replaced under the same name while the RPC was
                # in flight, so the turn holding this text belongs to a session that
                # the name has stopped resolving to. Reported alongside the containment
                # names rather than folded into them: it is an identity change, not
                # a constraint that newly holds.
                steer_audience_changed = [*steer_audience_changed, "slot_replaced"]
            if steer_audience_changed:
                # Deferred import for the cycle the steer import above documents.
                from kiro_crew.dashboard.chat_handlers import stop_slot_turn

                if slot._turn_generation != steered_turn_generation:
                    # The turn this text entered has already ended and another one
                    # has started on the same slot. Stopping now would cancel work
                    # that never received this steer, which is worse than the
                    # exposure being narrowed: the steered turn is over, so whatever
                    # it published is already published, and the fence recorded
                    # before the RPC still withholds the cross-surface legs for the
                    # remainder of the turn that holds it.
                    steer_audience_changed = [
                        *steer_audience_changed,
                        "turn_rolled_over_stop_skipped",
                    ]
                else:
                    # The text is already consumed by the turn at this point, so this
                    # stop is a best-effort narrowing of what the turn goes on to
                    # publish -- it is NOT part of delivering the message. Letting it
                    # raise would abort before the success audit and the ok:True
                    # return below, reporting a steer the turn has already taken as
                    # failed; the caller's natural response is to send again, which
                    # delivers the same text twice. The turn's own reply legs consult
                    # the audience fence recorded before the RPC, so containment does
                    # not depend on this stop succeeding.
                    try:
                        await _await_authorized_delivery(
                            stop_slot_turn(
                                state,
                                slot,
                                source="session_send_steer_containment",
                                escalate=False,
                            )
                        )
                    except Exception:
                        logger.exception(
                            "Containment stop failed after a steer was consumed "
                            "(slot=%s, newly held=%s); the fence still withholds "
                            "cross-surface publication",
                            slot.key,
                            steer_audience_changed,
                        )
                        steer_containment_stop_failed = True
        else:
            # Not steered: no text of ours is inside the running turn on this path, so
            # there is nothing for the record to protect -- the requeue arm's entry
            # faces the drain and the fallback arm re-runs the gate. Dropped by key so
            # another peer's concurrent steer keeps its own.
            slot._steer_audience_fences.pop(audience_fence, None)
        if not steered and not requeued:
            # STEER_UNAVAILABLE: no live steer-capable client, an RPC that lost the
            # text, or an identical steer already in flight. The message falls back
            # to the queue arm rather than being dropped — but that arm records the
            # containment holding at APPEND time for the drain to re-assert, and
            # the RPC above suspends, so the state it would record can differ from
            # the state this send was authorized against. Re-run the gate so the
            # snapshot cannot certify a link the authorization never saw.
            # The prewarm before the gate at the top of this function predates the
            # steer RPC, which suspends: a config edit during that suspension
            # invalidates the cache, and `authorize_target` -> `session_control_enabled`
            # would then run `KiroCrewConfig.load()` synchronously on the shared loop,
            # stalling every other session while a file is read and validated. Warming
            # again here is what `prewarm_enabled_check`'s own docstring prescribes for
            # a gate that sits after an await.
            async def _prewarm_regate() -> None:
                await prewarm_enabled_check()

            await _await_authorized_delivery(_prewarm_regate())
            regated = authorize_target(
                state,
                caller_session_key=caller_session_key,
                target=target,
                operation="send",
                precomputed_ownership_fenced=caller_fenced,
            )
            if regated is not slot:
                # Object identity, not ``regated.key != slot.key``. The delivery
                # below uses THIS ``slot`` object, so a target closed and resumed
                # under the same key during the RPC yields a fresh object whose key
                # still compares equal: the guard would pass, the message would land
                # on the detached object, and the caller would be told it succeeded.
                raise SessionControlError(
                    "that target resolved to a different session while the steer "
                    "was in flight; re-read the session list and send again",
                    code="target_moved",
                    status=409,
                )

    if steered or requeued:
        # Neither arm starts a turn: a steer runs inside one that is already going,
        # and a requeued steer waits for the next like any queued message.
        started = False
    else:
        # `_run_chat` is passed straight through, NOT wrapped in
        # `state.run_background_turn`: that cap is structurally unreachable here.
        # `run_background_turn` returns the coroutine untouched for an attended slot
        # (`state.py`, "this wrapper is inert"), `_ChatSlot.unattended` is
        # `bool(self._app) and not self._human_seen`, and `authorize_target` refuses
        # every `_app` target above (`app_scoped_target`) — so no target this
        # function can reach is ever unattended, and a wrapper would only add a
        # never-taken timeout arm. The composer's own queued path does the same
        # (`server.py` passes `_run_chat` directly).
        started = bool(
            slot.enqueue_or_run_prompt(
                prompt,
                _run_chat,
                state,
                # Only the QUEUE arm keeps it (the run arm has no entry): a
                # delivery that waits is the one a later drain can drop, and this
                # stamp is what lets that drop be reported back to us instead of
                # ending at the target's own transcript.
                extra_meta=send_origin_meta(
                    state,
                    caller_key,
                    manager_fenced=manager_fence_applied,
                    # THE SAME snapshot the admission above used, not a fresh read. These
                    # two stamps are written further down the same verb, so omitting it
                    # read the switch again on the loop -- and an operator flipping it
                    # meanwhile would stamp a policy this admission never applied, which
                    # is the one thing the field exists to record.
                    chain=manager_chain_applied,
                ),
            )
        )
    try:
        state.push_slots_update()
    except Exception:  # pragma: no cover - sidebar refresh is best-effort
        logger.debug("session_send: push_slots_update failed", exc_info=True)

    _audit(
        caller_session_key=caller_session_key,
        operation="send",
        slot_key=slot.key,
        outcome="allowed",
        # `steer` is what the caller ASKED for and `steered` is what happened, so a
        # fallback to the queue is readable in the trail rather than looking like a
        # caller that never asked.
        detail={
            "started": started,
            "steer_requested": bool(steer),
            "steered": steered,
            "chars": len(body),
            # Only when a steer's turn was actually stopped, so the common shape is
            # unchanged. Names the constraints that newly held, which is what a
            # reader needs to tell this apart from an ordinary delivery: the message
            # WAS delivered into the turn and the turn was then cancelled because
            # its audience had changed under the RPC.
            **({"steer_stopped_on": steer_audience_changed} if steer_audience_changed else {}),
            # The stop raised. Present only in that case, so its absence is not a
            # claim that a stop succeeded on a path where none was attempted. The
            # delivery outcome above is unchanged: the turn had already taken the
            # text, and the audience fence recorded before the RPC still withholds
            # cross-surface publication whether or not the stop took.
            **({"steer_containment_stop_failed": True} if steer_containment_stop_failed else {}),
        },
    )
    return {"ok": True, "target": slot.key, "started": started, "steered": steered}


#: The two broadcast modes, named rather than a boolean. ``queue`` waits for each
#: target's current turn; ``steer`` cuts into it. They are separate names because
#: the choice is not a refinement of one delivery — "tell everyone when they next
#: come up for air" and "interrupt everyone now" are different instructions, and a
#: caller that means the first must not reach the second by defaulting.
BROADCAST_MODES = ("queue", "steer")

#: The per-delivery bound this module ENFORCES, and the MCP client's one request
#: budget, are one name in ``kiro_crew.validation`` rather than two literals that
#: must be remembered together -- the same reason ``MAX_BROADCAST_TARGETS`` lives
#: there. See that definition for why a client budget below this bound would
#: discard the per-target report.

#: The refusal code a timed-out delivery reports. Distinct from
#: ``delivery_failed`` because a timeout is classified from delivery-owned
#: progress plus any exact text the target still retains: pre-authorization,
#: retained text, ambiguous or completed hand-over, and an unavailable target
#: each need different caller guidance. See :func:`broadcast_to_targets`.
BROADCAST_TIMEOUT_CODE = "delivery_timeout"

_TIMEOUT_RETAINED = "retained"
_TIMEOUT_NOT_HANDED_OVER = "not_handed_over"
_TIMEOUT_UNKNOWN = "unknown"


def _timeout_delivery_observation(
    state: "DashboardState",
    slot: Any | None,
    prompt: str,
    delivery_progress: _DeliveryProgress,
) -> str:
    """Classify a cancelled delivery from its own progress and retained text."""
    if slot is None or state._slots.get(slot.key) is not slot:
        return _TIMEOUT_UNKNOWN
    if prompt in slot._pending_steers or any(
        isinstance(entry, dict) and entry.get("content") == prompt for entry in slot._queue
    ):
        return _TIMEOUT_RETAINED
    if delivery_progress.steer_await_entered:
        return _TIMEOUT_UNKNOWN
    return _TIMEOUT_NOT_HANDED_OVER


def broadcast_audience(state: "DashboardState", caller_key: str) -> list[str]:
    """The slot keys *caller_key* HOLDS and can still address, sorted.

    The DEFAULT audience of a broadcast: a conductor's own workers. Read through
    :func:`_managed_by_other`, the same reading the ownership fence uses for the
    verbs a handoff widens — and ``send`` is one of them, so this has to agree with
    it. Reading ``_created_by`` here instead would put a handed-away session in its
    giver's default audience, where every delivery is then refused, and leave it out
    of the audience of the manager that can actually reach it. On a gateway where
    nothing has been handed over the two readings are the same value, because
    :func:`manager_of` answers ``_created_by`` until a handoff writes the field.

    An audience this returns can still be refused per target (a worker that went
    incognito, gained a channel mirror, or is an app's) by the gate each delivery
    runs. Nothing here authorizes a delivery.

    A slot whose manager cannot be read is EXCLUDED, not refused, and the direction
    matters. Enumerating is not authorizing: for the FENCE, an unknown manager must
    refuse, because answering would GRANT a reach the log cannot show. For an AUDIENCE,
    an unknown manager must exclude, because excluding NARROWS reach. Both are fail
    closed; they differ because one grants and the other lists.

    Refusing the whole audience instead is an outage, and a long one: one open slot
    whose recorded manager was closed before a restart carries
    ``_manager_grant_inadmissible`` until a fresh handoff clears it, so every default
    broadcast and every roster read would 409 for every caller -- including unfenced
    owners, and including other workspaces -- for the life of the process. The cost of
    excluding is that a caller which genuinely holds that slot does not reach it by
    broadcast; it can still address it by name, where
    :func:`authorize_target` gives it the real refusal with the real reason.

    The one exception is a caller that could hold the slot whatever the read says: its
    creator, or the session the field already names. Excluding those would hide a
    caller's own session from its own roster on a read fault.

    The caller itself is excluded: ``authorize_target`` refuses a self-target, and
    silently collecting a refusal the caller could not have avoided would make
    every broadcast report one failure.

    Sorted by key so the delivery order is stable across calls. A broadcast is not
    atomic — deliveries land one at a time — so a caller reading two reports needs
    the same order in both to compare them.
    """
    out = [
        key
        for key, slot in list(state._slots.items())
        if key != caller_key and _holds_for_audience(state, slot, caller_key)
    ]
    return sorted(out)


def _holds_for_audience(state: "DashboardState", slot: Any, caller_key: str) -> bool:
    """Whether *caller_key* holds *slot* for the purpose of LISTING it.

    The audience counterpart of :func:`_managed_by_other`, and the only place the two
    differ: an unreadable manager excludes the slot here instead of raising, because
    excluding narrows reach while the fence's refusal protects a grant. See
    :func:`broadcast_audience` for why the directions are opposite.

    A caller that could hold the slot whatever the read says still gets it -- the
    creator, or the session the field already names -- so a read fault cannot hide a
    caller's own session from its own roster.

    The caller's own identity goes with its key, same as at the fence. A mismatch
    EXCLUDES here rather than raising, which is this function's whole direction: a
    replacement under a retired manager's name does not get its predecessor's workers
    listed to it, and excluding is the narrowing answer.
    """
    try:
        return not _managed_by_other(state, slot, caller_key, _tab_identity(state, caller_key))
    except SessionControlError:
        # The field names the caller AND the record names the session asking. Both,
        # even on a read fault: the fault is a reason the manager could not be read,
        # not a reason to stop asking which session the record was written for, and a
        # replacement under a retired manager's name is the one case this fallback
        # must not catch. Asked through :func:`manager_record_names` so the fault path
        # and the ordinary path compare the same way.
        if caller_key and caller_key == str(getattr(slot, "_manager", "") or ""):
            return manager_record_names(slot, _tab_identity(state, caller_key))
        return bool(caller_key) and caller_key == str(getattr(slot, "_created_by", "") or "")


def _too_many_targets(count: int) -> SessionControlError:
    """The audience-cap refusal, from one place because it is raised from two.

    Refused rather than truncated. A silently-cut audience is a broadcast the
    caller believes reached everyone, and the sessions past the cut are the ones it
    will never think to check.
    """
    return SessionControlError(
        f"a broadcast reaches at most {MAX_BROADCAST_TARGETS} sessions and this "
        f"one names {count}; send to a named subset instead",
        code="too_many_targets",
        status=400,
    )


async def broadcast_to_targets(
    state: "DashboardState",
    *,
    caller_session_key: str,
    message: str,
    mode: str,
    targets: "list[str] | None" = None,
    caller_fenced: bool | None = None,
) -> dict[str, Any]:
    """Deliver *message* to several sessions at once, one at a time.

    Two modes, and they are the whole point of the verb being separate from a loop
    of :func:`send_to_target` calls the model writes itself: ``queue`` lets every
    target finish what it is doing, ``steer`` cuts into every running turn. A
    conductor telling eight workers "the base moved, rebase before you push" wants
    the first; one telling them "stop, the issue was already fixed" wants the
    second, and the difference is worth eight turns of latency.

    ``targets`` omitted means :func:`broadcast_audience` — the sessions this caller
    created. Naming them explicitly is for the subset case, and every named target
    goes through the same :func:`authorize_target` gate, so an explicit list can
    only ever be as permissive as one send at a time.

    PARTIAL DELIVERY IS THE NORMAL OUTCOME and the return shape says so per
    target. A target closed between the audience read and its own delivery, one
    that went incognito, one bound to a remote crew: each is one row with its
    refusal ``code``, and the remaining deliveries still happen. The alternative —
    abort on the first refusal — would leave a broadcast half-delivered with
    nothing saying which half, which is the one outcome a caller cannot recover
    from.

    Deliveries are SEQUENTIAL, never gathered. Each one takes the same slot locks,
    runs the same containment re-checks and (for a steer) suspends on an RPC, and
    running them concurrently would interleave those windows across sessions for
    no gain a caller can observe — the report is only read once all of them are
    done either way.

    Which is exactly why each delivery carries its own bound
    (``BROADCAST_TARGET_ALLOWANCE_SECS``): sequential delivery means one target
    that never answers is one target that starves every target behind it, and the
    urgent mode is the one that suspends on an RPC with no ceiling under it. On
    expiry that delivery is cancelled, reported as a ``delivery_timeout`` row, and
    the loop continues. The delivery records ONE marker: that it entered the steer
    await. Cancellation before that marker is provably before hand-over, so
    re-sending is safe. Cancellation once it is set is ambiguous -- inside the
    await or after it returned are indistinguishable here, and a second marker for
    the return would not separate them, because it could only be set once this one
    already was. Both read as unknown and must not invite a retry. The target's
    ``_pending_steers`` and queue
    still add useful information when they retain the exact text, but their shared
    absence cannot reconstruct delivery history. A target whose slot disappeared
    or was replaced also gets an unknown row. None of the rows claims delivery or
    certain execution.
    """
    if mode not in BROADCAST_MODES:
        raise SessionControlError(
            f"mode must be one of {', '.join(BROADCAST_MODES)}",
            code="invalid_broadcast_mode",
            status=400,
        )
    body = message.strip()
    if not body:
        raise SessionControlError("message is empty", code="message_empty", status=400)
    if len(body) > MAX_SEND_MESSAGE_CHARS:
        raise SessionControlError(
            f"message exceeds {MAX_SEND_MESSAGE_CHARS} characters",
            code="message_too_long",
            status=400,
        )

    # The caller gate runs ONCE, here, before any audience is read: a caller that
    # may not control sessions at all must be refused without learning how many it
    # created. The per-target gate below re-runs it for each delivery, which is
    # what actually binds every send — this one is the early, cheap refusal.
    await prewarm_enabled_check()
    deny = _deny_factory(caller_session_key=caller_session_key, operation="broadcast", target="")
    caller_key = refuse_caller_identity(state, caller_session_key=caller_session_key, deny=deny)
    caller_slot = refuse_caller_surface(state, caller_key=caller_key, deny=deny)
    # Captured for the re-check before the return below, on the same terms
    # `created_session_status` captures it: this verb suspends many times between
    # here and its payload, and the rows it returns name the caller's own sessions.
    caller_workspace = str(getattr(caller_slot, "workspace", "default"))
    ownership_fenced = (
        _caller_is_ownership_fenced(state, caller_key) if caller_fenced is None else caller_fenced
    )
    resolution_slots = _broadcast_resolution_slots(
        state,
        caller_key=caller_key,
        caller_slot=caller_slot,
        ownership_fenced=ownership_fenced,
    )

    explicit = targets is not None
    if explicit:
        # THE CAP, CHECKED BEFORE ANYTHING RESOLVES. Below the resolution loop this
        # check is correct and useless: reaching it costs one `_resolve_slot` per
        # name -- a copy of the authorized-slot set and two full passes over it,
        # with nothing awaited between them -- so an oversized list does the work
        # the refusal exists to prevent and is refused afterwards. Here the cost of
        # an oversized list is one integer comparison.
        #
        # Counted on the caller's RAW list rather than on the deduplicated audience,
        # so the bound is on what the caller submitted. A list of `cap + 1`
        # duplicate spellings of one session is therefore refused even though it
        # would collapse under the cap, and that is the intended reading: the cap
        # bounds the work this call asks for, and deduplication is that work.
        #
        # The route bounds this too, and both are deliberate: the route is the
        # boundary an in-sandbox shell reaches, this is the boundary every
        # in-process caller reaches.
        if len(targets or []) > MAX_BROADCAST_TARGETS:
            raise _too_many_targets(len(targets or []))
        # De-duplicated by an AUTHORIZED resolved slot while PRESERVING the
        # caller's order, so two spellings of one session still deliver once.
        # Resolution is restricted to slots the caller may address; a private
        # title therefore remains the caller's own string and cannot turn into a
        # slot key in the refusal row. This loses no real target because
        # ``authorize_target`` would refuse every slot outside this set anyway,
        # and every retained key still goes through that gate at delivery.
        # A name that cannot be resolved, or is ambiguous within the authorized
        # set, remains in the audience so the gate can return its refusal row.
        seen: set[tuple[str, str]] = set()
        audience: list[str] = []
        for raw in targets or []:
            name = str(raw).strip()
            if not name:
                continue
            try:
                slot = _resolve_slot(state, name, candidates=resolution_slots)
            except SessionControlError:
                slot = None
            identity = ("slot", slot.key) if slot is not None else ("unresolved", name)
            if identity not in seen:
                seen.add(identity)
                audience.append(slot.key if slot is not None else name)
        if not audience:
            # The caller named targets and not one of them is usable. Both silent
            # answers are wrong here: falling back to the default audience turns a
            # malformed argument into an accidental fleet-wide send, and reporting
            # "delivered 0 of 0" hides a call that never had a chance of working.
            raise SessionControlError(
                "targets was given but names no session; omit it to reach every "
                "session you created",
                code="target_required",
                status=400,
            )
    else:
        audience = broadcast_audience(state, caller_key)

    if len(audience) > MAX_BROADCAST_TARGETS:
        # The DEFAULT audience's own bound. An explicit list was already refused
        # above, before resolution, and deduplication only ever shrinks it -- so
        # this now guards the fence-built audience, whose size the caller did not
        # choose. It should be unreachable: ``broadcast_audience`` returns the live
        # slots this caller created, which ``MAX_SLOTS_PER_CREATOR`` bounds at or
        # below this cap (see ``MAX_BROADCAST_TARGETS``). Kept because "unreachable
        # because two constants are ordered correctly" is a property a later edit
        # can break, and truncating a fleet-wide send is the failure that must not
        # be the fallback.
        raise _too_many_targets(len(audience))

    results: list[dict[str, Any]] = []
    timeout_prompt = _SEND_PROVENANCE.format(
        caller=caller_key or "unknown", via=BROADCAST_VIA
    ) + sanitize_outbound(body)
    for name in audience:
        try:
            timeout_slot = _resolve_slot(state, name)
        except SessionControlError:
            timeout_slot = None
        delivery_progress = _DeliveryProgress()
        try:
            # Each delivery gets its OWN bound. Without one a single unresponsive
            # target starves every target behind it: the steer arm suspends on
            # `client.steer` -> `stdin.drain()` with no `wait_for` anywhere beneath
            # it, so a session whose process is wedged blocks this loop
            # indefinitely and targets 3-8 of a "stop, the issue was already fixed"
            # never hear it. The client's request budget then expires and the
            # caller is told the whole broadcast failed, discarding the per-target
            # report that is the only thing that could have said which half landed.
            # Bounding here and continuing turns that into one refused row.
            #
            # The bound wraps the WHOLE call rather than the steer RPC alone, which
            # is what gives the queue arm the same protection: that arm also
            # suspends before it delivers (`asyncio.to_thread(sel)`,
            # `prewarm_enabled_check`, and the config load that can sit behind it),
            # and a queue broadcast stalled on a config read starves the fleet just
            # as thoroughly as a wedged steer does.
            #
            # The bound wraps two pre-authorization awaits as well as the
            # delivery arms. A cancel at `asyncio.to_thread(sel)` or
            # `prewarm_enabled_check()` happens before `authorize_target`, so it
            # cannot leave this text pending or queued. A cancel at the steer RPC
            # can leave the exact prompt in `_pending_steers`, and turn teardown can
            # move it into a queue entry's `content` while cancellation unwinds.
            #
            # Capture the resolved slot and a delivery-owned progress marker before
            # the call. After cancellation, retained text still proves it may run,
            # but empty target containers prove nothing: consumption and a send that
            # never started both empty them. The progress marker distinguishes a
            # pre-authorization cancellation from the ambiguous steer await and every
            # post-return await. If the object disappeared or was replaced, the
            # outcome is unknown regardless.
            #
            # `send_to_target` records its `cancelled` audit only after
            # authorization. Cancellation at either earlier await therefore leaves
            # no target-level permission row, while cancellation in an authorized
            # delivery keeps that row unchanged.
            sent = await asyncio.wait_for(
                send_to_target(
                    state,
                    caller_session_key=caller_session_key,
                    target=name,
                    message=body,
                    steer=(mode == "steer"),
                    caller_fenced=caller_fenced,
                    via=BROADCAST_VIA,
                    _delivery_progress=delivery_progress,
                ),
                timeout=BROADCAST_TARGET_ALLOWANCE_SECS,
            )
        except asyncio.TimeoutError:
            # A timeout says only that this side stopped waiting. Delivery-owned
            # progress decides whether retry advice is safe; target containers only
            # add the stronger fact that an exact retained copy may still run.
            observation = _timeout_delivery_observation(
                state, timeout_slot, timeout_prompt, delivery_progress
            )
            if observation == _TIMEOUT_RETAINED:
                error = (
                    f"this target did not finish its delivery within "
                    f"{BROADCAST_TARGET_ALLOWANCE_SECS:.3g}s, so the broadcast "
                    "stopped waiting and moved on; whether the message has run is "
                    "UNKNOWN. The same text is pending or queued on that target and "
                    "may still run. Do NOT re-send it: a duplicate could run too"
                )
            elif observation == _TIMEOUT_NOT_HANDED_OVER:
                error = (
                    f"this target did not finish its delivery within "
                    f"{BROADCAST_TARGET_ALLOWANCE_SECS:.3g}s, so the broadcast "
                    "stopped waiting and moved on; the target has neither a pending "
                    "steer nor a queued copy, so the delivery did not reach the "
                    "hand-over. Re-sending the same text is safe"
                )
            else:
                if timeout_slot is not None and state._slots.get(timeout_slot.key) is timeout_slot:
                    error = (
                        f"this target did not finish its delivery within "
                        f"{BROADCAST_TARGET_ALLOWANCE_SECS:.3g}s, so the broadcast "
                        "stopped waiting and moved on; cancellation reached the steer "
                        "await or a later await, so whether the message has run is "
                        "UNKNOWN. Re-sending could execute it twice"
                    )
                else:
                    error = (
                        f"this target did not finish its delivery within "
                        f"{BROADCAST_TARGET_ALLOWANCE_SECS:.3g}s, so the broadcast "
                        "stopped waiting and moved on; the target is unavailable, so "
                        "whether the message has run or is pending or queued is UNKNOWN"
                    )
            logger.warning(
                "session_broadcast: delivery to %s exceeded %.3gs and was "
                "cancelled mid-flight; post-cancel observation=%s",
                name,
                BROADCAST_TARGET_ALLOWANCE_SECS,
                observation,
            )
            results.append(
                {
                    "target": name,
                    "ok": False,
                    "code": BROADCAST_TIMEOUT_CODE,
                    "error": error,
                }
            )
            continue
        except SessionControlError as exc:
            # Collected, not raised. This target is out of reach; the rest are not,
            # and the caller needs to know which is which. The refusal was already
            # audited by the gate that produced it.
            results.append(
                {
                    "target": name,
                    "ok": False,
                    "code": exc.code,
                    "error": exc.message,
                }
            )
            continue
        except Exception:
            # An unexpected failure on ONE delivery must not take the broadcast
            # down with it, and it must not be reported as a refusal either: a
            # refusal has a code the caller can act on, this has none. Logged with
            # its traceback and reported as the internal failure it is.
            logger.exception("session_broadcast: delivery to %s failed", name)
            results.append(
                {
                    "target": name,
                    "ok": False,
                    "code": "delivery_failed",
                    "error": "delivery failed unexpectedly; see the gateway log",
                }
            )
            continue
        results.append(
            {
                "target": sent.get("target", name),
                "ok": True,
                "started": bool(sent.get("started")),
                "steered": bool(sent.get("steered")),
            }
        )

    delivered = sum(1 for row in results if row.get("ok"))
    _audit(
        caller_session_key=caller_session_key,
        operation="broadcast",
        slot_key=caller_key,
        outcome="allowed",
        detail={
            "mode": mode,
            # WHERE the audience came from, because the two are different acts: an
            # explicit list is a caller naming sessions, the default is the fence's
            # own set. A trail that cannot tell them apart cannot answer "who chose
            # these targets".
            "audience": "explicit" if explicit else "created",
            "requested": len(audience),
            "delivered": delivered,
            "chars": len(body),
        },
    )
    # The entry gate RAN, but its verdict does not survive this verb's suspensions,
    # and every `results` row below names one of the caller's own sessions. A channel
    # mirror can be bound onto an already-open dashboard session while a delivery is
    # awaiting -- the channel picker (`messaging/session_resume.py`) and the Slack
    # link route (`slack/interactions.py`) both do it with no idle-slot requirement --
    # so a payload built under the entry gate would reach that channel's audience
    # carrying private session keys. Same re-check, same audited path, as
    # `created_session_status` does after its scan; workspace is compared for the
    # same reason (it IS reassigned on a live slot, while a live key is never rebound
    # to a new object, so a slot-identity check could not fire).
    #
    # Raised AFTER the allowed audit above, deliberately: the deliveries were each
    # individually gated and have already happened, so the trail must record them.
    # What is refused is the report, which is why the message says so -- a caller
    # that reads this as "the broadcast failed" and sends again delivers twice.
    refuse_caller_surface(state, caller_key=caller_key, deny=deny)
    if str(getattr(caller_slot, "workspace", "default")) != caller_workspace:
        raise deny(
            "the calling session moved workspace while this broadcast was in flight, "
            "so the per-target report is withheld. The deliveries already happened -- "
            "do NOT send the same text again; read the targets' own transcripts",
            "caller_changed_mid_broadcast",
        )
    return {
        "ok": True,
        "mode": mode,
        "requested": len(audience),
        "delivered": delivered,
        # True only for the DEFAULT audience: the caller has created nothing that
        # is still open, which is a real state and not an error. An explicit list
        # that names nothing is refused before delivery, so it cannot reach here.
        **({"audience_empty": True} if not audience and not explicit else {}),
        "results": results,
    }


def _created_tree_roster(caller_key: str) -> "tuple[list[str], str]":
    """Slots the crew log says hang under *caller_key*, and how good that read was.

    The DURABLE half of :func:`created_session_status`. ``state._slots`` knows only
    what is open right now, so a worker that was closed, or whose session was lost
    to a crash, disappears from it entirely — and "I dispatched eight and can see
    six" is exactly the question this verb exists to answer. The crew log's session
    tree is written when a session is opened and survives both, so it is what
    supplies the roster while the live slots supply the status.

    The second value is the read's own quality, and it is returned rather than
    folded into the list because the three answers are not interchangeable:

    * ``readable`` — the fold saw every unit the store holds.
    * ``incomplete`` — a unit's bytes could not be read, or the population was
      capped, so rows may be MISSING. The fold is still served: this reader
      DISPLAYS lineage rather than deciding on it, which is the case
      :class:`~kiro_crew.crew_log.session_tree.TreeReading` documents as free to
      ignore the flag — but a caller counting its workers must be told the count
      is a floor.
    * ``unreadable`` — the crew log is off, or the projection is not seeded for the
      store configured now. There is no roster at all and the answer is live-only.

    Reads the in-memory fold and does NO I/O, so it is safe on the event loop. An
    unseeded projection answers ``unreadable`` rather than seeding itself here,
    for the reason :func:`_slot_tree_parent` gives: a seed is a disk read.

    The parent edge it follows is the CURRENT one, so an adopted session is listed
    under whoever holds it now rather than under whoever opened it. That is the
    right answer for this verb — it reports the sessions the caller is answerable
    for — and it cannot widen the caller's reach, because an adoption is itself
    gated by :func:`authorize_target`.
    """
    try:
        if not crew_log_emit.enabled():
            return [], "unreadable"
        proj = projection()
        if not proj.seeded_for_current_store:
            return [], "unreadable"
        reading = proj.reading()
    except Exception:
        logger.debug("session tree roster could not be read", exc_info=True)
        return [], "unreadable"
    children: list[str] = []
    overflow = False
    for slot, node in reading.nodes.items():
        if getattr(node, "parent_slot", None) != caller_key or slot == caller_key:
            continue
        if len(children) < MAX_SESSION_STATUS_ROWS:
            children.append(slot)
        else:
            overflow = True
    children.sort()
    return children, ("incomplete" if reading.incomplete or overflow else "readable")


def _gone_slot_lifecycles(slots: "Sequence[str]") -> dict[str, dict[str, Any]]:
    """Fold each gone slot's lifecycle so a `gone` row can say CLOSED versus LOST.

    The crew log already records a ``session/closed`` edge and the status fold
    already projects it as ``lifecycle``/``closed_at`` (see
    :func:`~kiro_crew.crew_log.projection._status_render`). This reads that
    projection per slot and nothing more -- no new terminal vocabulary, no new
    verb -- so a worker whose tab was closed after it finished is told apart from
    one lost with its process.

    The read is slot-keyed (:func:`~kiro_crew.crew_log.projection.read_slot_projection`),
    so it folds over EVERY unit the slot ran under, newest last, and reports the
    lifecycle of the slot's current session. A slot that was reopened and later
    closed again reads ``closed``; the ``session/opened`` of the reopen resets the
    close the earlier life recorded, which the status fold already handles.

    Returns a map ``{slot: {"status", "closed_at"?}}``. A slot is ``closed`` (and
    carries its ``closed_at``) only when its fold's newest life ended by the
    DELIBERATE tab close -- ``close_reason`` is ``removed`` and no turn is still
    open (see :func:`_lifecycle_is_closed`). Everything else is ``lost``: an
    ``open``/``unknown`` lifecycle, a read that raised, OR a ``closed`` lifecycle
    whose edge came from a process recycle (``reset``: idle expiry, RSS recycle,
    model/provider switch, the watchdogs) or a ``destroy`` (including mid-turn) --
    teardowns the tab and conversation outlive. "Lost" is the fail-safe reading: it
    is the state a patrol re-dispatches, and reporting a worker lost when it may
    have finished is cheaper than reading a recycled or mid-turn worker as
    ``closed`` and telling a conductor to stop looking for it.

    Touches disk (one slot-keyed fold per slot), so it is called off the event loop.
    """
    from kiro_crew.crew_log.projection import read_slot_projection

    out: dict[str, dict[str, Any]] = {}
    for slot in slots:
        try:
            value = read_slot_projection(slot, "status").value
        except Exception:
            # A read fault is not evidence the worker closed cleanly, so it reads as
            # LOST -- the fail-safe answer, the one a patrol acts on by re-dispatching.
            logger.debug("gone row lifecycle could not be folded for %s", slot, exc_info=True)
            out[slot] = {"status": "lost"}
            continue
        if _lifecycle_is_closed(value):
            out[slot] = {"status": "closed", "closed_at": value.get("closed_at")}
        else:
            out[slot] = {"status": "lost"}
    return out


#: Close reasons that mean "the worker FINISHED and its tab was closed" -- the one
#: direction a gone row may call ``closed``. Only the deliberate tab ✕ / ``session_close``
#: teardown (``remove`` -> ``END_REASON_REMOVED``) qualifies. Every OTHER close edge is a
#: process recycle or a mid-turn teardown that the tab and conversation survive --
#: ``reset`` (idle expiry, RSS recycle, model/provider switch, the watchdogs),
#: ``destroyed``/``destroyed_sid_retained`` (``destroy``, including mid-turn) -- and must
#: read ``lost``, the fail-safe state a patrol re-dispatches. Reading one of those as
#: ``closed`` is the fail-OPEN harm this split exists to remove: it would tell a conductor
#: a lost worker finished, so it stops looking for it.
_FINISHED_CLOSE_REASONS = frozenset({END_REASON_REMOVED})


def _lifecycle_is_closed(value: "Mapping[str, Any]") -> bool:
    """Whether a folded slot status means the worker finished and closed its tab.

    ``True`` only when the fold's newest life is closed by the DELIBERATE end
    (``close_reason`` in :data:`_FINISHED_CLOSE_REASONS`), no turn is still open,
    AND the newest completed turn ended cleanly. A ``closed`` lifecycle alone is
    not enough: the ``session/closed`` edge is also written on every process
    recycle (``reset``) and on ``destroy`` (including mid-turn), teardowns the tab
    and conversation outlive.

    The ``last_stop_reason`` guard is the third fail-safe, and it is the one a
    deliberate MID-TURN close needs: closing a tab with a turn in flight cancels
    ``slot.task``, and the cancelled turn's ``finally`` writes a real
    ``turn/completed`` (``stop_reason`` ``cancelled``/``failed``/``error:*``) that
    CLEARS ``turn_open`` -- so the ``turn_open`` guard above does not catch it, and
    a worker killed mid-turn would read ``closed`` ("finished, nothing to
    re-dispatch") when its work did not finish. A newest turn that did not end
    cleanly (:func:`turn_outcome` other than ``"ok"``) therefore reads ``lost``.
    A clean ``end_turn`` and the ``None`` of a tab closed before any turn ran both
    classify ``"ok"`` -- a session that genuinely finished (or never started work)
    stays ``closed``.

    A close whose reason is a recycle, a close with a turn still open, a newest
    turn that did not end cleanly, or any non-closed lifecycle all read as
    not-closed -> the caller reports ``lost``, the fail-safe direction.
    """
    if value.get("lifecycle") != "closed":
        return False
    if value.get("turn_open"):
        # A session cut off with a turn in flight did not finish, however it was
        # torn down -- the projection leaves the open turn set on a close for
        # exactly this reader to see (crew_log/projection.py::_status_render).
        return False
    if turn_outcome(value.get("last_stop_reason")) != "ok":
        # The newest completed turn did not end cleanly. A deliberate mid-turn
        # close cancels the running turn, whose ``finally`` records a
        # ``turn/completed`` with an abnormal ``stop_reason`` and clears
        # ``turn_open`` -- so this is the only guard that still sees the work was
        # cut off. ``None``/``end_turn`` are both ``"ok"``, so a clean finish and a
        # never-worked tab close stay ``closed``.
        return False
    return value.get("close_reason") in _FINISHED_CLOSE_REASONS


def _bounded_status_title(value: object) -> str:
    """A display-safe title bounded before it is retained in a status row."""
    return sanitize_outbound(str(value or ""))[:MAX_SESSION_STATUS_TITLE_CHARS]


def _pending_approval_fields(state: "DashboardState", slot: Any) -> dict[str, Any]:
    """``{"pending_approval": True, "pending_approval_tool": ...}`` while *slot* waits.

    A turn parked on a tool approval keeps ``running`` true, so without this a
    creator polling the session sees ``working`` for the whole approval window.
    The answer is the slot projection's own ``pending_approval`` and
    ``pending_approval_info`` -- the fields the dashboard card reads -- so the
    two cannot disagree. ``to_dict`` walks the transcript, so it runs only once
    one of the two registries it consults has an open entry; an ordinary turn
    pays one dict scan and the response shape is unchanged.
    """
    futures = getattr(slot, "_approval_futures", None) or {}
    pending_for = getattr(state, "pending_coordinator_approvals", None)
    if not any(not fut.done() for fut in futures.values()) and not (
        callable(pending_for) and pending_for(slot.key)
    ):
        return {}
    projected = slot.to_dict()
    if not projected.get("pending_approval"):
        return {}
    fields: dict[str, Any] = {"pending_approval": True}
    tool = str((projected.get("pending_approval_info") or {}).get("tool") or "")
    if tool:
        fields["pending_approval_tool"] = redact_and_truncate(
            sanitize_outbound(tool), MAX_PENDING_APPROVAL_TOOL_CHARS
        )
    return fields


def _created_history_roster(
    state: "DashboardState", caller_key: str, caller_workspace: str
) -> tuple[dict[str, dict[str, Any]], str, int]:
    """Persisted birth metadata for sessions created by *caller_key*.

    This is the transcript-meta half of :func:`created_session_status`. It is a
    weaker source than the fenced crew log: ``created_by`` is already trusted by
    the member ownership boundary, but it is not gateway-authored lineage. Rows
    retain that distinction through their ``source`` value.

    The scan is read-only. ``list_sessions`` supplies the catalog and
    ``get_metadata_status`` supplies both the creator edge and a readability
    verdict for each candidate. A bad metadata row or a matching population past
    :data:`MAX_SESSION_STATUS_ROWS` makes the result ``incomplete`` rather than
    disappearing behind a claim of completeness. Only the bounded title needed
    by the response is retained; the full editable metadata dict is not.
    """
    log = state.conversation_log
    if log is None:
        return {}, "unreadable", 0
    try:
        sessions = log.list_sessions()
    except Exception:
        logger.debug("session history roster could not be listed", exc_info=True)
        return {}, "unreadable", 0

    rows: dict[str, dict[str, Any]] = {}
    incomplete = False
    omitted = 0
    for session in sessions:
        history_key = str(session.get("key", ""))
        slot_key = _recent_session_slot_name(history_key)
        if slot_key is None or slot_key == caller_key:
            continue
        try:
            meta, readable = log.get_metadata_status(history_key)
        except Exception:
            logger.debug("session history metadata could not be read", exc_info=True)
            incomplete = True
            continue
        if not readable:
            incomplete = True
            continue
        if str(meta.get("created_by", "")) != caller_key:
            continue
        if str(meta.get("workspace", "default")) != caller_workspace:
            continue
        if len(rows) >= MAX_SESSION_STATUS_ROWS:
            # Keep scanning only to report the exact overflow count. No metadata
            # or title beyond the cap is retained in this request.
            omitted += 1
            incomplete = True
            continue
        rows[slot_key] = {
            "title": _bounded_status_title(meta.get("title") or slot_key),
        }
    return rows, ("incomplete" if incomplete else "readable"), omitted


async def created_session_status(
    state: "DashboardState",
    *,
    caller_session_key: str,
    caller_fenced: bool | None = None,
) -> dict[str, Any]:
    """Every session this caller stood up, and what each one is doing now.

    The roster and the status come from THREE sources because none can answer
    every half. The crew log's session tree is durable and gateway-attested, while
    persisted history metadata records a birth before the first turn can write a
    tree edge. Live slots know exactly what is running but forget a session the
    moment it is gone. The union covers all three windows, and each row says which
    source placed it.

    ``status`` per row, and the five are distinct answers a patrol acts on
    differently:

    * ``working`` — a turn is in flight. Wait.
    * ``queued`` — idle, but messages are waiting to run. Also wait, but nothing is
      happening yet, so a steer would land on nothing.
    * ``idle`` — open and doing nothing. This is the one that needs a decision.
    * ``closed`` — the crew log names it, the dashboard does not hold it, and the
      log records a ``session/closed`` edge: a worker that FINISHED and closed its
      tab. Carries ``closed_at``. Ignore it; there is nothing here to message, and
      :func:`authorize_target` would answer ``target_not_found``.
    * ``lost`` — the crew log names it, the dashboard does not hold it, and no
      qualifying clean deliberate close was recorded: either no ``session/closed``
      edge at all, or the newest one is a ``reset``/``destroy`` teardown or a close
      whose last turn did not end cleanly — none of which means "finished, nothing
      to re-dispatch". Lost with the process that ran it (or its opener was retained
      out of the log). Re-dispatch it. ``closed`` and ``lost`` are the two fates the
      old single ``gone`` status could not tell apart, and they call for opposite
      actions.
    * ``unknown`` — history records the creator but neither a live slot nor an
      attested tree edge exists. The row proves the session was created without
      claiming whether it finished or was lost.

    READ-ONLY. The history catalog and metadata reads touch disk; callers keep the
    operation informational and expose their quality separately from the crew-log
    tree quality.

    The union is bounded at :data:`MAX_SESSION_STATUS_ROWS`, and that cut gets its
    OWN field: ``roster_omitted`` counts the rows dropped, zero when the whole
    union was retained. It is deliberately not expressed by degrading ``history``,
    which describes the transcript scan alone — a union cut is not a read fault in
    any source, and spelling it that way blames a read that completed while
    ``history_omitted`` still reports nothing cut.

    The ownership fence applies to the rows, not just to the verb: a fenced caller
    (a crew member, a cron, an agent-created session) sees the live sessions it
    created and nothing else, so this verb cannot become a way to enumerate the
    user's own sessions by their titles. A ``closed`` or ``lost`` row carries only
    a slot key the caller already knew, and no title, so it is listed either way.
    """
    deny = _deny_factory(caller_session_key=caller_session_key, operation="status", target="")
    caller_key = refuse_caller_identity(state, caller_session_key=caller_session_key, deny=deny)
    caller_slot = refuse_caller_surface(state, caller_key=caller_key, deny=deny)
    ownership_fenced = (
        _caller_is_ownership_fenced(state, caller_key) if caller_fenced is None else caller_fenced
    )
    tree_children, tree_state = _created_tree_roster(caller_key)

    # The first suspension in this verb, deliberately AFTER the complete
    # synchronous caller gate and fence resolution above. The route prewarms the
    # config cache immediately before entering this coroutine, so moving an await
    # into that window would let an intervening config edit put blocking config IO
    # back on the loop. The scan itself globs, stats and reads transcript metadata,
    # so run the complete operation in one worker-thread hop.
    caller_workspace = str(getattr(caller_slot, "workspace", "default"))
    history_children, history_state, history_omitted = await asyncio.to_thread(
        _created_history_roster, state, caller_key, caller_workspace
    )
    # The gate above RAN before the suspension, but its verdict does not survive
    # one: a channel mirror can be bound onto an already-open dashboard session
    # while this scan is on a worker thread -- the channel picker
    # (`messaging/session_resume.py`) and the Slack link route
    # (`slack/interactions.py`) both do it with no idle-slot requirement -- and
    # the rows below carry other sessions' TITLES, the names of the user's
    # private work. Published past a gate that passed, they reach that channel's
    # audience. So re-call the same gate on the same terms and refuse through the
    # same audited path, the way the queue drain re-checks mirror identity
    # between admission and delivery (`_probe_channel_mirror`) rather than
    # trusting its admission.
    #
    # Slot IDENTITY is deliberately NOT compared: `get_or_create_slot` returns the
    # registered slot for an existing key and the single `put_slot` publish runs
    # only where the key was absent, so no live key is rebound to a new object --
    # a check for it could never fire. WORKSPACE is compared, because that one is
    # reassigned on a live slot (`chat_handlers` commits one, `channel_slots`
    # adopts one from metadata) and it is the boundary every row below is filtered
    # on: a workspace that moved mid-scan would filter rows against a boundary the
    # history scan did not use.
    refuse_caller_surface(state, caller_key=caller_key, deny=deny)
    if str(getattr(caller_slot, "workspace", "default")) != caller_workspace:
        raise deny(
            "the calling session moved workspace while its roster was being read; " "call again",
            "caller_changed_mid_read",
        )

    # Live slot state stays on the event loop and is read only after the history
    # worker returns. Reading it inside the worker would let the result go stale
    # before these rows are built. This read names the sessions the caller HOLDS --
    # ``manager_of``, so a handed-over session is listed under the manager that can
    # reach it rather than the giver that cannot -- and not workspace, so it is not
    # the workspace-boundary read; that is `resolvable_keys`, built below AFTER the
    # last suspension in this verb. It can refuse (409) when a slot's manager has not
    # been read yet, because the membership test reads that value: a roster built from
    # an unread field names the wrong children, and a caller cannot tell that roster
    # from a correct one.
    live_children = broadcast_audience(state, caller_key)

    rows: list[dict[str, Any]] = []
    roster = set(tree_children) | set(history_children) | set(live_children)
    # The UNION cut is its own fact, with its own field and its own count. It must
    # not be signalled by degrading ``history_state``, which describes the
    # transcript scan alone: a union cut is not a read fault in any source, so that
    # spelling says two false things at once -- it blames a transcript read that
    # completed, and it leaves ``history_omitted`` reporting nothing cut while rows
    # are dropped. Neither is actionable, because re-reading history cannot recover
    # a row the union bound cut.
    #
    # Zero when nothing was cut, so a reader distinguishes "no overflow" from a
    # cut whose size it must know to judge the roster it was handed.
    roster_omitted = max(0, len(roster) - MAX_SESSION_STATUS_ROWS)
    retained = sorted(roster)[:MAX_SESSION_STATUS_ROWS]
    # A `gone` row -- the tree names it and the dashboard does not hold it -- splits
    # into `closed` and `lost`, and the two call for OPPOSITE actions: ignore a
    # worker that finished and closed its tab, re-dispatch one lost with its
    # process. The split is read from the ``session/closed`` edge the status fold
    # already projects, so it is a disk fold per gone slot and runs OFF the loop,
    # the same worker-thread discipline the history scan above keeps. Scoped to the
    # retained, tree-attested, now-absent slots so no fold is paid for a row that
    # will not carry the distinction.
    gone_slots = [key for key in retained if key in tree_children and state.get_slot(key) is None]
    gone_lifecycles = (
        await asyncio.to_thread(_gone_slot_lifecycles, gone_slots) if gone_slots else {}
    )
    # The gone fold above is the LAST suspension in this verb, and the live-row
    # containment must be computed after it, not before. A live worker can be moved
    # across the workspace boundary mid-fold -- `api_chat_slot_workspace` reassigns
    # `_slot.workspace` on a running slot with no idle requirement -- so a
    # `resolvable_keys` built before this await would admit a worker on its
    # pre-move workspace and then carry its `display_title` out of the caller's
    # workspace in the row loop below. So re-call the caller gate on the same terms
    # and REBUILD containment here, after the final await, exactly as the block
    # after the history scan does: the boundary every row is filtered on is read
    # from the live slots as they stand once nothing else can suspend.
    refuse_caller_surface(state, caller_key=caller_key, deny=deny)
    if str(getattr(caller_slot, "workspace", "default")) != caller_workspace:
        raise deny(
            "the calling session moved workspace while its roster was being read; " "call again",
            "caller_changed_mid_read",
        )
    # The SAME containment set `session_broadcast` resolves names against, resolved
    # here once and applied to every live row below. Built AFTER the gone fold's
    # await (and its re-check), so a worker moved across the boundary during that
    # fold is already excluded -- its `workspace` is re-read here as it now stands,
    # and the fail-safe follows: a row the moved worker would carry is dropped by
    # the `resolvable_keys` membership test below rather than leaked.
    resolvable_keys = {
        slot.key
        for slot in _broadcast_resolution_slots(
            state,
            caller_key=caller_key,
            caller_slot=caller_slot,
            ownership_fenced=ownership_fenced,
        )
    }
    for key in retained:
        slot = state.get_slot(key)
        from_tree = key in tree_children
        from_history = key in history_children
        sources = [
            source
            for source, present in (
                ("crew_log", from_tree),
                ("history", from_history),
                ("live", slot is not None),
            )
            if present
        ]
        if slot is None:
            if from_tree:
                # The attested tree still owns this status. History may corroborate
                # the birth, but it does not turn editable metadata into lineage.
                #
                # CLOSED vs LOST, read from the ``session/closed`` edge the status
                # fold projects: a worker that closed its tab after finishing is
                # ``closed`` and carries its ``closed_at``; one gone without a
                # recorded close -- lost with its process, or whose opener was
                # retained out of the log -- is ``lost``. Defaults to ``lost`` when
                # the fold was not made (a slot absent from the map), the fail-safe
                # reading a patrol re-dispatches rather than drops.
                lifecycle = gone_lifecycles.get(key, {"status": "lost"})
                gone_row: dict[str, Any] = {
                    "target": key,
                    "status": lifecycle["status"],
                    "source": "+".join(sources),
                }
                if lifecycle["status"] == "closed":
                    gone_row["closed_at"] = lifecycle.get("closed_at")
                rows.append(gone_row)
                continue
            if not from_history:
                # Neither tree nor history, and no live slot. A row reaches here
                # only because it was in ``retained`` -- read from the creator's
                # live children BEFORE the gone-fold await -- and its slot has since
                # gone: a worker the caller created that never took a first turn (so
                # no tree edge) and left no history row (the history scan filters on
                # workspace) can have its tab closed WHILE the fold runs on its
                # worker thread, landing here with ``slot is None`` and both durable
                # sources absent. Reading ``history_children[key]`` would raise
                # ``KeyError`` and fail the whole roster. Drop the row: the worker is
                # gone with no durable record to report, which is the fail-safe
                # reading -- the same "nothing to say about a slot that vanished with
                # no attested trace" the ``lost`` default elsewhere takes.
                continue
            meta = history_children[key]
            rows.append(
                {
                    "target": key,
                    "title": meta["title"],
                    "status": "unknown",
                    "source": "history",
                }
            )
            continue
        if slot.key not in resolvable_keys:
            # ONE predicate for both verbs. These rows carry `display_title`, which
            # for a channel-linked or mirrored session is derived from a conversation
            # other people are in, and a creator whose worker is linked LATER would
            # otherwise be handed that title on its next roster read. The row is
            # dropped rather than reported title-less for the reason the workspace
            # clause inside the predicate gives: `session_send` refuses this target
            # too (`linked_session_target`, `mirrored_target`, `ephemeral_target`,
            # `app_scoped_target`, `not_creator`), so a row here would be a listing
            # of work the caller cannot message.
            #
            # Membership in `_broadcast_resolution_slots` rather than a second copy
            # of its clauses, because one containment enforced in two places is one
            # containment that can differ between them: a clause restated here is a
            # clause that has to be restated again on every change to the predicate.
            continue
        queue_depth = len(slot._queue)
        running = bool(slot.running)
        rows.append(
            {
                "target": key,
                "title": _bounded_status_title(slot.display_title),
                "status": "working" if running else ("queued" if queue_depth else "idle"),
                "running": running,
                "queue_depth": queue_depth,
                **_pending_approval_fields(state, slot),
                "source": "+".join(sources),
            }
        )

    _audit(
        caller_session_key=caller_session_key,
        operation="status",
        slot_key=caller_key,
        outcome="allowed",
        detail={
            "rows": len(rows),
            "tree": tree_state,
            "history": history_state,
            "history_omitted": history_omitted,
            "roster_omitted": roster_omitted,
        },
    )
    return {
        "ok": True,
        "caller": caller_key,
        # How much of the durable roster this answer rests on. A caller that reads
        # only `sessions` and gets a short list cannot tell "you created three" from
        # "I could read three", and for a patrol deciding whether a worker was lost
        # those are opposite conclusions.
        "tree": tree_state,
        # History metadata is a separate, weaker roster source. Its quality must
        # not be folded into `tree`, whose value describes only the crew-log fold.
        "history": history_state,
        # Exact count of matching transcript rows not retained due to the bound.
        # Kept separate from ``history`` so callers can distinguish a read fault
        # from deliberate overflow while old clients still see ``incomplete``.
        "history_omitted": history_omitted,
        # Exact count of rows the UNION of the three sources lost to
        # ``MAX_SESSION_STATUS_ROWS``. Separate from ``history_omitted`` because it
        # is a different cut with a different remedy: this one is not a read fault
        # in any source, and no source's quality field describes it. Zero when the
        # whole union was retained.
        "roster_omitted": roster_omitted,
        "sessions": rows,
    }


def read_messages(
    state: "DashboardState",
    *,
    caller_session_key: str,
    target: str,
    limit: int = DEFAULT_READ_MESSAGES,
    since: int | None = None,
    caller_fenced: bool | None = None,
) -> dict[str, Any]:
    """Read *target*'s transcript tail plus enough state to poll it.

    ``next_since`` is the cursor to poll with; passing it back as ``since`` on the
    next call returns only what arrived in between, which is the whole
    wait → read poll loop. ``running`` says whether the target is still
    working, so a caller knows the difference between "nothing new yet" and
    "finished and idle".
    """
    if limit < 1 or limit > MAX_READ_MESSAGES:
        raise SessionControlError(
            f"limit must be between 1 and {MAX_READ_MESSAGES}", code="invalid_limit"
        )
    slot = authorize_target(
        state,
        caller_session_key=caller_session_key,
        target=target,
        operation="read",
        precomputed_ownership_fenced=caller_fenced,
    )
    # The reader's own audience was recorded by the caller-side gate inside
    # ``authorize_target`` (:func:`record_audience_admission`), so a mirror gained
    # or retargeted between this read and the reply it feeds withholds the
    # reader's cross-surface publication. No per-verb stamp here: the gate is the
    # one point every verb passes, which is what keeps a read path from missing it.

    # Indexes are ABSOLUTE positions in the session, not offsets into the live
    # window. A slot keeps only the most recent ``_MAX_SLOT_MESSAGES`` in memory
    # and credits each trimmed row to a frozen-prefix counter, so window length
    # stops growing once trimming starts. A cursor derived from that length
    # would freeze at the cap and never see another reply; adding the
    # frozen-prefix count makes it monotonic for the session's whole life.
    raw_window = list(slot.messages)
    # The DURABLE frozen-prefix counter, never ``_disk_older_count``: that one
    # counts every trimmed row, transient ones included, while the positions
    # below are built over the durable rows the filter keeps. Basing on the
    # all-rows counter shifted every position as soon as a transient row was
    # trimmed, and a ``since`` read then served a durable message the caller
    # already had. ``_disk_older_durable_count`` counts exactly the rows the
    # ``TRANSIENT_ROLES`` filter below would have kept, so the two spaces agree
    # for the session's whole life. Defensive ``getattr`` matches how the
    # existing code reads ``_disk_older_count``: a slot restored by an older
    # build simply has no trimmed prefix yet.
    base = int(getattr(slot, "_disk_older_durable_count", 0) or 0)
    # Stop the cursor before the streaming tail (see ``TRANSIENT_ROLES``): those
    # rows are deleted when the segment flushes, so a cursor past them would sit
    # beyond the list that replaces them and never return the finished reply.
    messages = [m for m in raw_window if m.get("role") not in TRANSIENT_ROLES]
    durable_end = len(messages)
    total = base + durable_end
    if since is not None:
        if since < 0:
            raise SessionControlError("since must be >= 0", code="invalid_since")
        if since < base:
            # The cursor points into the trimmed prefix: those rows exist only
            # on disk now, and this read serves the in-memory window. Starting
            # at ``base`` instead would silently skip every row in
            # ``[since, base)`` — a poller that lagged a whole window behind
            # would lose messages with nothing in the response saying so. The
            # refusal is loud and the tail-read fallback recovers, exactly like
            # the past-the-end case below.
            raise SessionControlError(
                "this session is long enough that the messages at your cursor "
                "have been trimmed from memory — read without `since` to get "
                "the latest messages",
                status=409,
                code="cursor_unavailable",
            )
        # A cursor PAST the end is the remaining inexact case, and it is not the
        # same as a stale one: rewind and regenerate shrink a transcript, so
        # `total` can move backwards under a caller that is still holding the old
        # position. Clamping it to `total` would start the read at the end and
        # silently skip every replacement row written below the old cursor, with
        # nothing in the response saying so. So this refuses loudly rather than
        # answer approximately. Reads without `since` are unaffected.
        if since > total:
            raise SessionControlError(
                "this session is shorter than your cursor — it was rewound or "
                "regenerated, so earlier positions no longer line up — read "
                "without `since` to get the latest messages",
                status=409,
                code="cursor_unavailable",
            )
        start = since
        # Positions are absolute; the window slice below is offset-relative, so
        # subtract the durable prefix that is no longer in memory.
        offset = start - base
    else:
        # A tail read never refuses (only a `since` below the trimmed prefix or
        # past the end is), and the two spaces come apart here: slice the
        # in-memory window by OFFSET, but report the index in ABSOLUTE terms so
        # the number still means "position in the session". Conflating them
        # returned an empty window, because `total` counts the frozen prefix
        # the list does not hold.
        offset = max(0, durable_end - limit)
        start = base + offset
    window = messages[offset:][:limit]

    out: list[dict[str, Any]] = []
    for offset, msg in enumerate(window):
        content = str(msg.get("content", "") or "")
        # ``redact_and_truncate`` scans the COMPLETE text before slicing. Cutting
        # first would split a credential straddling the boundary into a prefix
        # that no longer matches the scanner, so the fragment would ship.
        emitted = redact_and_truncate(content, MAX_READ_CONTENT_CHARS)
        row: dict[str, Any] = {
            "index": start + offset,
            "role": str(msg.get("role", "") or ""),
            "content": emitted,
            "ts": str(msg.get("ts", "") or ""),
        }
        if len(content) > MAX_READ_CONTENT_CHARS:
            row["truncated"] = True
        out.append(row)

    _audit(
        caller_session_key=caller_session_key,
        operation="read",
        slot_key=slot.key,
        outcome="allowed",
        detail={"returned": len(out)},
    )
    return {
        "ok": True,
        "target": slot.key,
        "title": sanitize_outbound(slot.display_title),
        # Busy means "more output is coming", which is exactly what a poller needs
        # to decide whether to wait.
        "running": bool(slot.running),
        # True when the target is mid-reply: rows exist that the cursor
        # deliberately does not cover yet, so "nothing new" here does not mean
        # "nothing happening".
        **({"streaming": True} if durable_end < len(raw_window) else {}),
        "queue_depth": len(slot._queue),
        # A turn parked on a tool approval is still ``running``; this says it is
        # waiting on a person rather than working.
        **_pending_approval_fields(state, slot),
        # The model the target's turns use, and a session_set_model pick still
        # waiting for its next turn, so a caller can see whether its pick took.
        # Redacted: the owner's picker stores whatever string it is given.
        # The served model when the backend has reported one (it reflects an
        # inherited default or an active fallback), otherwise the pin.
        "model": redact(slot.served_model or slot.model or ""),
        **_pending_pick_view(slot._pending_model_pick),
        "reasoning_effort": slot.reasoning_effort or "",
        "total": total,
        # The cursor to poll with next. This is NOT `total`: when more than
        # `limit` rows are new, the window stops short of the end, and a caller
        # that polled `since=total` would jump the gap and never see the rows in
        # between. `next_since` is the absolute position just past the last row
        # actually returned, so consecutive polls cover every row exactly once.
        # `total` stays in the response as the backlog depth — the difference
        # from `next_since` is how far behind the caller still is.
        #
        # Returned on trimmed sessions too: positions are based on the
        # durable-only prefix counter, so they stay exact after rows age into
        # the frozen prefix. The refusals above cover the cases that genuinely
        # cannot be exact (a cursor under the trimmed prefix, or past the end of
        # a rewound transcript).
        "next_since": start + len(out),
        "messages": out,
    }


def _summary_text(value: object) -> str:
    """One summary string, redacted and then cut to ``MAX_SUMMARY_CHARS``."""
    text = value if isinstance(value, str) else ""
    # Redact the whole string first (the helper's own order), then cut, so the
    # marker reflects what the redacted text lost.
    redacted = redact_and_truncate(text, len(text) * 4 + 256)
    if len(redacted) <= MAX_SUMMARY_CHARS:
        return redacted
    return redacted[:MAX_SUMMARY_CHARS] + " …[truncated]"


def _bounded_summary(payload: dict) -> dict[str, Any]:
    """The fields a patrol reads, bounded, with a count of what each cut dropped.

    Redaction runs on every emitted string through ``redact_and_truncate``, the
    sink ``read_messages`` uses: the sidecar was redacted when written, and
    re-redacting on the way out covers a scrubber rule added since.
    """
    raw_intents = [i for i in (payload.get("intents") or []) if isinstance(i, dict)]
    intents: list[dict[str, Any]] = []
    for intent in raw_intents[:MAX_SUMMARY_INTENTS]:
        progress = [p for p in (intent.get("progress") or []) if isinstance(p, str)]
        steps = [st for st in (intent.get("next_steps") or []) if isinstance(st, dict)]
        intents.append(
            {
                "title": _summary_text(intent.get("title")),
                # The panel's single word, not the raw progress axis: a
                # ``completed`` intent that was never verified reads
                # ``needs-you``, which is the case a patrol most needs to see.
                "state": _summary_text(
                    intent.get("state")
                    or derive_state(str(intent.get("status") or ""), intent.get("verified"))
                ),
                # The latest progress is what a patrol needs; the next steps in
                # the order the summarizer ranked them.
                "progress": [_summary_text(p) for p in progress[-MAX_SUMMARY_ITEMS:]],
                "progress_omitted": max(0, len(progress) - MAX_SUMMARY_ITEMS),
                "next_steps": [_summary_text(st.get("what")) for st in steps[:MAX_SUMMARY_ITEMS]],
                "next_steps_omitted": max(0, len(steps) - MAX_SUMMARY_ITEMS),
            }
        )
    notes = [n for n in (payload.get("constraints") or []) if isinstance(n, str)]
    return {
        "intents": intents,
        "intents_omitted": max(0, len(raw_intents) - MAX_SUMMARY_INTENTS),
        "constraints": [_summary_text(n) for n in notes[:MAX_SUMMARY_NOTES]],
        "constraints_omitted": max(0, len(notes) - MAX_SUMMARY_NOTES),
    }


async def read_summary(
    state: "DashboardState",
    *,
    caller_session_key: str,
    target: str,
    caller_fenced: bool | None = None,
) -> dict[str, Any]:
    """Return *target*'s cached intent summary, the one the side panel shows.

    Authorized by the gate :func:`read_messages` uses (``authorize_target``):
    the summary is derived from the transcript, so a caller that may not read
    the transcript may not read its digest either. ``operation`` only labels
    the audit line, so a refusal reads as ``session_control.summary``.

    A cache read only. It never generates, so it never spends a model call:
    summaries are written at turn end by the background pass, or on the panel's
    explicit POST, and neither is reachable from here. Gated on
    ``session_summary.enabled`` like the panel GET, so switching the feature
    off stops serving sidecars written while it was on.

    The ``authorize_target`` gate runs before the first suspension, so the
    handler's ``prewarm_enabled_check`` stays valid across it.
    """
    # Late import: chat_summary pulls in the chat package, which imports this
    # module at load time.
    from kiro_crew.dashboard.chat_summary import read_cached_intent_summary

    def _authorize(*, recheck: bool) -> "_ChatSlot":
        return authorize_target(
            state,
            caller_session_key=caller_session_key,
            target=target,
            operation="summary",
            skip_enabled_check=recheck,
            precomputed_ownership_fenced=caller_fenced,
        )

    slot = _authorize(recheck=False)
    running = bool(slot.running)

    cfg = await asyncio.to_thread(KiroCrewConfig.load)
    enabled = bool(cfg.session_summary.enabled)
    payload: dict | None = None
    stale = False
    log = state.conversation_log
    if enabled and log is not None:
        payload, stale = await read_cached_intent_summary(log, slot)
    # Both reads above suspend, the sidecar read for as long as the transcript
    # lock's acquire ceiling. The caller or the target can gain a channel link
    # or a mirror in that window, so the gate runs again, synchronously, before
    # anything is returned, and the answer must still be the same slot.
    if _authorize(recheck=True) is not slot:
        raise SessionControlError(
            "the target session was replaced while its summary was read; try again",
            status=409,
            code="target_replaced",
        )
    bounded = _bounded_summary(payload or {})

    _audit(
        caller_session_key=caller_session_key,
        operation="summary",
        slot_key=slot.key,
        outcome="allowed",
        detail={
            "enabled": enabled,
            "present": payload is not None,
            "intents": len(bounded["intents"]),
        },
    )
    return {
        "ok": True,
        "target": slot.key,
        "title": sanitize_outbound(slot.display_title),
        "running": running,
        "enabled": enabled,
        "stale": stale,
        "generated_at": (payload or {}).get("generated_at"),
        **bounded,
    }
