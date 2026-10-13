"""Which slots have ever been handed over -- one trusted, cheap answer.

The question "was this session ever handed to another session" has to be answered for
every slot a start restores, and the two obvious places to keep the answer are both
wrong.

ON THE SLOT'S OWN METADATA LINE it is cheap and NOT TRUSTED. The line is the
transcript's, which the session's own writers rewrite, so the party whose grant a
handoff revoked can clear the field that says the grant exists. Clearing it is not the
same class of forgery as inventing a ``created_by``: that invents a creator for a slot
out of nothing, while this hands a DELIBERATELY REVOKED authorization back to a party
that already exists and is already authenticated. Cheaper to do and worth more.

IN THE SLOT'S OWN CREW LOG it is trusted and NOT CHEAP. A slot that was never handed
over has no entry to find, and proving that absence means reading the whole surviving
log of that slot -- so the cost falls on every restored tab on every start, forever,
for a feature most of them never used.

So the answer lives HERE: one append-only index, in the same fenced store the entries
are in, listing the DISTINCT slots a handoff has committed for. Read once per process
and cached. A slot absent from it was never handed over and is served by its creator
with no history read at all; a slot present in it has its own log read, which is the
path that already existed. The index grows with the number of slots ever handed over,
not with the number of handoffs and not with the size of any log, so there is no replay
and no per-slot listing.

WHAT MAKES IT SAFE IS THE SHAPE OF ITS FAILURES, and the natural handling of a missing
file is the unsafe one: an absent file reads as the empty set, and the empty set says
"nothing was ever handed over", which serves every slot from its creator. That is
exactly the answer a revoked party wants. So absence is split in two, the same split
:func:`~kiro_crew.crew_log.session_tree.slot_manager` makes:

* the store, or the index inside it, is NOT THERE: decided, and empty. No handoff can
  have been recorded on a store nothing ever wrote, so the first run of a gateway does
  not refuse every slot.
* the index IS there and cannot be read, parsed, or is past its bound: UNDECIDED. The
  caller refuses every slot until a read succeeds, because a short read of this file is
  indistinguishable from "no handoff" and that is the one direction that restores a
  grant the log recorded going away.

A WRITE THAT CANNOT LAND REFUSES THE HANDOFF, for the same reason the entry and the
tree edge do: an unindexed grant is one the next start will not read, so publishing it
would give the recipient a reach that disappears at the next restart while the giver
silently gets it back.

HOW IT IS READ, in one sentence, because three rounds of defects came from reading it
lazily instead: THE INDEX IS READ ONCE, OFF THE EVENT LOOP, AND THE RESULT IS PASSED
DOWN AS A VALUE. A hydrate pass takes one :class:`HandoffIndex` before it builds
anything and hands that same immutable answer to every slot it builds. Nothing downstream
of that read touches the filesystem, and nothing loop-affine consults this module at all.

Two rules follow, and both are the opposite of what a cache invites:

* A READER NEVER PUBLISHES. Only :func:`record_handed_over` writes
  :data:`_snapshot`, inside its own lock, in the same critical section that writes the
  file. A reader that published could overwrite a newer set the writer had already
  published -- dropping a committed handoff from the in-memory answer, and then from the
  FILE at the next whole-file rewrite, which restores the reach the log recorded going
  away. A cold read therefore returns its answer to its own caller and leaves the
  snapshot alone.
* AN UNDECIDED ANSWER IS NEVER KEPT. Caching one makes a transient fault permanent while
  every refusal it produces keeps telling the caller to retry. It costs nothing to
  re-read, because the read happens once per pass rather than once per slot.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Final, NamedTuple

from kiro_crew.atomic_write import atomic_write
from kiro_crew.crew_log.errors import CrewLogError
from kiro_crew.crew_log.store import KIND_SESSION, checked_crew_log_root
from kiro_crew.validation import MAX_SHORT_STRING

logger = logging.getLogger(__name__)


class HandoffIndex(NamedTuple):
    """One read of the handed-over index: the slots it named, and whether it decided.

    THE VALUE THAT GETS PASSED DOWN. A hydrate reads this once, off the loop, and hands
    the same instance to every slot it builds, so the answer cannot change between two
    slots of one pass and no slot's construction reads a file.

    A ``NamedTuple`` so it is immutable and shares nothing: a reader holding one cannot
    be affected by a later write, and a writer cannot mutate what a reader is part-way
    through. It is still a plain ``(slots, decided)`` tuple to anything that wants to
    unpack it.

    *decided* is False only when the index EXISTED and this read could not establish what
    it said. An absent store is decided and empty -- see the module docstring for why
    that split is the whole safety of this file.
    """

    #: The distinct slot keys a handoff has committed for, per this read.
    slots: frozenset[str]
    #: Whether this read established what the index says.
    decided: bool

    def names(self, key: str) -> bool:
        """Whether *key* was handed over according to this read."""
        return bool(key) and key in self.slots

    def hydrates_unresolved(self, key: str) -> bool:
        """Whether a slot built under *key* must come up with its manager UNRESOLVED.

        The question every hydrate actually asks, answered here rather than at each
        build path, because it has two reasons and a path that remembered only one is
        how an unread value gets served as "the creator holds it": the slot was handed
        over and its own log has not been read yet, OR this read did not decide and
        nothing can be concluded about any slot.
        """
        return not self.decided or self.names(key)


#: The fail-closed value. Passed by a caller whose own read of the index failed, and by
#: one that cannot read it at all, so "the read did not happen" and "the read did not
#: decide" reach the hydrate as the same answer -- which is the only safe one, since the
#: empty set would say "nothing was ever handed over" and serve every slot from its
#: creator.
UNDECIDED: Final[HandoffIndex] = HandoffIndex(frozenset(), False)

#: A LEAF OF ITS OWN under the kind root, not a file beside the unit directories: the
#: readers that list that root treat each child as a candidate unit, and a bare file
#: there would be one more shape each of them has to learn to skip.
INDEX_DIR: Final[str] = "handoff-index"

#: The file inside it. One file, because the answer is one set.
INDEX_NAME: Final[str] = "slots"

#: The most slots this index will carry. A handed-over slot is a session a person or an
#: agent deliberately gave to another session, so the real population is small; the
#: bound is here because this file is READ ON EVERY START and a reader that would
#: accept any size is a reader an append loop can make expensive. Past it the read is
#: UNDECIDED and the write REFUSES, which costs a handoff rather than losing one.
MAX_INDEXED_SLOTS: Final[int] = 4096

#: And the byte ceiling the parse applies before it splits anything, so a file grown by
#: something other than this writer cannot be read into memory first and judged after.
MAX_INDEX_BYTES: Final[int] = MAX_INDEXED_SLOTS * (MAX_SHORT_STRING + 1)

#: Serializes WRITERS, and only writers. A reader never takes it, which is the whole
#: point: this index is consulted from the hydrate, and the hydrate is LOOP-AFFINE --
#: it runs on the event loop. The writer holds this across an ``atomic_write`` with
#: ``fsync=True``, so a reader that waited on the same lock would park the entire loop,
#: and the stall watchdog with it, for the length of one handoff's disk sync.
_write_lock = threading.Lock()

#: WHAT THE WRITER HAS PUBLISHED, or ``None`` when this process has recorded nothing.
#: An IMMUTABLE snapshot, rebound as a whole and never mutated in place, so a reader
#: needs no lock to see a consistent one: it takes the name once and whatever it gets is
#: a complete answer, old or new.
#:
#: WRITTEN BY :func:`record_handed_over` AND BY NOTHING ELSE, in the same critical
#: section that writes the file. A reader must not publish what it read, even a decided
#: answer: its read may have started before a write that has since landed and published,
#: so storing it would put the OLDER set back -- losing a committed handoff from memory,
#: and then from the file at the next whole-file rewrite, which hands the revoked
#: creator its reach back. The point of this name is to spare the writer a re-read of a
#: file it already knows the contents of; it is not a read cache.
_snapshot: "HandoffIndex | None" = None


def _index_path() -> Path:
    """The index file's path under the validated store root.

    ``checked_crew_log_root`` is what establishes the root is the real one and not a
    link to somewhere the data home's protections do not reach, so the path is built
    through it on every call rather than cached.
    """
    return checked_crew_log_root(KIND_SESSION) / INDEX_DIR / INDEX_NAME


def reset_caches() -> None:
    """Forget what the writer published. Tests, and a process rebinding to another store."""
    global _snapshot
    _snapshot = None


def handed_over_slots() -> HandoffIndex:
    """One READ of the index. Call it OFF THE LOOP and pass the result down.

    This is the only read in the system, and it is blocking disk work: every caller is
    either already on a worker thread or hands it to one. Nothing loop-affine may call
    it -- a hydrate takes one of these before it builds anything and passes it as a
    value (see :func:`HandoffIndex.hydrates_unresolved`), which is what keeps the file
    read off the event loop and out of the per-slot path.

    NOTHING HERE WRITES :data:`_snapshot`. A published set is served back when the writer
    has one, because then this process already knows the file's contents; otherwise the
    file is read and the answer goes to the caller only. A reader that published could
    put an older set back over a newer published one, which drops a committed handoff --
    the worst outcome this module has, since the next whole-file rewrite then drops it
    from disk too.

    No lock either, in either direction. One name is read and whatever it holds is a
    whole snapshot, because the writer rebinds it rather than mutating it, and the writer
    holds its own lock across an ``fsync``: sharing one lock would make this read wait
    out a handoff's disk sync.

    NEVER RAISES, so no caller needs a wrapper that decides what an exception means --
    and a wrapper is where that decision would be made differently by the second one to
    need it. Anything unforeseen answers :data:`UNDECIDED`, which is the same answer as
    a read that could not establish the file: "the read did not happen" and "the read
    did not decide" are the same fact to a caller, and both must refuse rather than
    serve the empty set.
    """
    published = _snapshot
    if published is not None:
        return published
    try:
        return _read()
    except Exception:
        # No traceback, for the reason the handlers inside ``_read`` give: the frames
        # under this one hold the index itself. Reached only by something ``_read``
        # does not model -- a data home that cannot be resolved at all -- and answered
        # as undecided rather than raised, because a raise here lands in the middle of
        # a build that has no answer for it either.
        logger.warning("handoff index: could not be consulted, so it is UNDECIDED")
        return UNDECIDED


def _read() -> HandoffIndex:
    try:
        path = _index_path()
    except CrewLogError:
        # The root itself is refused -- a linked or wrong kind directory. Nothing can be
        # established about a store this reader will not open.
        logger.warning("handoff index: the crew log root could not be established")
        return UNDECIDED
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        # DECIDED and empty: no handoff has ever been committed on this store, which is
        # every gateway that has not used the verb. Treating this as undecided would
        # refuse every slot on a first run.
        return HandoffIndex(frozenset(), True)
    except OSError as exc:
        # NO TRACEBACK, and that is a decision about this frame rather than a style.
        # Retaining one keeps every local alive: ``raw`` and ``slots`` here ARE the
        # index -- the slot key of every session ever handed over on this store -- and
        # ``path`` is a location inside the operator's own data home. The failure is
        # fully described by its errno, so the stack buys nothing and would put a bulk
        # list of identifiers into a log that outlives the fault.
        logger.warning(
            "handoff index: the index exists and could not be read (%s), so which "
            "slots were handed over is UNDECIDED and the manager-dependent verbs are "
            "refused until a read succeeds",
            exc.strerror or type(exc).__name__,
        )
        return UNDECIDED
    if len(raw) > MAX_INDEX_BYTES:
        logger.error(
            "handoff index: %s is %d bytes, past the %d-byte bound, so which slots were "
            "handed over is UNDECIDED and the manager-dependent verbs are refused",
            path,
            len(raw),
            MAX_INDEX_BYTES,
        )
        return UNDECIDED
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        # No traceback, for the reason above: the frame holds the raw index.
        logger.error("handoff index: the index is not valid UTF-8, so it is UNDECIDED")
        return UNDECIDED
    slots = {line for line in text.split("\n") if line}
    if any(len(slot) > MAX_SHORT_STRING for slot in slots):
        logger.error("handoff index: %s holds an over-long slot key", path)
        return UNDECIDED
    if len(slots) > MAX_INDEXED_SLOTS:
        logger.error("handoff index: %s holds %d slots, past the bound", path, len(slots))
        return UNDECIDED
    return HandoffIndex(frozenset(slots), True)


def record_handed_over(slot: str) -> bool:
    """Record that *slot* has been handed over. Answers whether the index says so now.

    True when the index holds *slot* after this call, INCLUDING when it already did:
    the fact is "this slot has been handed over at least once", so the second handoff of
    a slot writes nothing and is not a failure. That is what keeps the file proportional
    to handed-over slots rather than to handoffs.

    False on every way the write can fail, and the caller then refuses the handoff
    rather than publishing it: a grant the index does not carry is one the next start
    will not read, so the recipient would hold a reach that vanishes at the next
    restart while the giver silently gets it back.

    Rewritten whole rather than appended to. The set is small and bounded, the write is
    atomic, and a whole-file replace cannot leave a half-written final line -- which an
    append can, and which would read as a slot key nobody handed anything to.
    """
    key = str(slot or "")
    if not key or len(key) > MAX_SHORT_STRING:
        logger.error("handoff index: refusing to record %r as a handed-over slot", slot)
        return False
    global _snapshot
    with _write_lock:
        # The writer may trust what IT published, because nothing else writes that name
        # and every publish happened inside this lock after its own file write landed.
        published = _snapshot
        known, decided = published if published is not None else _read()
        if not decided:
            # The index cannot be read, so it cannot be rewritten either: replacing it
            # with what this process believes would DROP every slot the unreadable file
            # holds, turning one unreadable index into a permanently wrong one.
            logger.error(
                "handoff index: cannot record %s because the index could not be read, "
                "and rewriting it from a partial view would drop the slots it holds",
                key,
            )
            # Nothing is published here: the next writer reads again, so a fault that
            # clears does not outlive itself.
            return False
        if key in known:
            return True
        if len(known) >= MAX_INDEXED_SLOTS:
            logger.error(
                "handoff index: %d slots already recorded, at the bound, so %s cannot be "
                "recorded and the handoff is refused",
                len(known),
                key,
            )
            return False
        updated = known | {key}
        try:
            atomic_write(
                _index_path(),
                "".join(f"{one}\n" for one in sorted(updated)),
                fsync=True,
                newline="\n",
            )
        except (OSError, CrewLogError) as exc:
            # NO TRACEBACK on the REFUSAL PATH especially. This frame holds ``known``
            # and ``updated`` -- the whole set of handed-over slot keys -- beside the
            # one being added, so a retained stack here would turn the fail-closed
            # answer into a bulk disclosure of exactly the identifiers the index is
            # fenced to protect. The slot key names what did not land, which is what a
            # person acting on this needs; the errno says why.
            logger.warning(
                "handoff index: %s could not be recorded (%s), so the handoff is refused",
                key,
                getattr(exc, "strerror", None) or type(exc).__name__,
            )
            return False
        # THE ONE PUBLISH, rebound as a whole, AFTER the write landed and still inside
        # the lock: a lock-free reader sees the previous complete answer or this one and
        # never a half-built set, and no reader can put an older set back over it
        # because no reader writes this name at all.
        _snapshot = HandoffIndex(frozenset(updated), True)
        return True
