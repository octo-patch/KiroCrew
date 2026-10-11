# Session Control Module

## Overview

Session control lets one of the user's chat sessions observe and interrupt
another: open a new session, fork a session so the new one carries a
transcript, stop an in-flight turn, close (archive) a session, and read a
transcript tail.
It exists because a session cannot see what its peers are doing. A session that
has spent an hour on a PR cannot tell whether the session watching the build has
finished, and today the only way to find out is for the human to switch tabs and
look. Session control lets the session ask directly.

The `session_*` tools on `kirocrew-dashboard`, one strict-internal route each, and
two config switches: `agent.session_control` plus the member-dispatch bypass
ceiling. Every route in the Route column is on `_STRICT_INTERNAL_API_PATHS`; an unlisted one is
unreachable in production because the caller's `X-Internal-Secret` is ignored.

| Tool | Route | What it does |
|------|-------|--------------|
| `session_create` | `POST /api/session-control/create` | Open a new, empty session in the caller's workspace, optionally filed into a sidebar folder at creation |
| `session_fork` | `POST /api/session-control/fork` | Open a new session that CARRIES a copy of a source session's transcript — the caller's own by default — the way the dashboard's Fork button does; optionally titled, filed, and cut at a fork point |
| `session_stop` | `POST /api/session-control/stop` | Stop another session's in-flight turn |
| `session_end_wait` | `POST /api/session-control/end-wait` | Wake a session the caller CREATED from the `wait` tool early, keeping its turn; any other target is refused `not_creator`, for every caller class |
| `session_retry` | `POST /api/session-control/retry` | Re-run a failed last turn through `continue_slot_turn`, the Resume button's path, with `require_interrupted=True` (`turn_not_failed` otherwise) and `authorize_target` re-run synchronously under the slot lock before the continuation is queued |
| `session_continue` | `POST /api/session-control/continue` | The same `continue_slot_turn` call without `require_interrupted`, as the Continue button makes it, through the helper `retry_target` shares (`_continue_turn_of_target`); the result's `body` is `resumed` or `continued`, whichever continuation `_is_interrupted` chose, and the allowed audit line records it. Not in any auto-grant: a member or conductor is asked before each call |
| `session_set_model` | `POST /api/session-control/set-model` | Record a pending model and/or reasoning-effort pick on an idle session; `apply_pending_model_pick` commits it at the start of the target's next turn after re-running `authorize_target` in the same synchronous step. Each half yields separately to a newer user pick from its own control. A busy target is refused with `target_busy` and keeps its settings |
| `session_reload` | `POST /api/session-control/reload` | Relaunch the agent process of an idle session the caller created, through `chat_handlers.reload_slot_session` (shared with the tab menu's Reload session). The transcript is kept and gets one notice naming the caller. Self, remote-crew and busy targets (turn running or starting, queued messages, sub-agents) are refused |
| `session_rename` | `POST /api/session-control/rename` | Set the caller's own title or a target's through `chat_title.apply_manual_title`, the sidebar rename's code, after `authorize_target` with `allow_self`; a scheduled run's own session is still refused as `unattended_target`. The title is final (`_TITLE_ORIGIN_USER`). An empty title, one over 200 characters (before or after `_scrub_text` redaction, which lengthens credential-shaped text into markers), one with no character that renders (only whitespace, characters `terminal_safe._is_invisible` reports such as zero-width, variation-selector and Hangul-filler characters, or the blank Braille pattern U+2800), one containing U+2800 anywhere (scanning normalization keeps it, so it could split a secret past the redactors), or one with a control character or any text-direction control (embeddings, overrides, isolates and the U+200E/U+200F/U+061C marks), or any other `Cf` format character except U+200C/U+200D and the tag characters U+E0020..U+E007F that spell flag subdivisions (`_is_allowed_format_char`; the tag block's other `Cf` character, U+E0001 LANGUAGE TAG, is refused; its unassigned code points are not `Cf`), is refused with `invalid_title`; if the history write fails the live slot keeps and broadcasts the new title (best-effort, like the sidebar and channel renames; the next full save writes it) and the verb answers `title_persist_failed` (500) instead of `ok` |
| `session_generate_title` | `POST /api/session-control/generate-title` | Regenerate a session's title through `chat_title.generate_manual_title` and `chat_title.apply_regenerated_title`, the sidebar "Regenerate title" code: one awaited titling-model call; the result replaces the current title (a hand-set one included) and stays auto. After `authorize_target` (no `allow_self`), the target must be one the caller created (`not_creator`, 403) for every caller class. One admitted call per target per `create_rate_limit.WINDOW_SECS` (300 s) through the shared creation limiter keyed by the target slot, stamped before the model call (`title_rate_limited`, 429). After the model answers, the slot identity and closing flag are checked (`target_replaced`, 409) and `authorize_target` runs again synchronously; then every field is assigned before the first await, as `session_rename` does, and the save only makes that change durable. The save passes `still_current` (slot identity), so `_persist_title` refuses the write under the history lock when the slot left its key, and the verb then answers `target_replaced` (409) rather than overwriting a successor's saved title. A rename that landed during the model call wins (`changed: false`). If the history write fails the live slot keeps the title and the verb answers `title_persist_failed` (500). The sidebar push is skipped when the slot left its key during the save |
| `session_close` | `POST /api/session-control/close` | Close (archive) another session, or the caller itself, as the tab ✕ does — heavier than stop, and recoverable rather than a delete |
| `session_revive` | `POST /api/session-control/revive` | Bring an archived session back into the live sidebar, as clicking it in the History tab does — the mirror of close, optionally filing it into a folder |
| `session_send` | `POST /api/session-control/send` | Deliver a message that another session runs as its next turn, or cut it into the turn already running (`steer`) |
| `session_broadcast` | `POST /api/session-control/broadcast` | Deliver ONE message to several sessions — by default every session the caller created — in a required `queue` or `steer` mode, reporting the outcome per target |
| `session_status` | `GET /api/session-control/status` | List the sessions the caller stood up and what each is doing, with the roster taken from the crew log's session tree so a session that is gone still appears |
| `session_adopt` | `POST /api/session-control/adopt` | Take another session under the caller in the session tree, keeping the parent it had on record; refused when the target is already above the caller |
| `session_handoff` | `POST /api/session-control/handoff` | Give a session to another session, which manages it from then on: the target's current manager (its creator until a handoff says otherwise) names a recipient it also manages, the recipient gains the reach to send, read, stop, close, adopt, release and hand on while the giver loses it, and the sidebar edge moves with it through the same record `session_adopt` writes |
| `session_release` | `POST /api/session-control/release` | Let a session the caller holds, or the caller itself, out from under its parent; the released session keeps its own subtree |
| `session_read_message` | `GET /api/session-control/read` | Read another session's transcript tail + liveness |
| `session_summary` | `GET /api/session-control/summary` | Read another session's cached intent summary + liveness, authorized as `session_read_message` is; never generates one |

`session_handoff` is the only writer of `_ChatSlot._manager`, the slot's record of
the session that HOLDS it. `manager_of` answers `_manager or _created_by`, so a
gateway where nothing has been handed over authorizes exactly what it authorized
before the field existed, and the fence reads the manager only for the operations
`HANDOFF_WIDENED_OPERATIONS` names (`send`, `read`, `summary`, `stop`, `close`, `adopt`,
`release`, `handoff`) and the creator for every other verb. `summary` is in that set with
`read` because it is authorized as a transcript read is and is generated from the
transcript. `broadcast_audience` reads the same `_managed_by_other`, because `send` is one
of those operations and an audience that disagreed with the fence judging each delivery
would put a handed-away session in its giver's default audience, where every delivery is
then refused, and leave it out of the audience of the manager that can reach it.
`_created_by` is never rewritten, so `_delegation_lineage_fenced`, the per-creator slot cap
and the create and fork gates keep their own answer.

The LOG is the ONLY store for WHO manages a session. The slot's metadata line never
carries the manager: the codec has no `manager` row, so no save writes one and no restore
reads one. A copy there would be written on every save and overwritten on every restore,
and it was itself how a revoked manager came back, because a line outlives the grant it
names. `created_by` stays on the line because nothing else records it.

WHETHER A SLOT WAS EVER HANDED OVER IS NOT ON ITS LINE. The line is the transcript's,
and the transcript's writers include the session whose grant a handoff revoked -- so a
field there is one the losing party can clear, and clearing it IS the whole
authorization: an unhanded slot is served by its creator without reading anything.
Clearing it is also worth more than forging `created_by` and cheaper: forging invents a
creator out of nothing, while clearing hands a deliberately revoked grant back to a
party that already exists and is already authenticated.

So the fact lives in the fenced crew log, as one append-only index of the DISTINCT
slots a handoff has committed for (`crew_log/handed_over.py`, at
`<store>/handoff-index/slots`). It is read once per process and cached, so a slot the
index does not name is decided by a set lookup and reads no history -- the property the
line was there for, with the authority somewhere its subject cannot write. It grows
with the number of slots ever handed over, not with handoffs and not with any log's
size, so there is no replay and no per-slot listing. The index entry is written after
the `session/manager_set` entry lands and before the manager is published, so it never
names a handoff the log does not hold; a grant that lands without it is inert, because
an unindexed slot's log is never consulted. One that cannot be written refuses the
handoff 409 `handoff_not_recorded` and takes the grant back on the log.

ITS FAILURES ARE SPLIT, because the natural handling of a missing file is the unsafe
one: an absent file reads as the empty set, and the empty set says "nothing was ever
handed over", which serves every slot from its creator -- exactly the answer a revoked
party wants. So an absent store or absent index is DECIDED and empty, since no handoff
can have been recorded on a store nothing ever wrote and a first run must not refuse
every slot; an index that EXISTS and cannot be read, parsed, or is past its bound is
UNDECIDED, and every slot is then refused until a read succeeds. A write cannot proceed
on an undecided read either: rewriting the file from one process's partial view would
drop every slot the unreadable file holds, turning a temporarily unreadable index into a
permanently wrong one.

ITS FAILURES ARE LOGGED WITHOUT A STACK. Every frame in the index's reader and writer
holds the index -- the slot key of every session ever handed over on this store -- beside
a path inside the operator's data home, so a retained traceback would turn a fault, or a
refusal, into a bulk disclosure of exactly the identifiers the store is fenced to
protect. The refusal path is the one that matters most: it is reached when a handoff is
being refused, and a stack there trades fail-closed for a leak. The errno says why and
the slot key says what did not land, which is what acting on the log needs.

THE COST, stated rather than left to be found. Every manager-dependent answer now
depends on the index being readable, where an unindexed slot previously needed nothing.
That is a new availability dependency and it is deliberate, because of its BLAST
RADIUS: a corrupt index refuses the verbs a handoff widens, with a retryable code, and
nothing else. The gateway still starts, slots still hydrate and list, and every verb
that does not read the manager still works; repairing or removing the file clears the
refusal with no restart. The fear a reader brings to "fail closed on an unreadable file"
is the other outcome -- a gateway that will not boot -- and that difference is the whole
reason this trade is the right one, so a test pins both halves rather than only the
refusal.

HOW IT IS READ IS THE REST OF THE DESIGN, in one sentence: THE INDEX IS READ ONCE PER
PASS, OFF THE EVENT LOOP, AND THE ANSWER IS PASSED DOWN AS A VALUE. A hydrate takes one
`HandoffIndex` in its own worker-thread step before it builds anything and hands that
same immutable answer to every slot it builds; `metadata_codec.apply` takes it as a
REQUIRED parameter and touches no filesystem, so nothing loop-affine consults the index
module at all. A caller whose own read failed passes `UNDECIDED`, which hydrates every
slot unresolved.

Three findings came out of reading it lazily instead, and each looked local and correct
at the site that made it. A read inside the loop-affine apply is a blocking read on the
EVENT LOOP, once per restored tab -- and once per tab forever when the index cannot be
read, because an undecided answer is deliberately not kept. One lock shared by the
reader and the writer would have been worse still: the writer holds its lock across an
`atomic_write` with `fsync=True`, so a reader waiting on it parks the loop, and the
stall watchdog with it, for the length of a handoff's disk sync. The lock therefore
serializes WRITERS ONLY.

A READER NEVER PUBLISHES, and that is the sharpest of the three. Only the write path
stores the in-memory snapshot, inside its own lock, in the same critical section that
writes the file -- which spares the next writer a re-read of a file it already knows.
A reader that stored its answer would lose a handoff: its read can begin before a write
that has since landed and published, so storing it puts the OLDER set back, and the next
whole-file rewrite then drops that slot from DISK too. The grant the log recorded going
away is back, by way of a cache. What makes the lock-free read safe is the same property
from the other side: the snapshot is immutable and rebound as a whole, so a reader takes
the name once and whatever it gets is a complete answer, old or new, never a half-built
set.

And AN UNDECIDED ANSWER IS NEVER KEPT, which costs nothing now that the read is once per
pass. Keeping one would make a transient fault permanent -- one `OSError` and every
handoff refused for the life of the process, while the refusal keeps telling the caller
to retry in a moment: a fact that could not be read served as a settled answer, plus a
promise with nothing behind it.

The commit point is this verb's own
`session/manager_set` entry, appended inside the lock after the tree edge has landed and
awaited through `awaiting_settlement`, and `_manager` is published only once that entry is
on disk. That wait answers three values rather than two, because the two ways of not being
on disk need opposite handling: an append the writer gave up on left nothing behind and
refuses 409 `handoff_not_recorded` outright, while an append still QUEUED when the bound
runs out may land at any time afterwards and would then be the newest statement on the
log. The second case is CORRECTED rather than abandoned -- `_revoke_manager_record` queues
a further entry behind it, and the writer is FIFO per unit, so whichever order the two
land in the newest statement is the correction.

That correction needs the log to be able to say "back to nobody", which is what
`session/manager_set.to_creator` states: a statement that could only name a successor
cannot CLEAR a manager, and naming the creator instead would record a handoff TO the
creator rather than none at all. A slot that had a published manager reverts to that
manager; one that had none reverts to its creator.

A recorded manager is RE-VALIDATED before the rebuild publishes it, through
`_manager_still_admissible`, which re-asks the admission the verb checked before it
published: the manager is not a member's own DM surface (by key, so it answers without a
session), it is still open, it AGREES with the managed session on its memory store, and
it is not a session that cannot be reached at all (`_live_target_refusal`). Fail closed --
a reason that cannot be determined is a reason.

`resolve_slot_manager` is the ONE way the field is ever established on a slot, and every
path that builds one calls it before publishing: the boot reconstruction, a revive, a
resume from history (those two share `resume_slot_from_history`, so one call covers both)
and targeted rehydration (`rehydrate_slot_from_history_async`, reached when a cron
notification wakes a worker the startup budget skipped). One helper rather than a call
added per path as each is found, because that is the shape of hole this line has closed
four times.

FOUR ATTRIBUTES, ONE WRITER. `_manager`, `_manager_tab`, `_manager_unresolved` and
`_manager_grant_inadmissible` are one fact in four slots -- what this process can say
about who drives this session -- and `publish_manager` is the only thing that writes
them. It takes all four as arguments, assigns all four, and contains no `await`; a
caller that needs an awaited value, such as the occupancy read off the log, awaits it
BEFORE calling and passes the result in, so the suspension happens while the slot still
holds its previous whole statement. Spread across the resolve, the verb, the
compensation and the hydrate, the four were writable from twenty-one places, and the
same ordering defect was found at four of them in turn: flags cleared, a read awaited,
the key written after -- which publishes "resolved" with an empty key, and an empty key
on an indexed slot reads as "the creator holds it". Each point-fix was correct and each
left the other points, because the invariant is not statable at a site. Two source
tests pin it: a fifth assignment site fails one, an `await` inside the writer fails the
other.

The same shape on the entry: `_emit_manager_record` is the only caller of
`emit.on_manager_set` outside the emitter. `manager_tab` is optional there and has to
be, since the entry forms that name no manager carry no occupancy -- so omitting it is
silent at the write and expensive at the read, where the restore brings back a bare
name and the key-to-key compare then refuses the manager that is still live. One
assembly that names every field is what makes the occupancy non-optional for the two
writers, the handoff's own entry and the correction that takes it back. A third call
site fails a source test.

ATOMIC IS NOT ORDERED, which is why `publish_manager` also carries a REVISION. Writing
the four together stops a reader seeing half of two statements; it says nothing about a
statement that was already old when it arrived. A resolve reads the crew log off the
loop and then publishes, and a handoff can commit inside that window -- entry, index,
publish -- so the older read then publishes the manager the handoff just revoked. Four
atomic writes of a stale fact are still a lost update. A caller that suspends takes
`manager_revision` first and hands it back as `expect_revision`; a publication whose
revision has moved is DISCARDED and answers False. EVERY publication the resolve derives
from a read hands it back, and the revision is taken before the FIRST suspension, which
is the index read rather than the log read. The two earliest branches are the cheapest
to miss and the worst to: they publish a CLEARED statement, so a stale "never handed
over" verdict does not merely go stale, it wipes a manager a handoff published during
that read -- and an empty manager is the revoked creator. A source test enumerates the
publications rather than listing them. Not a retry: a read that lost to the
authority would re-read the log the authority just wrote. An authoritative writer -- the
handoff publishing a grant it already made durable, a refusal forcing the slot
unresolved -- passes nothing and always wins, which is what puts the discard on the
right side of the race. The counter is process-local and never persisted: the log is
the authority across restarts, and a counter on a line would claim an order between two
processes that never compared their reads.

EVERY STATEMENT ABOUT THE LOG ADVANCES IT, including a correction that changes no value.
A refused handoff whose entry already landed is corrected on the log, and a correction
that touched nothing else would leave a resolve already in flight holding the refused
recipient and a revision snapshot nothing advanced -- so its publish would pass the
check and grant exactly the reach the refusal withheld. The working correction path
therefore restates the restored manager and occupancy; the value is the one the slot
already holds, so advancing the revision is the whole effect. The failing path needs no
addition, because publishing UNRESOLVED advances it already. That is why no lock spans
the window: the restatement is unconditional and the resolve's publish is not, so a
resolve that publishes first is overwritten and one that publishes later is discarded.

IDENTITY BELONGS ON THE RECORD, NOT ONLY ON THE COMPARISON. `_manager_record_refusal`
asks, when the value is READ, whether the session answering to the manager's key is the
one the record names. That cannot save a value that was WRITTEN for a different session
with nothing recording which. A slot key is a name: close a handed-over session, open
one under the same name, and both sessions' units sit in that slot's store -- the newer
one holding no manager entry, so a read qualified by name alone falls through it to the
older one's grant and a fenced manager drives the replacement with no new handoff. So
the target's tab identity is REQUIRED on every `session/manager_set` entry
(`slot_tab`), and
`session_tree.slot_manager` takes the asking session's identity and matches every entry
against it. An entry for another incarnation is passed over, a session with no entry of
its own gets the decided answer "no handoff was recorded", and a session that cannot
say which incarnation it is gets an UNDECIDED one rather than a creator-held one. A
handoff of a target with no identity is refused rather than recorded weakly, and the
emitter will not assemble such an entry at all.

The hydrate reads only WHETHER the index names this slot, which is the one question it
owns: does the crew log have to be read for it at all. Which incarnation a grant was
written for is answered once, by the log, where every entry carries the identity --
asking it in both places would put the answer in two places and the next hole in
whichever one was forgotten. The index is keyed by slot alone for exactly that reason.

AND A PATH THAT REFUSES NEVER REWRITES WHAT IT REFUSES, which is the same rule facing
the other way. "Who holds this key now" and "who the record names" are different values
that agree in the ordinary case, which is what makes mixing them up quiet. Only the
handoff may read the live occupant (`_tab_identity`), because that is where the record
is MADE, and it reads it once and uses the one value for both the entry and the publish.
Every other path APPLIES a record and publishes the occupancy that record carries. The
inadmissible restore got this wrong: it stamped the live occupant of a retired manager's
key, so a fenced replacement sitting on that key had its own identity published, the
next failed recovery handoff copied it into the correction, and the log then named the
replacement -- control nobody handed over, one store rebind from taking effect. A source
test pins the two permitted readers, and `publish_manager` states the rule: every caller
hands in a recorded value and no caller mints one.

It is backed by DEFAULT-DENY rather than by remembering. A slot's manager lives only in
the crew log, so a slot hydrated from a metadata line has no established manager --
`metadata_codec.apply` sets `_manager_unresolved` for every hydrating purpose, and only
`resolve_slot_manager` clears it, and only on an authoritative read. While it stands,
every read that authorizes against the manager is refused 409 `manager_unresolved`. So a
build path added later that publishes a slot without consulting the log does not silently
revert a handoff and hand a revoked giver its reach back: it produces a slot that refuses
the manager-dependent verbs, which is visible. A freshly minted slot never carries the
flag -- nothing was handed over before it existed.

AND EVERY ONE OF THOSE REFUSAL BODIES IS GENERIC, which is a containment rule rather
than a style. They are raised from the READ, and the read runs BEFORE the ownership
fence has decided the caller may reach the slot at all -- that is the whole point of
putting the guard on the read, and it is also what makes the wording a disclosure
surface. A body that interpolated the target's title answered "what is this stranger's
session called" to any fenced caller willing to name a key, and one that named the
manager's key answered "who is driving it": being refused was the way to ask. So no body
carries the target's title, the manager's key, or any slot key, and a test asserts all
three are absent from every pre-authorization refusal. The CODE stays specific, because
the caller still has to know whether to retry and a code says nothing about whose session
it is. Same rule as the tracebacks this module drops on its fail-closed paths: a path
that refuses must not become the way to read what it protects.

THE REFUSAL IS ON THE READ, not at a gate. `require_manager_of` is the one way an
authorization answer reads who manages a slot, and it refuses for both ways the value can
be unknown before returning it; `manager_of` is the raw read and belongs to the rebuild
alone. The refusal used to be two conditions at the top of `authorize_target`, keyed on
`operation in HANDOFF_WIDENED_OPERATIONS` -- a predicate that APPROXIMATES "this
operation reads the manager", and it was wrong in both directions. Too wide: the fence
short-circuits on `ownership_fenced`, so an unfenced caller never reads the manager and
its answer could not change, yet it was refused anyway -- which on a gateway that never
hands anything off is every caller, after every restart. Too narrow: three sites reach
the value without passing that gate -- the handoff verb's own `already_manager` check,
its commit's `previous_manager`, and `broadcast_audience`, which filters every slot by
who manages it for any caller. Put on the read, the guard cannot be reached around, and
a test pins it from source: no function outside `require_manager_of` and the rebuild may
call `manager_of`.

ONE PREDICATE ANSWERS "may this caller drive this slot", and that is `_managed_by_other`.
Everything that can go stale while a published record STANDS lives in
`_manager_record_refusal`, which `require_manager_of` reaches and nothing else does: the
IDENTITY of the session answering to the manager's key (`_tab_id`, which is persisted,
so a resume carries it across while a new occupant of the name mints its own) and STORE
EQUALITY between the two slots, the positive principal test. Both were established when
the grant was admitted and both can stop being true afterwards, under the manager's own
slot lock rather than the handed session's -- so a grant admitted while the stores
agreed would keep answering after they stopped agreeing, and the next restart is what
would notice, since the rebuild re-validates on exactly that comparison and drops the
grant. Re-asked on the read, the drain re-asserting an admission gets both with no code
of its own, which is the reader that found the shape. A second site comparing a manager
read against a key fails a source test; `_managed_by_other` is readable for a WRITE so
the fresh handoff the refusal names stays reachable.

A KEY NAMES A SLOT AND NOT A SESSION, and that gap is the defect this line has closed
fourteen times, each time as a new copy of the comparison rather than a new idea about
it. The sessions answering to one key over a gateway's life are many: a close and a
resume put a new object under the same name, and the explicitly-named keys
(`cron-{id}`, `workflow-{run_id}`, a channel's own) are handed on deliberately. So "the
manager's key equals the asker's key" answers a question about names while the caller
asked one about sessions, and the difference is a grant a replacement inherits from the
session that was given it and then lost it. The fix is therefore structural rather than
a review habit: `manager_record_names` is the ONE place a tab identity meets the record,
`_managed_by_other` is the one predicate built on it, and a source test asserts that NO
key comparison anywhere stands without one of the two being asked in the same function.
The count it asserts is zero standing alone -- not a list of approved sites, because an
approved-site list is what a fifteenth instance gets added to.

Two consequences follow from stating it that way. Every asking site passes its OWN
identity, including the live callers where the compare is redundant by construction (a
live caller is the occupant of the key it was authorized under), because an exemption is
a list and a list is what the next caller is added without. And `_created_by` stays
outside the rule, for a reason rather than by omission: it is written at birth and never
written again, so no grant moves through it and there is no revoked predecessor for a
replacement to inherit from.

An ABSENT record names nobody, which is the direction that has to be right too. An empty
`_manager` is the decided claim "the creator holds it", carrying no identity to compare,
and reading that as "names somebody other than the asker" would fence every creator out
of its own worker -- on a gateway where nothing was ever handed over, that is every
delivery there is. And the handoff's own `already_manager` check asks through the same
predicate rather than beside it: a key-only answer there refused the one handoff that
clears a stale record, which is exactly what every other refusal about that record names
as the way out, so the denial became permanent for the life of the slot.

EVERY STAMP RECORDS A RESOLVED VERDICT, and that set is enumerated rather than
remembered. `send_origin_meta`'s `manager_fenced` says whether the manager fence APPLIED
to this admission, which is the whole licence for the drain to re-compare the sender
against the manager. Two sources are legitimate: `manager_fence_applied`, which
`send_target` derives once under its own name so no closure reads the possibly-`None`
argument, and the literal `False` from a verb whose gate never reads the manager. A raw
`caller_fenced` is neither -- it is the CALLER's fence status, a different question --
and stamping it lets the drain drop an accepted delivery as `sender_no_longer_manages`
against a predicate that was never applied to it, after the verb has already answered
ok. `retry` is the case: it is not in `HANDOFF_WIDENED_OPERATIONS`, so its gate
authorizes on `_created_by` alone, and its stamp says the fence did not apply. A source
test enumerates the stamp sites, fails on any other source, and reads `retry`'s
exemption off `HANDOFF_WIDENED_OPERATIONS` rather than off a comment, so adding `retry`
to that set fails the test instead of silently making the stamp wrong.

At the drain, `manager_unknown` carries every one of those refusals as one snapshot
boolean, compared only when the entry RECORDED it. The boolean set is the wrong
fail-closed floor for this one: the restore path strips the containment stamp
deliberately and every hydrated slot starts `_manager_unresolved`, so counting it against
the unmarked baseline would make the first drain after a restart delete every prompt a
person had typed. Those were admitted by the authenticated composer and never read the
manager. A STAMPED entry is still dropped -- its admission did read the value, and the
drain cannot re-read it.

WHICH REFUSAL IT IS DECIDES WHETHER A QUEUED MESSAGE IS HELD OR DROPPED, and the one
boolean hides that. Fail-closed governs granting reach and never governs keeping data,
so a read that has not LANDED holds the entry: the refusal arms its own re-read
(`_arm_manager_reresolve` fires from `manager_unresolved` and from no other code), so the
wait costs a pass, and destroying an accepted message on a read in flight spends a
person's work to protect a reach nobody can show was at issue. The other two are SETTLED
answers about a record that stands -- a recorded manager that cannot hold the slot, and a
key a different session answers to -- and nothing moves either but a person handing the
session over again. Waiting on one of those is not the conservative choice but the
unbounded one: a held entry is marked so a merge run breaks at it, so a held HEAD shuts
the queue behind it for the life of the process, with the prompts a person typed into
that session shut in with it, and the entry is delivered to nobody in the end anyway. So
a settled refusal drops with the reason the refusal already carries and the sender told.
`manager_read_is_pending` is the one place that distinction lives, because the two
readers that hold data back would otherwise each spell it -- and spelling it twice is how
the three came to be treated alike.

THE REFUSALS THAT WAIT ARE A SET, NOT A CODE, and the sweep asks for it rather than
listing it (`manager_refusal_holds_the_entry`). There are two: an unread manager, which
re-arms its own read, and a hop of the manager line mid-publication
(`manager_line_pending`), which the close or append already in flight settles either
way. The sweep compared against the first alone, so the second -- a 409 whose own
message says the delivery is held rather than judged -- put the entry on the drop list
and told the sender its accepted message had been dropped. A reader cannot forget a
member of a set it does not spell, which is why the set lives with the refusals and the
sweep holds no copy of it.

PROVENANCE DESCRIBES WHO AUTHORISED THIS TEXT, so replacing the text drops it.
`forget_admission_provenance` is called wherever a queued entry's content is replaced,
and everything the drain does with the stamp -- re-comparing the sender against the
manager, counting the manager unknown, reporting a drop back -- stops applying with it.
The case that forced the rule is the worst one on this path: a lead queues a prompt on a
busy worker, the person at the dashboard edits the queued row, and the entry still
carries the lead's stamp -- so handing the worker over makes the sweep delete THE
PERSON'S OWN WORDS as a lapsed cross-session grant, and until a save carries the queue
that row is the only copy. Unconditional on a replacement rather than special-cased to
the authenticated editor: an edit from a third session leaves a stamp naming neither the
author nor the editor, which is the same defect, and a rule that holds for every
replacement cannot be forgotten by the next path that adds one. Nothing is granted by
dropping it -- the entry was already admitted, and both readers treat an unstamped entry
as one whose manager was never read, which is the keep-the-data direction. An UNEDITED
delivery from a sender whose reach demonstrably ended still drops, and its sender is
still told: the exemption is about the text having changed, not about the check.

FAIL-CLOSED GOVERNS GRANTING REACH; IT NEVER GOVERNS KEEPING DATA. The refusals above
decide who may act on whom, and they fail closed because the cost of guessing is a reach
nobody was given. A queued message is not that decision: it is content somebody was
already told had been accepted. So at the drain the two cases part. A grant that
DEMONSTRABLY lapsed -- the manager moved, `sender_no_longer_manages` -- drops the entry
and tells the sender, because holding it would deliver it the moment the manager came
back. A manager that merely cannot be READ holds the entry instead: not delivered,
because delivering would grant a reach this gateway cannot verify, and not dropped,
because nothing about the message is known to be wrong. `_drop_stale_admissions` reports
a held head and the drain starts nothing from it; `require_manager_of` arms the re-read
on every refusal, so the hold ends on its own.

The CALLER half of an admission is re-asserted at the drain too, and separately, because
it is not a per-slot constraint: it is a relation between the target's current manager and
one entry's stamped sender. While the fence read `created_by` it needed no re-check -- a
slot's creator is written at birth and never moves, so the relation that admitted an entry
could not flip while it waited. A handoff moves it, so a lead can queue a message on a
busy worker, hand the worker to a conductor, and have the message run under an
authorization the handoff revoked. `sender_lost_its_reach` compares the stamped sender
through `_managed_by_other` -- the SAME predicate `authorize_target` admitted the entry
with, not a second manager read placed at the drain -- and an unread manager raises out of
it, which is a drop under `manager_unknown` rather than a claim about the sender. Only a
cross-session delivery is re-checked: a human-typed entry carries no `send_origin_slot`,
and neither does a restored one, so neither is dropped for a relation their admission
never read.

The read's refusal is also NARROWED to the slot it is about. The strict listing exists so
a caller that validates against it cannot pass against a record that is quietly short,
and refused store-wide one damaged unit makes every slot's listing incomplete -- so a
slot whose own log reads perfectly comes up unresolved and its manager-dependent verbs
are refused until somebody repairs an unrelated session. An unaccounted unit is skipped
only when its own `session/opened` entry attributes it ELSEWHERE, read positionally from
line 2 because `read_head` withholds the entry exactly when the header is the thing that
failed. One that cannot be attributed still refuses, and a cross-slot caller keeps the
store-wide refusal. `CrewLogError` is also caught on that path: it is not an `OSError`,
so it escaped every handler and surfaced as a crash rather than as the undecided read it
is.

A per-slot refusal also ARMS the read that clears it. `resolve_slot_manager`'s callers
are all build paths and none runs again for a slot already open, so an unresolved slot
used to stay unresolved until the process restarted while every refusal said "retry in a
moment". One bad unit anywhere in the store is enough to get there, since the listing
behind the read is store-wide and a `CrewLogError` from it is not an `OSError`. The
re-read is scheduled from the refusal, deduplicated per slot, so the window is
self-clearing and the instruction is true.

UNRESOLVED is its own state and never collapses into a decided one. An undecided read
establishes nothing, so the slot goes unresolved and carries no key either -- the four
attributes move together, and every reader is refused on the flag while the re-read the
refusal arms is what restores the key. An empty `_manager` on a resolved slot is the
decided answer "the creator holds it", which a read that could not see the bytes cannot
make. A store that does not EXIST is decided -- nothing was ever recorded there, so no
handoff can exist -- which is why `slot_manager` answers `FileNotFoundError` on the
listing as "no handoff" and only a store that is there and unreadable as undecided. Without
that split a first run, where no crew-log store exists yet, would refuse every slot.

WHETHER THE LOG CAN BE READ IS THE READER'S ANSWER TO GIVE. `slot_manager` already
separates those two cases, so nothing upstream consults `emit.enabled()`. Emission and
readability are different facts: switching the crew log off stops new entries and leaves
every entry already on disk exactly where it is. Reading a disabled emitter as "no
handoff is recorded" turns an operator's config change into a revoked giver getting its
reach back, and the index makes that inference visibly wrong rather than merely
unsupported -- an indexed slot is one whose handoff committed, so "nothing was ever
recorded" is contradicted by the slot's own line.

A handoff is also refused when the TARGET is not ownership-fenced and the RECIPIENT is
(`deputy_refused`). The fence decides what a caller may reach and, for the verbs a handoff
widens, it reads the manager -- so a fenced agent-created worker given one of the person's
own tabs drives a session the fence does not apply to, and through it reaches every session
in the workspace while reading the results back. Each half of how it got there is
admissible on its own, which is why the refusal is on the combination. Refused here rather
than by making a managed slot count as fenced, which would change the TARGET's fence status
as a side effect of being managed: a person's own tab would silently lose its own reach for
having been handed to someone. Fenced to fenced (a lead's workers to a lead's conductor)
and unfenced to unfenced (a person's own tabs) both pass; only the mixed direction builds
a deputy.

The verb applies that SAME store comparison, so the answer a caller gets is the answer
the next boot will give. `_receiver` compares the recipient against the giver AND against
the target, before the append and again after the LAST suspension before the publish,
which is the index write rather than the append: that write takes a lock off the loop
and
cross-process lock off the loop, and inside it the recipient's own routes -- which take
the recipient's slot lock and not this verb's -- can switch its agent, rebind its memory
store, attach a channel link, or close and reopen the session under the same key. A
verdict formed before it and acted on after it is a verdict about a session that may no
longer be the one being handed a worker. The earlier check is kept because it is free
and spares the log a correction in the common case. Three sessions
make those two comparisons disagree -- an owner-rooted lead on the default store
dispatches a member-store worker and a default-store conductor, so giver and recipient
agree while recipient and target do not -- and a handoff admitted on the giver's binding
alone would be accepted now and silently revoked at the next start, which is worse than
refusing it up front. The refusal names the mismatch (`recipient_other_store`). The log stays the authority on WHAT was
recorded and was never the authority on whether that record is still allowed: a recipient
can be rebound to another principal's store, linked to a channel or closed while the
gateway is DOWN, and no append-only entry can know it. This is also what closes the gap a
corrective append leaves, because a correction that never landed and a crash between the
entry and its correction both leave the log stating a grant that was refused, and this is
the one point such a grant would otherwise take effect.

`rebuild_managers_from_log` RESTORES `_manager` from the newest entry per slot, through
`session_tree.slot_manager`, which answers `(manager, decided)` so a caller can tell "no
handoff was recorded" from "the log did not say". It is the only way the field is ever
populated at a start.

Its IMPORT is not on the boot path either, which is a separate cost from its run. The
restore schedules the rebuild as its own task, and the `session_control` import sits in
that task's coroutine rather than in the restore function around it: session control is
optional and importing it pulls its whole dependency set in, so an import on the boot
path is paid by every start, including starts with session control switched off that
never reach the rebuild. A lazy-import count cannot see this -- the import was already
inside a function, so moving it changes no name and no total -- so a source test pins
WHICH function holds it.

It is NOT on the boot path, and it is scoped by the index. An indexed slot's read is
proportional to its surviving history -- a unit is only PROVED to record no handoff by
reading all of it -- so `session_restore` runs the pass as its own task rather than in
front of READY, and only indexed slots have anything to read. There is no process-wide
window: every slot whose answer could be guessed wrong is an indexed slot, and every indexed
slot hydrates unresolved and is refused on its own until its read lands, so there is no
state in which an unread value is answered from `_created_by`. A refusal also arms the
read for its own slot, so a caller that arrives first does not wait on the pass. The log decides in three directions, and only two of them
return the session to its creator. An entry-less log and a decided revert (the
`to_creator` entry form) both leave `_manager` empty, which IS the statement "the creator
holds this session". A recorded manager that fails revalidation -- closed, or rebound to
another principal's store -- makes neither statement, so manager-dependent access is
DENIED rather than reverted: "the recipient is not admissible" and "no handoff happened"
are different facts, and reverting would invent a transfer back and hand a revoked grant
to the session that gave it away. An undecided read establishes nothing and leaves the
slot unresolved, so a briefly unreadable store costs a handoff until a read succeeds
rather than handing anyone reach the log cannot show they were given. Denying strands
nobody: an unfenced caller's reach never came from this field, so a person can still
adopt, release or close such a session, and the way back is a fresh handoff.

The verb writes NO slot line, which is the other half of the same decision. The entry is
the handoff and `_manager` is this process's copy of it, so there is no second write to
order, no partial state to report -- the reply carries no `cached` key -- and no cache to
fall out of date. Publishing on the strength of a `best_effort` save instead, back when the
line did carry the field, is what let a save that failed plus a restart before the periodic
flush retried it restore the manager a handoff had just revoked.

A handoff whose commit is REFUSED puts the parent back. `_reparent_target` makes the tree
edge durable before the commit runs, which is the order the record owes -- a field written
first and an append that then failed would leave the gateway acting on a move no record
carries -- and that leaves a window where the edge is on disk and the manager has not
moved. `_landed_then_committed` closes it: a commit that raises triggers `_restore_parent`,
which durably adopts the target back under the parent it had, or RELEASES it when it had
none (an adoption naming an empty parent is not an edge, and the emitter refuses to write
one), or releases it when restoring that parent would CLOSE A CYCLE. A rollback is a
forward write like any other, so `_restoring_would_cycle` checks it against the same
settled-plus-queued view: the window runs after its verb gave up its place in the mutation
lock, so the previous parent can have been reparented under the target meanwhile by moves
each checked and each fine, and the restoration is then the edge that completes the loop.
That check fails closed by answering True, which is the opposite direction from the forward
path, because a compensation has no refusal available -- releasing loses nesting, which is
visible and repairable, where persisting a cycle is neither.

The rollback is ANNOUNCED as well as checked. It takes a `_tree_append_waiter` like a
forward append, which fences the slot and -- on the adoption, the only branch that adds an
edge -- publishes `slot -> previous parent` in `_tree_pending_edges`. Checking the rollback
against the queued edges while not publishing it among them would close one direction and
leave its mirror image open: the next mover's cycle check would read a tree without the
edge this compensation is about to write. The fence and the edge are cleared by the
waiter's own future, so a wait that runs out leaves them standing until the writer settles,
which is the state a later decision most needs to see. Without that the target hung
under the recipient while the giver was still its
manager, so the recipient could not drive it and the giver's own `session_release`
answered `not_parent`. The compensation is bounded and never raises its own error over the
refusal that caused it; a compensation that cannot be written is logged as the one case an
operator has to see. The window runs in its own task behind `asyncio.shield`, so a caller
that goes away cancels the wait and not the work, and `_hold_tree_fence` keeps the slot
fenced for as long as the task runs -- a cancelled caller unwinds out of the mutation lock
while the commit is still settling, so the lock alone would not keep the next verb out.

The caller's wait and the window's are DIFFERENT bounds, which is the same idea applied to
a slow writer rather than an absent caller. A request cannot be held for as long as the
writer may take, so it is answered 409 `tree_write_pending` on `_TREE_APPEND_TIMEOUT`
while the task goes on waiting for the real verdict up to `_TREE_SETTLE_TIMEOUT`. A window
that ended on the caller's bound would end with the edge still queued and the commit never
attempted, which is the split the compensation exists to prevent; so when even the longer
bound runs out, the window compensates on the way out rather than leaving an edge a start
could read as a move that happened.

Because that window outlives the request, the CYCLE CHECK runs against the settled tree
plus every queued, unsettled reparent edge (`_tree_pending_edges`, overlaid by
`_tree_with_pending_edges`). The projection advances only when an append settles, so a
check against the settled tree alone cannot see a move the writer is already holding --
and two such moves can close a loop neither closes on its own. Both reparenting verbs
register their edge, because a queued adoption closes a loop exactly as a queued handoff
does. If the combined view cannot be built the move is REFUSED `tree_not_ready`, on the
rule this whole module applies to an unreadable log: "no loop" concluded from a view you
could not read is a confident wrong answer, not a missing one.

A handoff also fences BOTH endpoints (`fence_parent`) rather than the target alone, and
the window owns the parent's fence so it is released when the work is. That is now a
cheap EARLY refusal in front of the real check rather than the guarantee itself: it
catches the two-node case, where the next move shares a slot with the queued one, and it
catches it without folding a view. It cannot catch a larger loop -- with `D -> A` and
`B -> C` settled and `A -> B` queued, a handoff of `C` to `D` names neither endpoint --
which is why the validator and not the fence is what closes the hole. Refusing a reparent
whose endpoint has an unsettled write is `tree_write_pending`: fail-closed, retryable, and
lasting only until settlement.

Serializing tree mutations until pending writes settle would close the same hole and is
rejected: it pins the loop's serialized region to one slow disk, so a single stalled
writer stalls every unrelated tree write. That is the defect `_tree_pending_slots` exists
to avoid.

The endpoint FENCE is scoped to `session_handoff`. `session_adopt` and `session_release`
carry the same pre-existing hazard on their own paths and are tracked in
[#19253](https://github.com/kirodotdev/KiroCrew/issues/19253). The cycle VALIDATOR is not
scoped that way: it reads every queued edge whichever verb registered it, so a queued
adoption is folded into a handoff's check and the reverse.

The audit and the sidebar broadcast are written from INSIDE the window, not in the
caller's frame. A caller that times out raises through `_audit_denials`, which records
`denied` for that timeout; the window then commits and the reach really does move, so an
`allowed` line written in the caller's frame is never written at all and the SEL is left
saying the opposite of what happened. `_reparent_target` passes one `on_committed` for
both, and each verb supplies its own `after_commit` beside it -- the handoff's is the
sidebar push, and the adoption asks for none. The verb's pre-lock refusals are audited
too: the shared `gate` and the handoff's own recipient resolution both sit inside
`_audit_denials`, so `already_manager`, `member_recipient`, `recipient_other_store` and a
replaced recipient each leave a record, which is what `backend-security-controls`
requires of a permission decision.

The recipient is authorized by `authorize_target` like a target and then tested for
its PRINCIPAL: a `member-` DM key is refused by key alone, and a recipient whose
`memory_store` differs from the giver's is refused `recipient_other_store`. That test
is positive on purpose -- `_store_is_member_owned` answers False for an unreadable or
incomplete record, which withholds member status for an admission and ADMITS one for a
refusal, so the recipient rule reads the two slots' own bindings instead. The recipient is
re-established against the LIVE slot object TWICE, and the second time is the one that
decides: immediately before the publish, with nothing suspending after it. The append
suspends, and the recipient's own routes take its slot lock and not this verb's, so an
agent switch, a store rebind, a new channel link or a same-key reopen inside that
suspension would otherwise move the reach on a verdict formed before the change. The
earlier check is kept only because it is free and spares the log a correction in the
common case. A declined publish refuses 409 `recipient_replaced` and never writes
`_manager` -- but the entry is already the authority by then, so declining to publish is
not enough on its own: the handoff is taken back on the log through
`_revoke_manager_record`, or a start would grant at boot the very reach the refusal
withheld. The parent can then be put back knowing the manager never moved. It shares `_reparent_target` with `session_adopt`, which is where both verbs' tree
checks, id resolution, in-lock re-authorization and same-key replacement refusals live.

`session_adopt` and `session_release` reshape the session tree, so both go
through `authorize_target` like every other verb and add tree checks of their
own. Adopt refuses 409 `would_cycle` when the target is an ancestor of the
caller, and 409 `tree_unavailable` / `tree_not_ready` when the tree cannot be
read. Both also refuse 409 `target_replaced` when the slot under the target
key, compared by object identity after the final authorization, is a
different session from the one captured before the first suspension (a close
plus a same-key reopen); nothing is appended and a retry acts on the current
session. Release waives the self-target refusal (`allow_self=True`), since a
session releasing itself reaches no peer, and refuses 409 `already_root` or 403
`not_parent`. Both wait until the tree append is durable before answering, so a
takeover that never reached the disk is refused rather than reported. Neither
verb is in the conductor or member grant sets (`_CONDUCTOR_DASHBOARD_GRANTS`,
`_MEMBER_DASHBOARD_GRANTS`), and both are in `CHANNEL_AGENT_BLOCKED_TOOLS`.

Each row above is one row of `mcp_dashboard.TABLE` (a `mcp_tools.table.ToolTable`):
the row declares its route, its identity (`"strict"`), its descriptor and its
`_run_session_*` body. The table runs `_session_control_gate` before every strict
row, so a caller that cannot be verified is refused before any request is sent,
and the body finds the verified key in `ctx.caller_key` and sends it on every
session-control request it makes on the caller's authority. When `folder` is
given, the folder-filing reads of `/api/chat/slots` and `/api/chat/folders` carry
the frame's attribution key, and a folder segment the filing creates is written
under the tree-shaping gate's own strictly verified key. The body's dashboard
client refuses a route its row does not declare, so
a row's routes are the complete list of what its tool reaches: the Route column
above, plus, for `session_create`, `session_fork` and `session_revive`, the chat
slot and folder routes that file the new session when `folder` is given.
`test_mcp_call_site_auth_coverage.py` checks every declared route against the
internal allowlists. `SESSION_CONTROL_TOOLS` is read off the rows
(`TABLE.names("strict")`), which is what keeps the gate, the channel containment
list and the registration tests' pinned set naming one set. A new session verb is
a new row in `_session_tools()`, its body, and its schema in
`MCP_DASHBOARD_SCHEMAS`.

**Two verbs here write into another session's conversation: `session_send` and
`session_broadcast`.** Reading returns a transcript tail, stopping cancels a turn
the way the Stop button does, creating opens an empty session, and sending
delivers a message that the target runs as its next turn. Delivery is the
sharpest verb and is bounded accordingly: the body is redacted through
`sanitize_outbound` before it is persisted, it is prefixed with a `[sent by
session <caller> via <verb>]` envelope so the target's transcript can never
render it as something the person typed, and channel agents are blocked from it
outright.

`session_broadcast` is not a second delivery path. It resolves an audience and
then calls `send_to_target` once per target, so every bound above holds per
delivery unchanged — the same gate, the same containment snapshot, the same
queue-drain re-check — and a broadcast can reach nothing a sequence of
`session_send` calls could not. What it adds is the audience rule and the
reporting contract: the default audience is the caller's OWN created sessions
(`broadcast_audience` reads the same `_created_by` field the ownership fence
reads, so it is a subset of what the fence admits by construction), deliveries
are SEQUENTIAL rather than gathered, one target's refusal is COLLECTED as a row
instead of aborting the rest, and the audience is capped at
`MAX_BROADCAST_TARGETS` (50) with an over-cap request refused rather than
truncated — a silently-cut broadcast is one the caller believes reached everyone.
That cap sits at `MAX_SLOTS_PER_CREATOR` and must never fall below it: the default
audience is bounded by the per-creator slot cap and by nothing else, so a lower
broadcast cap would make the default path refuse an audience the caller never
chose. An explicit list is bounded at the ROUTE — both its length and each
element's — before any name is resolved, because resolution is per-name work and
the MCP schema's `maxItems` does not bind a caller that reaches the route itself.
The `mode` argument (`queue` | `steer`) is required with no default, because the
two are different instructions and defaulting either way silently substitutes one
for the other across a whole fleet.

Sequential delivery is what makes a per-delivery bound part of the contract
rather than a refinement of it: one target that never answers is one target that
starves every target behind it, and the steer arm suspends on the kiro-cli RPC
with no ceiling beneath it, so a wedged session can outlast any request budget.
Each delivery therefore runs under `BROADCAST_TARGET_ALLOWANCE_SECS` and is
cancelled on expiry, yielding a `delivery_timeout` row and continuing.

The bound wraps the whole `send_to_target` call. Cancellation can therefore land
at `await asyncio.to_thread(sel)` or `await prewarm_enabled_check()` before
`authorize_target`, as well as at the steer RPC after the text was registered.
The delivery records its own progress across the cancellation boundary with ONE
marker: that it entered the steer await, set before any later await can run. The
timeout handler combines that fact with the original slot:

- Cancellation before authorization never reaches the marker, so the delivery
  provably did not reach hand-over and re-sending the same text is safe.
- Cancellation once the marker is set is ambiguous: inside the steer await and
  after it returned are indistinguishable, and a second marker for the return
  could only be set once this one already was, so it would separate nothing. Both
  rows say the outcome is unknown and warn that re-sending
  could execute the instruction twice.
- A matching `_pending_steers` or queue entry remains useful evidence that the text
  may still run, but empty containers cannot distinguish a never-started delivery
  from one whose consumed state was cleared as it progressed.
- A missing or replaced original slot cannot be inspected safely. The row says
  the outcome is unknown and gives no re-send advice.

No timeout row claims that the message was delivered or that it certainly ran.
The target-level `cancelled` audit remains limited to an authorized delivery;
cancellation at either earlier await records no target permission decision.

A cancelled delivery leaves the pre-RPC audience fence on the slot unpopped,
which withholds that turn's cross-surface reply legs for its remainder:
fail-closed, and the correct
direction for a delivery whose authorization stopped being re-checkable. The
bound's value tracks the per-target allowance the MCP client spends on the same
call (`BROADCAST_TARGET_ALLOWANCE_SECS`), since a per-delivery ceiling above the
client's per-target share would let a full audience expire the one HTTP request
and discard the per-target report the verb exists to produce.

`session_status` is read-only and is the roster half of the same picture. It
answers a question no source can answer alone: live slots know what is RUNNING
but forget a session the moment it is closed or lost with its process; the crew
log's session tree (`session_tree_projection`, an in-memory fold, no I/O) is
DURABLE and gateway-attested but receives its creator edge only when the first
turn opens; persisted transcript metadata records `created_by` at birth, before
that edge can exist, but is a weaker, agent-editable source. The union is the
answer, and every row names the sources that placed it (`crew_log`, `history`,
`live`, joined with `+`).

The row's `status` separates `working` / `queued` / `idle` / `closed` / `lost` /
`unknown`. A tree-backed row absent from the dashboard is `closed` or `lost`: the
roster records that a session the dashboard no longer holds still existed, and
its folded lifecycle says which fate it met. `closed` (carrying `closed_at`) is
the DELIBERATE tab close -- the worker finished and its tab was closed
(`session/closed` with reason `removed`, written by `_close_slot`); nothing is there to re-dispatch. `lost` is
the fail-safe default: a session gone with a turn still open, or whose newest completed turn did not end
cleanly (a deliberate mid-turn close cancels the running turn, whose `turn/completed`
carries an abnormal `stop_reason` and clears `turn_open`), or whose only close
edge came from a process recycle (`reset`) or a `destroy`, or whose fold could
not be read -- the state a patrol re-dispatches. A history-only row is `unknown`:
its metadata proves the session was created, but does not claim whether it
finished or was lost. Splitting `closed` from `lost` is what resolves the
completed-versus-lost conflation #14213 named; the split reads the lifecycle the
status fold already projects, adding no new crew-log vocabulary. The persisted
field is already read by the member ownership boundary in
session-control authorization, so using it for an informational roster row asks
no more trust of it than the existing fence. It does not become crew-log lineage,
and its distinct `source` keeps that visible.

The response reports each durable read independently. `tree` is only the crew-log
fold's quality (`readable` / `incomplete` / `unreadable`); `history` is the
transcript metadata scan's quality under the same three values. Neither field
claims the combined roster is complete when the other source is degraded. An
`incomplete` scan makes its contribution a floor, while `unreadable` means it
contributed no rows. `history_omitted` counts transcript rows the scan itself
dropped at its bound. The UNION's own cut is a third, separate fact with a third
field: when the three sources together exceed `MAX_SESSION_STATUS_ROWS`,
`roster_omitted` carries the exact number of rows dropped, and it is zero when the
union was retained whole. It must never be reported by degrading a source's
quality instead — that blames a read which completed and, since no source omitted
anything, reports nothing cut while rows are missing. The ownership fence applies to all rows and not merely to
the verb: a history candidate is admitted only when its persisted `created_by`
exactly equals the caller, and the workspace boundary is re-applied before its
title is exposed. A session created by another caller is absent.

**Delivery has two authorization moments, and both are enforced.** An idle target
runs the prompt immediately, under the authorization that admitted it. A busy
target QUEUES it, and the queue entry carries the containment that held at
admission (`containment_meta`); the drain recomputes those constraints and drops
any entry for which one is newly held (`newly_held_constraints`), so a target that
gains a channel mirror between enqueue and drain never broadcasts the delivered
text. The mirror half of that snapshot is an identity, not a boolean (a mirror
retargeted A→B keeps "mirrored" true while substituting the audience), and it is
composed from BOTH store accessors the delivery legs read — `get_mirror_link` and
`get_slack_link` — because the first shadows the second: it answers the explicit
`mirror` row alone, while the dashboard's slack-link binds its thread onto a
channel-born slot's own session key without touching that row. Read through the
mirror alone, a thread bound while an entry waited would leave the identity
unchanged and the drain would post into a room the admission never saw; the
composed identity reports it as `mirror_retarget`. That identity is a set of
rooms (`mirror_audience`), and the drain compares it as one: a room the admission
never saw — a retargeted mirror, a thread bound beside it — drops, while a room
that has since gone away (the thread unlinked, the mirror cleared) is a narrowing
that admits, because every room the delivery can still reach was admitted. The re-check is not specific to this module — a human-typed message into a
busy session drains through the same path — which is why it lives at the drain
rather than in each caller (#5911).

**A dropped queued delivery is reported to its SENDER, not only its target.**
`session_send` answers `started: false` when a busy target queues the message,
which on its own says the message will run later; every signal of a later drop —
the retracted queue card, the visible notice, the broadcast — lands on the
target's transcript, which the sender does not read. So the queue entry also
carries the sending slot (`send_origin_meta`, stamped at admission beside the
containment snapshot), and the drop appends a notice to the sender's own
transcript naming the target, the constraint that changed and an excerpt of the
dropped text (`notify_send_origin_dropped`); the SEL row names the sender as the
drop's `origin`. The stamp rides `meta` rather than a consumption callback
because `meta` is one of the keys a queued prompt is persisted with while a
callback-carrying entry is excluded from that write, so a callback would trade
the relay's survival across a restart for a notice that cannot survive one
either. A requeued steer keeps the stamp because the requeue copies the
admission dict onto the new entry's meta. Five cases deliberately produce no
notice: a human-typed entry carries no sender; a session that queued onto itself
already reads the target's own notice; a sender closed while the message waited
has no transcript left, and the SEL row is what keeps that outcome recoverable;
a reused slot key now holding a DIFFERENT occupant (`origin_tab` does not match
the slot's `_tab_id`) is treated as the sender being gone; and a structurally
exempt entry is never dropped at all. The report is
best-effort and does not gate the drop — withholding the message is the
authorization decision, and it must not depend on the notice landing.

**A deliberate discard reports to the sender too.** The drain is not the only
place a stamped entry leaves the queue unrun: a person cancelling its queue card
(`api_chat_slot_queue_cancel`) and a hard stop clearing the queue
(`stop_slot_turn`'s escalated branch) discard it as well, and the sender's last
word is still the `started: false` receipt. Both sites read the entry's stamp and
call `notify_send_origin_discarded` with their cause (`SEND_DISCARD_QUEUE_CANCEL`,
`SEND_DISCARD_HARD_STOP`), which appends a notice to the sender naming the target,
what discarded the message and an excerpt, and saying it will not run. The wording
names the discard rather than a constraint, because neither discard changes an
authorization. The recipient is resolved by the same check as the drop notice
(`_send_origin_recipient`), so the same four cases write nothing: no stamp, a
self-send, a closed sender, and a reused key held by a different tab. The hard
stop reports exactly the entries it takes off the queue, before the settle can
suspend; a send that lands during the settle stays queued and is not reported,
because it will still run. Best-effort here too: the discard
was already decided by the person or the stop.

**The stamp names a session, not a key, and does not survive a restart.** It
carries the sender's `_tab_id` beside its slot key and is omitted unless both are
present, because a slot key does not identify a session: a plain
`get_or_create_slot(name)` mints a fresh slot object on a free key and the
explicitly-named keys are deterministic (`cron-{job.id}`, `workflow-{run_id}`, a
channel's own), so a closed sender's key is handed to the next occupant, whose
link scope and audience are declared per creation. The notice therefore requires
the live slot's tab to equal the stamped one and otherwise treats the sender as
gone, which it is. `sanitize_restored_queue` strips the whole stamp, joining the
containment snapshot and the turn actor: those are read, but this one names a
WRITE TARGET, so a stamp carried back off the metadata line would append the
entry's own text to a session the editor does not own, with nothing that retracts
it. A delivery that outlives a restart and is then dropped reports to nobody while
the delivery itself still survives, which is the price of the stamp living in
`meta`. The channel counterpart, `CHANNEL_RECIPIENT_META_KEY`
(`channel_recipient_meta` / `channel_recipient_of` / `notify_channel_recipient_dropped`,
stamped by `dashboard/channel_handoff.py` for a message a channel conversation
queued into a resumed dashboard session), is stripped for the same reason with a
wider blast radius — it names a conversation on a network surface — and its notice
re-runs the outbound recipient check with the principal the channel authorized on
inbound before anything is sent (see [messaging](messaging.md#a-busy-resumed-dashboard-session-takes-the-slots-own-machinery-discord)).

**`steer: true` asks for a third outcome on a busy target.** Instead of waiting
for the running turn, the message cuts into it (`steer_into_running_turn`, the
same path the dashboard composer's mid-turn steer uses), so a caller watching a
worker go the wrong way can say so while the work is still in flight rather than
after it lands. The result names the outcome: `steered` for an injection,
`started` for a turn begun, and neither for a queued message.

Two properties keep the steer arm from being a hole in the queued arm's checks.
The delivery happens NOW rather than later, which removes the queue's waiting
window and replaces it with a narrower one: the steer RPC suspends on
`stdin.drain()`, and containment can move under that suspension.

- **Containment.** The drain re-check exists because a queued prompt runs later
  than the moment it was authorized. Nothing suspends between `authorize_target`
  and the steer call, so the text is committed against the containment the gate
  cleared. Each of the three outcomes then closes the RPC's own window in the only
  way still available to it. A steer that could not be injected re-runs the gate
  before falling back to the queue, and a target that resolves to a different slot
  OBJECT is refused (`target_moved`) — by identity, not by key string, because a
  session closed and resumed under the same name compares equal while the delivery
  would land on the replaced object. A requeued steer becomes an ordinary queued
  entry and faces the drain's re-check, unexempted (see Provenance), and its
  admission stamp comes from the SEND's gate rather than from reading the slot at
  teardown. That distinction is the whole protection: the teardown runs past the
  RPC's suspension, so a slot read there folds a mirror linked during the
  suspension into the baseline, and the drain then reads the widened audience as
  one the authorization saw. A SUCCESSFUL
  steer cannot be recalled — kiro-cli has acknowledged consumption — so instead the
  containment holding at admission is compared against the containment holding
  after the RPC, and if a constraint newly holds, the turn is stopped. That is a
  cooperative cancel with `escalate=False`: one automatic decision, not a person
  who watched a stop fail to take. It matters because
  `_deliver_cross_surface_reply` resolves the mirror LIVE at reply-delivery time,
  so a mirror linked mid-turn is a real audience and not one fixed at turn start.
  The delivery is still reported as successful — it happened, and an `ok: false`
  would invite a retry that double-sends — with the constraint names recorded in
  the audit trail under `steer_stopped_on`.

  The stop alone would not be enough, because it reacts and the turn can finish
  first: `_deliver_cross_surface_reply` runs from the turn's own completion path and
  resolves the mirror LIVE, so a turn that completes inside the RPC's suspension
  publishes before any post-RPC check resumes, and a sent reply cannot be recalled.
  So the send also RECORDS the containment it was admitted under on the slot, before
  the RPC and synchronously with the authorization, and that record stays for the
  whole turn. The publisher then judges the fence at delivery -- the Slack legs
  through `cross_surface_withheld`, the channel-neutral leg on the one row it
  delivers to (`publication_withheld`, below): it
  compares the containment holding THEN against each recorded admission and withholds
  the cross-surface leg when a constraint newly holds.

  The decision lives with the publisher because that is the only moment it cannot go
  stale. A check the sender runs when its RPC returns says nothing about a mirror
  bound between then and the reply, and the reply is what reaches the channel. So the
  sender records and, on what it can see, stops the turn; whether the reply may be
  published is the publisher's question, answered in the same synchronous moment it
  publishes. A channel conversation resumed into the session records the same fence
  for a mid-turn steer of its own (`dashboard/channel_handoff.py`), without the stop —
  a human's own message clears no containment gate for a stop to narrow — so the
  publisher's question has one answer whoever cut into the turn. Its records are
  keyed by audience and capped at `MAX_PENDING_STEERS` (one per distinct
  containment snapshot per turn, never one per message; at the cap no fence is
  evicted and the message takes the slot's queue instead, refused only when that
  queue is itself full), where a peer delivery's
  are one random token each, popped by the sender on every outcome but a landed
  steer and a cancellation.

  Two consequences worth stating. An ordinary steer costs the channel audience
  nothing: the comparison is exact rather than precautionary, so a turn nobody
  interfered with publishes normally. And the record is TURN-SCOPED -- the teardown
  empties it unconditionally -- so one turn's withheld reply never judges the next by
  an authorization that was never about it.

  The record has a second writer, for the opposite direction. The caller-side
  admission gates -- `refuse_caller_surface`, which every targeted session-control
  verb passes, `_refuse_ineligible_creator`, which a creation (and a fork of the
  caller's own transcript, neither of which passes `authorize_target`) passes, and
  the work ledger's `_caller_key`, which every ledger route passes --
  stamp the CALLER's own containment on the caller's slot the moment they admit it
  (`record_audience_admission`), because what those verbs return -- a peer's
  transcript, the roster of created sessions and their titles, a brief or a ledger
  -- becomes part of the caller's reply and the same publisher resolves the
  caller's mirror live at delivery: a mirror gained, retargeted, or widened by a
  Slack thread between the admission and the publication would hand privately-read
  content to an audience the admission never saw, and a sent reply cannot be
  recalled. Written at the gates and never per verb, so a read path cannot be added
  without it. What is stamped is the containment of the very row the gate JUDGED:
  the owner-DM predicate answers a verdict (`judge_owner_dm` -> `OwnerDmVerdict`)
  whose `admission` is built from the one read its clauses evaluated, and
  `record_audience_admission` takes that object and has no read of its own -- a
  record built from a second read would describe whatever row was live at that
  second read, so a retarget landing between validation and record would be
  recorded as the admitted audience without having been judged, and the publisher
  would find live and recorded equal and publish the private read into the new
  room. Written only INSIDE a dashboard-runner turn -- under the identity
  `_run_chat` publishes on the slot at turn start (`_active_turn_session_key`)
  and retires in the same teardown that empties this record -- because that
  teardown is the only thing that clears it: a channel-born session's own turns
  run in the messaging driver, which sets no such marker and runs no such
  teardown, so an owner-DM conductor calling these tools from its channel turn
  would otherwise leave an entry nothing clears, and a later mirror change would
  make the publisher withhold the mirror leg of an unrelated dashboard reply on
  that slot. Nothing is lost by not recording there: the dispatcher publishes its
  own replies to the conversation the turn came from, and the runner's publisher
  is this record's only reader. ONE entry per distinct audience per turn: the key is the snapshot
  itself (`AUDIENCE_ADMISSION_KEY_PREFIX` + the containment snapshot), so a
  conductor polling a worker every few seconds re-records the same audience as a
  no-op and the record holds one entry per distinct containment state the turn
  passed through -- each a real change the publisher must see -- never one per
  call; there is deliberately no cap, because evicting an admission would turn the
  fail-closed publication gate fail-open. The publisher's question is unchanged --
  the containment holding at delivery is compared against every recorded
  admission, steer or gate, and the cross-surface legs are withheld when a
  constraint newly holds (`mirrored`, `mirror_retarget`, `linked`, ...), while
  the transcript keeps the reply -- but the channel-neutral leg asks it of the ONE
  row it delivers to: `_deliver_cross_surface_reply` reads the binding once
  (`_read_mirror_binding`), judges that row against the record
  (`publication_withheld`, the same comparison) and resolves its transport from the
  same row. Decided on one read and delivered from another, a retarget landing
  between the two -- the mirror-link writer runs off the loop, in
  `asyncio.to_thread`, so nothing orders it against two reads on the loop -- would
  be judged as the admitted DM and sent to the room that replaced it. The Slack
  legs keep asking `cross_surface_withheld` synchronously with their own sends,
  and each also judges the thread it CACHED at turn start as the room it is
  (`slack_publication_withheld`, the same comparison over a binding naming only
  that thread): the cached destination is not the live binding, so a thread
  unlinked mid-turn -- before a read whose admission therefore names no Slack
  room -- is in neither side of the live comparison while still receiving the
  reply; judged as a room no admission saw, it is withheld.
  Same exactness (a mirror that did not move,
  or moved back, publishes), same turn scope. A retarget is also refused outright
  at the caller's NEXT call, since every gate judges the current row; the record
  closes the window between the last admitted call and the reply it fed. It covers
  the newly admitted owner-DM mirror (retarget to a room) and the ordinary
  unmirrored tab (a mirror gained mid-turn) alike. Pinned in
  `test/test_session_control_owner_dm.py`
  (`test_a_read_records_the_readers_audience_so_a_retarget_before_publication_withholds`,
  `test_a_read_by_an_unmirrored_tab_withholds_a_mirror_gained_before_publication`,
  `test_every_admission_gate_records_the_audience_not_only_the_transcript_read`,
  `test_polling_records_one_audience_entry_per_turn_not_one_per_call`,
  `test_a_retarget_landing_during_publication_never_reaches_the_new_room`,
  `test_the_slack_legs_judge_the_thread_they_cached_not_the_live_binding`,
  `test_a_binding_that_moves_after_turn_start_is_judged_per_destination`).
- **Provenance.** Both arms hand over the same text: redacted through
  `sanitize_outbound` and prefixed with the
  `[sent by session <caller> via <verb>]` envelope, where `<verb>` is the tool
  that sent it. So an injected steer can no more pose as human typing than a
  queued delivery can, and a worker can tell a fleet-wide instruction from one
  aimed at it alone -- the same sentence means different things in those two
  cases. This is the
  property the composer's own gate protects by keeping app-authenticated sends
  off the steer path; here an explicit envelope answers it.

  It reaches one level further down. `directive_user_origin` exempts a queue entry
  from the drain's LINKED drop because "the author typed into the session's own
  surface", and deriving that from the slot would be wrong because
  `steer_into_running_turn` has more than one caller (`session_send` is one).
  Provenance is REPORTED by the caller
  (`user_origin`) and recorded per in-flight steer, defaulting to false so a future
  caller cannot acquire the human's exemption by saying nothing. The composer keeps
  it; a peer's steer does not.

A steer is a delivery MODE, not a permission: `authorize_target` decides who may
send, unchanged, and a relay archive (`executor == "remote"`) stays refused with `relay_archive_read_only`
before either arm is reached. `steer` is strictly typed at both entry points (the
tool schema and the HTTP handler) and defaults false, so a caller that omits it
keeps the queue-or-run behaviour.

`session_create` earns its place on its own, not as the front half of a delivery
design: an agent that has just worked out that a job needs its own session can
open it pre-named and bound to the right agent, in the caller's workspace, and
hand the person a key they can read and stop. Without it the person does that by
hand -- new tab, retype the title, pick the agent -- and the two observation verbs
have nothing to point at that the agent itself put there. It deliberately does
NOT seed a first message: that would be delivery.

`session_create` also takes an optional `folder` — a folder id or `/`-separated
human path, resolved with `chat_folder_create`'s `parent` semantics (missing
segments created, behind the same tree-shaping gate) — and files the slot as
part of creation (#6118). The caller's OWN slot is filed the same way with
`chat_folder_file_self` (folder tools, same server): it takes no `session`
argument, resolves the target from the verified caller key, and so can be
granted where `chat_folder_move_session` is withheld — a conductor files itself
in the goal's folder and then creates its workers under `<goal>/<agent>`. Filing is
part of creation rather than a second call (`chat_folder_move_session`), because a
folder deleted between two calls would leave the session unfiled with the create
already done. The handler assigns `folder_id` inside the same synchronous window
that configures the slot, holds `suspend_slots_push` across the whole
allocation-to-persist span (so the slot's first broadcast frame already shows it
filed, and a slot whose birth write fails is never broadcast at all), and
carries the placement in the persist-at-birth metadata, so no caller or client
ever observes an unfiled session and the placement survives a restart.
An unresolvable folder refuses the whole create — nothing exists yet, so refusal
loses nothing — existence is confirmed read-only under the folder-store lock
(`read_folders`) before the allocation, and the move path's Model-B un-hide runs
only after the filing has landed, so a refused create leaves no folder-tree
mutation behind.

Filing into a folder also gives the child that folder's nearest inherited project
directory as its project, as dashboard-native creation does; the workspace and
memory boundary stay the caller's. An invalid folder project refuses the create
with 400 `folder_project_invalid`, and a folder whose project changes while the
creation is in flight refuses with 409 `folder_target_changed`.

`session_create` also takes an optional `agent_kind`, with one value: `template`.
It is the selection namespace for `agent`, and it exists for the one case a bare
`agent` cannot express: a shared template that a crew member of the same name
shadows. Name-first resolution (the kind omitted) selects the member, and a
caller that may not bind that member's private store is refused
`memory_delegation_denied` even when it wanted only the template. The stock
`kirocrew-worker` is such a name on installs where the pre-0.8 enrol-on-mount
agent sync registered it as a crew member with its own V2 store (a binding that
is immutable by design, so no setting undoes it): a conductor running as a crew
member's DM thread on the shared store, or a child a parent conductor created,
could not dispatch it. `agent_kind: "template"` resolves the template in
`resolve_agent_bindings(selection_kind="template")`, so the child takes the
caller's own store exactly as an unshadowed template child does (see "A created
worker receives one execution identity"). On an install with no shadowing member
the template kind resolves exactly what the bare name resolves — the resolver
skips only the alias lookup — so the goal conductor's charter and the
goal-conductor skill always pass it when they dispatch `kirocrew-worker`, and no
conductor meets the refusal first. There is no `member` kind on this route: a bare
name already selects the member, so the value would only restate that
resolution. The kind is selection input, never authority, and every gate on this
route still applies. A stated kind requires `agent` (`agent_kind_requires_agent`,
400), any value other than `template` is refused `invalid_agent_kind` (400), and a
kind that does not resolve is refused `agent_unresolved` naming the namespace
rather than falling back to whoever answers by default. When a kind was stated,
the slot records the namespace the pick resolved in (`agent_kind`), and the
persist-at-birth metadata carries it, so an idle template child of a shadowed
name rehydrates in its namespace instead of re-resolving the bare name
member-first. The MCP tool's reply to a `memory_delegation_denied` refusal names
the two real remedies — re-send with `agent_kind: "template"`, or the owner's own
tab — and says the binding is not a setting, because the refusal's deliberate
terseness had left callers inventing a Settings control that does not exist.

The path walk itself never leaves an empty or duplicate folder behind. When the
`folder` path still has segments to create, `session_create` first posts the
same create with `dry_run: true` against the deepest folder that already
exists. `create_session(dry_run=True)` runs every gate up to the allocation,
including both slot ceilings, and returns `{"dry_run": true}` without minting a
slot or spending a rate-limit token. A refusal there is returned before any
segment exists. The folder endpoint refuses an agent caller (the internal
transport) a sibling whose trimmed, case-folded name is already taken under the
same parent, tested under the folder-store lock (`folder_name_exists`, 409).
When the one colliding sibling belongs to the caller's own principal, the
endpoint returns it instead (200, `"reused": true` on the response only), so a
lost race resolves to the winner's folder in the same request. A twin owned by
anyone else is refused; the walk re-reads the tree once and then refuses rather
than forking a same-name twin. The browser keeps a person's freedom to name two
folders alike.

An app or crew-member agent may nest a new folder directly under the folder its
OWN calling session is filed in (`chat_folders.caller_home_slot`), even when the
person owns that folder, so a conductor the person filed in `Ops` puts its
workers in `Ops/<agent>`. The slot must be the caller's: an app's `_app` must
match, and a member's principal must be the one the chat gate stamped from this
request's verified key. The slot's `folder_id` is read under the folder-store
lock. The new folder is owned by the agent. Nowhere else in the person's tree
opens, and renaming, moving or deleting the person's folders stays refused. A
member's folder list adds that home folder and its ancestors, the path its own
`[FOLDER]` line already shows, so the walk resolves `Ops` instead of missing it.

`session_create` also takes an optional `model` — the model the child starts
on, pinned as the person's own pick in the model dropdown would be (same
`_model_rejected_reason` guard, same pick-generation bump). An id the guard
refuses fails the whole create with `model_rejected`; omitted, the child starts
on the agent's or the global default.

### `session_fork`: a created child that carries a transcript

`session_create` opens an empty session, and the case it cannot serve is the
one that produced this verb: an agent that has spent a long investigation in one
session and now wants to split the follow-up into several, each of which should
START with what was found. Writing the findings to disk and seeding each child
with a pointer is the workaround; the dashboard's own Fork button already does
the right thing for a person, and `session_fork` is that button reachable from
the MCP surface.

**One fork core, two entry points.** `chat_fork.api_chat_slot_fork` (the human
route, `POST /api/chat/slots/{slot}/fork`) is a thin wrapper — slot lookup, slot
cap, App Kit ownership, body parsing — around two shared coroutines:
`resolve_fork_source`, which freezes the parent's memory identity and refuses a
parent whose persisted mode is unrecognised, and `fork_slot`, which does
everything from the transcript snapshot (taken under `_fork_lock` through
`transcript_snapshot.read_consistent_transcript` with the `FORK` rules, the
pending-rewrite flush being the fork's own save; then the rotated-archive rebuild)
through the child's mint, message copy, save and the deleted-source rollback. The
session-control verb calls the same two, so what an agent's fork copies and what
it inherits — agent, model, memory store and mode bound at birth, project,
folder, tags, `forked_from` — is by construction what a person's fork copies
and inherits, except the sidebar pin (last table row). The human route's behaviour is unchanged by the split; the core
takes the request-derived facts as arguments (`request_app`, `origin`,
`count_user_session`, `jev_route_allowed`, the SEL caller and operation) and the
two routes answer them differently:

| Fact | Human route | `session_fork` |
|---|---|---|
| `origin` | `request_slot_origin(request_app)` | The caller's authority, as `create_session` derives it: `CRON` for a cron caller or a cron's descendant, else `USER` |
| `count_user_session` | `True` — a human request-layer session for the pulse survey | `False` — an agent-made session is not one (`test_session_pulse_session_count` sweeps for the literal, so the core takes it as a parameter and the only literal `True` stays in `chat_fork.py`) |
| `jev_route_allowed` | `is_owner_dashboard_request` | `False` — arming a second routed session is the owner's own click, which no agent caller made; the child keeps the parent's pinned `model` |
| SEL operation | `chat.slot_fork` | `session_control.fork`, on the core's rows and on this module's own `_audit` row (`session_fork`), so the two entry points stay distinguishable |
| Sidebar pin (`stamp`) | `_inherit_pin(source.slot)` copies the parent's `pinned` onto the child before its birth save, so the fork stays beside its parent in the pinned group; the response carries `pinned` | Never pinned: `pinned` marks a session as human-owned |

The core's refusals are `web.json_response` objects — that is the module's
refusal shape and its coded sites are what the error-code ratchet
(`test_chat_fork_error_codes`) pins, so the split did not re-type them.
`fork_session` reads the `{error, code}` body back into a `SessionControlError`
(`_fork_refusal`), so a caller sees `value_out_of_range`,
`no_messages_to_fork`, `fork_snapshot_unstable` and the rest under the same
codes the human route reports.

**Authorization is the union of two existing rules, not a third.** A fork
manufactures a session the caller then owns, so the caller must be an eligible
CREATOR: `_refuse_ineligible_creator` runs on entry and again adjacent to the
allocation, exactly as in `create_session`, plus the same switch, unattended
and identification gates in the same order. Copying a source's transcript is a
READ of it, so a `source` that is not the caller goes through `authorize_target`
with operation `fork` — the same verb-level requirement, fence and refusal
codes `read_messages` has (`ephemeral_target`, `workspace_mismatch`,
`linked_session_target`, `not_creator`, and so on; the fork tests assert the
codes match read's, case for case). Those decisions are re-asserted at the act,
not only at admission: `fork_session` hands `fork_slot` a synchronous `recheck`
that it runs immediately before minting the child and again after the memory
bind, before any row is copied. The recheck re-runs the caller's eligibility,
the source's liveness, `authorize_target` for a peer source (so a source that
gained a channel mirror or moved workspace mid-fork refuses with read's own
code), the folder's existence against the committed folder list (so a folder
deleted mid-fork refuses `folder_not_found` rather than filing the child under a
dangling id), and both slot ceilings (so two forks in flight cannot each pass the
entry check and both land over the cap). The caller's trust posture is read
inside the stamp, in the same synchronous window, so a grant revoked while the
transcript was being read is not reapplied to the child. Behind that, the source's workspace is part of the identity
`resolve_fork_source` freezes and `fork_slot` re-compares, which is what catches
a workspace move on the caller's OWN transcript, where no target check runs
(`store_unavailable`, retryable). The caller's own session is the one target
`authorize_target` cannot admit (`self_target`), and it is the DEFAULT source
here: a session copying its own transcript crosses no boundary, so an omitted
`source`, and a `source` that resolves to the caller, both take the self path
and skip the target guard while keeping the creator gates.

**What the child gets on top of the human fork**, applied by a `stamp` callback
`fork_slot` runs on the child before its birth save, so it lands in the same
metadata line as the transcript and is on disk before the slot is broadcast (no
second persistence window; a failed save withdraws the whole child): `title` (a final, user-origin name; else the fork's `↳ Fork of <parent>`, which the background `refresh_forked_title` pass renames from the latest person-typed question when the parent's title was auto),
`folder_id` (else the parent's folder, which the human fork inherits; an
unknown folder refuses the whole fork before any copy, confirmed read-only under
the folder-store lock like `create_session`, and the Model-B un-hide runs only
after the filing has landed), creator attribution (`created_by` = the caller,
plus `_created_by_sid` / `_lineage_minted` as at a create, so the other verbs
reach the child and the per-creator ceiling counts it), and the caller's
session posture — `_trust` and `_trust_reads`, with `_trusted_patterns` and
`_trust_scope` excluded for the reasons "What a created child inherits" gives.
A fork spends the `session_create` rate budget and is bounded by the same live
and per-creator slot ceilings; it is a session the caller manufactured, whatever
it starts with.

**Deliberately not offered.** No `agent`, `model` or `mode` override: a
transcript belongs to the memory boundary it was written in, and `chat_fork`
binds the child's execution identity from the parent's for exactly that reason.
No tail fork (`direction` is fixed to `head`); the human route keeps it behind
`dashboard.tail_fork_enabled`. No `prompt`: the child starts idle, and seeding
it is `session_send`'s job — the same split that keeps `session_create` from
being the front half of a delivery.

### What a created child inherits

Creation copies two different kinds of state, and the split is deliberate.

**Identity** — the child is created in the caller's workspace (the memory
boundary; a child left in `default` would be both a boundary crossing and
unaddressable by its own creator), inherits the caller's agent when none is
named, takes that workspace's project directory as its cwd, and is attributed to
the caller via `created_by` so the per-creator slot ceiling is countable. The
caller's ACP session id is also frozen onto the child at this mint
(`_created_by_sid`), from the live caller handle, so the child's `session/opened`
lineage cites the creator that was live when it was made rather than a
replacement that may take the creator's slot before the child's first turn. The
id is backend-authored, so it is bounded where it is retained: one past
`MAX_ACP_SESSION_ID_LEN` (`kiro_crew/validation.py`, the constant every store of a
backend session id shares) is dropped at mint, never truncated -- the sid
is optional and absent is a legal record. The mint also sets `_lineage_minted`, the
in-memory witness that THIS process stamped both fields; neither the witness nor
the sid is written to the transcript, which an agent's file tools can edit, so a
metadata edit cannot forge the gateway-authored lineage the fenced crew log
records (see `crew-log-emitter.md`).

**Approval posture** — the caller's `_trust` and `_trust_reads` transfer, so a
trusted operator's dispatched worker does not stall on a prompt nobody is
watching. This is the posture `parent_trusted` already gives a `spawn_run`
subagent, which reads the parent's stored `"auto"` policy, so the same delegation
behaves the same whether or not it gets a sidebar tab.
No session-store write happens at creation: the child has no ACP session yet
(`set_approval_policy` no-ops on a missing session), and `chat_runner` already
derives the persistable policy from `_trust` on every session create/resume, so
the subagent spawn gate sees it from the child's first turn.

Two grants are excluded, and the exclusions are load-bearing:

| Not inherited | Why |
|---|---|
| `_trusted_patterns` | Per-command grants ("`npm test` is fine"), not a posture. A pattern is judged against the session the operator was LOOKING at, while a dispatched worker runs model-authored work they have not seen, so the same glob can admit a command the grant was never asked about. Inheriting them also pays for itself in neither direction: with `_trust` set the child already auto-approves via `_slot_is_trusted`, so the list is dead weight, and because `chat_runner` matches patterns independently of `_trust` it changes an outcome only when the operator withheld session trust and approved single commands instead — the case that must keep asking |
| `_trust_scope` | A TTL-bounded, SEL-audited `SafetyOverride` scope, re-checked on every approval. Forking the key would hand a second session a credential whose revocation this path cannot observe, so the child would keep auto-approving after the scope that justified it is gone. An unattended worker that needs one gets its own, from whatever owns its lifecycle |

The posture that transfers is the one held at **allocation**, read off the
re-resolved caller in the synchronous window after the last gate — not the one
read on entry. Creation awaits project and agent resolution, execution identity
validation and optional folder confirmation before the slot exists.
An operator selecting `normal` in any of those windows would otherwise have a revoked posture resurrected by a
create already in flight. Revoking mid-call yields an untrusted child.

Nothing about trust is persisted at birth. The birth metadata carries
`tab_id`, `origin`, `created_at`, `workspace`, `agent`, `project`, `title`,
`memory_mode`, and `folder_id` / `created_by` / `model` when set — no trust field — so a
restart returns the child to interactive along with its creator.

The create's SEL record carries `agent`, `folder_id`, `model` (empty when none
was asked for), and what the child was
born with: `inherited_trust` and `inherited_trust_reads`, present on both
outcomes so `"false"` is positive evidence the posture did not transfer. That is
what makes an auto-approved tool call in a dispatched session traceable to the
creator's posture rather than unexplained.

**Known gap.** Inheritance is transitive — a trusted child is itself an eligible
creator — and revoking the creator's trust afterwards does not cascade to
already-born workers, so one click covers a dispatch tree that outlives the
click's scope. Bounded by the slot caps, by trust being in-memory only, and by
the global Trust picker (no slot selected), which sets `normal` across every live
slot. A per-slot revoke that walks `created_by` is tracked in issue #8589.

`kirocrew-dashboard` rather than `kirocrew-core`, because these tools are not a
capability every session should carry. That server is an **assignable set**: it
is absent from the default agent's spec and loads only for an agent whose own
spec references it, so an ordinary session spends no context on tools it will
never call. The set already holds the chat-folder tools, and the two classes are
granted together on purpose — an agent given the job of organizing sessions is
the same agent that should be able to see what they are doing. A test pins that
bundling so neither half can leave the set unnoticed.

Discovery is not new: `list_sessions` already enumerates the caller's sessions,
and its keys are what `target` accepts.

## Authorization

Deny-by-default, and checked in **one** place — `authorize_target` — for every
verb that takes a target (`stop`, `send`, `close`, `read`, and `fork` when its
`source` is another session), so a guard cannot be
present on one and missing on another. (`session_create` has no target to
authorize; it checks the caller's own eligibility with the same refusals, and
`session_fork` applies both: the creator eligibility for the caller, and the
target guard for a source that is not the caller itself.) Every refusal is recorded in the SEL as
`session_control.<op>` with `outcome=denied`, so an attempt to reach a session
that is out of bounds is visible after the fact even though nothing happened.

| Refusal | Status | Why |
|---------|--------|-----|
| Config switch off (`agent.session_control` explicitly `false`) | 403 | Operator withdrew the capability from every agent at once. Defaults to true — the agent's `kirocrew-dashboard` mount is the grant. **Exception:** a crew member bypasses this switch while `agent.member_dispatch` is true (its default) — recognised either as a `member-*` DM caller key OR as a chat slot bound to that member's private V2 store; see "Member callers" below |
| Caller session cannot be identified | 403 | An unidentifiable caller makes the self-target guard blind |
| Caller is an unattended session (`workflow-*`) | 403 | A `workflow-<run_id>` slot exists only once its originating tab is gone, so there is no owning session to fence it to. **Exception:** a cron slot (`cron-*` caller key) is admitted and fenced by creator ownership instead — see "Cron callers" below |
| Caller is itself incognito, temporary, or app-scoped | 403 | Caller-side isolation — the direction the target-side checks cannot see |
| Caller is an APP-owned cron (`created_by` starts `app:`), a cron whose job cannot be found, or a cron whose authoring session is no longer open | 403 | `app_owned_cron_caller` / `cron_owner_unverifiable`. A cron tab is minted without `app=`, so the `_app` check above cannot see an app's own scheduled job; ownership is read from the JOB instead, and an unverifiable owner fails closed — see "Cron callers" below |
| Caller is channel-linked (`linked_session_key` set) | 403 | The exfiltration direction: a linked caller's conversation IS a channel thread, so a read would hand a private dashboard transcript to that channel's readers. `CHANNEL_AGENT_BLOCKED_TOOLS` keys on the agent identity; a linked slot is a second route to the same surface. **Two exceptions:** a `cron:<job_id>` link, which names the job's own run transcript and republishes to nobody; and a 1:1 DM whose only human is the configured owner (`owner_dm_refusal` answering `""`) — see "Owner-DM channel callers" below |
| Caller's own session is no longer open | 403 | Nothing to attribute the operation to |
| Caller changed workspace while a creation was in flight | 403 | Creation resolves the workspace's project directory off-loop, so it suspends between authorizing the caller and allocating the slot. Both decisions that read the caller's workspace -- the memory boundary the child inherits, and whether the answering agent is bound to that workspace -- are invalidated by a move, and re-deciding the binding here is not available: it needs a config load, which must not run on the event loop |
| Named agent does not resolve to a configured one | 403 | The resolver falls back to the default agent, which passes the workspace check because it is the caller's own default -- so no boundary is crossed, but the created session would store and advertise a name that is not what answers. `ResolvedBindings.requested_resolved` states that contract for callers that store the requested name. Refused rather than rewritten to the effective agent: nothing exists yet, so a corrected name costs one retry, whereas an existing slot keeps its stored name verbatim so a momentarily stale resolution cannot permanently rebind it |
| Caller may not bind a child to the selected member's private store | 403 | `memory_delegation_denied`; one check, on the route every branch of agent resolution has already produced, before slot allocation. Two admissions: the store is the caller's OWN, which requires this process's vouched identity and the caller's durable record to agree, or the caller's delegation lineage is not fenced (`_delegation_lineage_fenced`: an owner-rooted chain is admitted; a cron-, channel- or member-rooted chain, or a broken one, is fenced). Same-store workers remain allowed; unfenced global callers retain member assignment. See "A created worker receives one execution identity" |
| Caller changes history key, agent or memory store during creation | 400 | `caller_memory_changed`; the live caller must still match the identity checked before awaited preparation |
| Target is the caller | 403 | A session controlling itself has no exit |
| Target is unattended (`cron-*`, `workflow-*`) | 403 | A `workflow-<run_id>` slot is display-only and a cron's turns are driven by a schedule. Not exempted for a cron CALLER: a cron may create and drive its own children, never another job's tab |
| Target is incognito or temporary | 403 | Never addressable, matching `list_sessions` |
| Target is app-scoped | 403 | App sessions are the app's, not a peer's |
| Target is channel-linked (`linked_session_key` set) | 403 | Its conversation is mirrored to Slack/Telegram, so reaching it crosses a surface boundary both ways — and its stop cannot be honoured, because the stop path addresses `dashboard:<slot>` while a linked slot's turns run under its linked key |
| Target or caller has an outbound channel mirror (`get_mirror_link`) | 403 | The same boundary reached by the other mechanism. `linked_session_key` marks a channel-BORN slot; a dashboard-born slot given a mirror link republishes its turns to a channel just as surely, and the link lives in the session store rather than on the slot, so the slot-side check reads empty on exactly the session that mirrors. **Caller-side exception, two shapes:** an owner DM whose mirror IS its own conversation, and a dashboard-born caller whose outbound mirror IS the owner's own 1:1 DM (the conductor tab the operator linked to their phone) — the same audience either way, established by `owner_dm_refusal` before either caller-side channel refusal runs; the threadless Slack row every unlinked channel session carries reads as no mirror from the store itself (`SessionMap.get_mirror_link` never synthesizes a Slack mirror without a thread), and a Slack thread bound beside the mirror row (`get_slack_link`) counts as a second audience. **A paused mirror refuses exactly like a live one.** `_has_channel_mirror` reads the binding, never `mirror_paused`: the dashboard's Disconnect row mutes outbound delivery and keeps the binding, inbound from that conversation still routes into the session (`SessionBinder.resolve_inbound` does not read the flag either), and one click on the same row resumes delivery — so a paused binding is a latent audience that returns without any further authorization, and admitting the session while it stands would let a session control peers between two clicks of the same toggle. Fail-closed here means the way back into session control is to **sever** the binding: the menu's `Unlink from X` item (`mirror-unlink` / `slack-unlink`) or an in-channel `/unlink`, both of which drop the row and the pause flag with it (#14068; pinned in `test_session_control_boundaries.py::test_a_paused_mirror_still_refuses`). The exemption reads the binding the same way, so a paused mirror to the owner's own DM is admitted exactly as a live one is — the audience did not change. Whether a dashboard-born session mirroring to the owner's OWN DM is that contained audience was left open by #14084 and is decided by #15288: it is, on the same clauses, with the peer read off the transport's own record of the DM (see "Owner-DM channel callers" below) |
| Target is in another workspace | 403 | Workspaces are the memory boundary |
| Target names no open session | 404 | A mistake, not an authorization failure |
| Title matches more than one session | 409 | Guessing means acting on the wrong conversation |

### Member callers: switch bypass, bounded by creator ownership

A crew member is a **conductor by design**: it dispatches work into worker
sessions it creates, patrols them, and reports back, with no operator
configuration. A member runs in TWO kinds of slot, and both are that operating
model rather than an optional capability, so both are recognised as a member
caller (`_member_caller`, the single predicate the switch bypass and the
ownership fence both read):

- **(a) its pinned DM slot** (caller key prefixed `member-`, created only by
  `POST /api/members/{slug}/thread`); and
- **(b) an ORDINARY dashboard chat slot** (`chat-<n>-<ts>`) whose bound memory
  store is that member's private V2 store — the store a DM slot would be bound
  to. A member agent (e.g. `kirocrew-conductor`) also runs in an ordinary chat
  slot bound to its V2 store, and its whole operating model (`session_create` /
  `session_send` / `session_read_message` / `session_stop` / `session_close` /
  `session_revive`) runs from there, so refusing it in a chat slot would leave the member
  chat-only in the surface it exists to drive. The identity here is the STORE,
  not the key: a store counts iff its config record carries a non-empty
  `owner_member` AND `memory_version == 2` AND that owner is still an active
  agent bound to exactly this store (`_store_is_member_owned`, read from the
  loaded config record — never the on-disk manifest, which would be blocking IO
  at the synchronous fence — and **failing closed** on an unreadable or degraded
  `memory_stores` section). Admission never widens to a private V2 caller whose
  store is not a crew member's. The gate carries the admission it made into the
  inner fence rather than letting the fence re-read the record later — see "A
  member-created worker stays inside its own private memory" below.

Two rules give a member caller its shape:

- **The `agent.session_control` switch does not gate a member caller.** Members
  work out of the box — this is the zero-configuration contract, and it is a
  deliberate trade-off: an operator who turned session control off has NOT
  thereby disabled member dispatch. The operator ceiling on that bypass is
  `agent.member_dispatch` (bool, default **true**). Left at its default it
  reproduces this exactly — the member bypasses the switch. Set to `false`, a
  member caller stops bypassing and falls back under `agent.session_control`
  like any ordinary caller, so an operator who withdrew session control can
  keep member DM threads chat-only without disabling the member itself. The
  ceiling is read at the switch gate (`member_dispatch_enabled()`) and, like
  the switch, **fails closed** — an unreadable config withdraws the bypass
  rather than granting it.
- **A member caller may only act on its own reporting line.** How far that line
  reaches is `agent.session_manager_chain` (bool, default **false**). OFF — the
  shipped default — the line is ONE level: the caller's `manager_of` and nobody
  above it, which is the reach session control grants without this knob, so a
  gateway that never sets it authorizes exactly what it authorized before the
  knob existed. Everything below about climbing the line, and the hand-up on a
  departure, is what turning it ON adds. Read like the switch itself and
  **fails closed** — an unreadable config keeps the line off rather than widening
  it — and resolved ONCE per operation by the caller, then carried, because the
  fence runs inside the close's no-suspension window where a config load must not
  happen.

  Slot creation records `created_by` (the creator's caller key) in the slot's birth metadata;
  it is persisted with the session and rehydrated on restart (both restore
  paths). For the verbs a handoff widens — send, read, stop, close, adopt,
  release and a further handoff — `authorize_target` reads the target's MANAGER
  LINE: its `manager_of` (`_manager` when a handoff wrote one, else
  `created_by`) and then each manager above that, climbing live one hop per
  slot under a shared depth bound. A caller on that line is admitted; every
  other caller is refused (`not_creator`, 403). For every other verb —
  `create`, `fork`, `rename`, `set_model`, `retry` — it still reads
  `created_by` alone, so minting a session against the caller's memory binding
  stays with whoever created it.

  The line is not only a handoff's doing. With nothing handed over it IS the
  creation chain, so a member that created the session that created the target
  reaches the target: the member already drives the middle session, so it could
  already have the target driven by telling that one to, and reading the line
  removes the relay rather than a boundary. One level deep, the line and
  `created_by` are the same answer.

  Every edge of the walk fails CLOSED — an empty hop, a cycle, a hop whose slot
  is gone, a line past the bound — so a line that cannot be stated denies reach.

  This ownership boundary binds **even when the global switch is enabled**, so a
  member never widens to the ordinary caller's reach. It binds identically for
  case (b): the chat-slot member is fenced to its own line, never the user's own
  sessions. Every other refusal in the table above still applies to member
  callers unchanged.
- **A departed manager is read THROUGH when the line is walked**, when
  `agent.session_manager_chain` is on. With it off the walk never runs and a
  departure changes no answer, which is what a gateway without the knob does.

  WHAT THE SWITCH GATES IS THE RECORDING TOO, with one exception, and the line
  between them is whether the fact can be rebuilt afterwards. The departure
  writers are gated: a departure is a statement about the line, its path can read
  the published switch, and the crew log already carries the same statement, so a
  departure skipped while the switch was off is brought back by the boot pass.
  The OCCUPANCY of a key is not gated. A take is the only moment the generation
  is knowable and nothing reconstructs it later — and the moment keys are taken
  in bulk is the restore at startup, which runs before any async reader has
  published the switch. Gated there, every restored session would close with no
  occupancy to key a departure on, the write would refuse because the key's
  occupancy is not its own, and the walk would read that hop as the end of a
  line: after every restart a lead would lose reach over a closed restored
  conductor's workers, which is the case this feature exists for. What it costs
  with the line off is a bounded in-memory table nothing reads.

  When a session leaves the live slot table for good — a tab close, an
  idle-archive sweep, or a history delete — that path records ONE value about the
  departing session: the manager it reported to, read while its slot is still in
  hand. No other session's `_manager` is rewritten, nothing is saved, and no loop
  visits other slots. The walk resolves the departed hop from that record and
  carries on up, so a finished conductor does not strand the sessions it was
  running. `created_by` is never rewritten either.

  A session that reported to nobody records NOBODY, since there is no hop above it
  to resolve to, and the walk reads an entry naming nobody as the END of a line.

  **A session whose OWN published manager has closed is REFUSED, and this is the
  known limit of the walk.** The read of a session's own edge goes through the
  chokepoint, which compares the recorded manager against the session answering
  to that key now; a key with no live occupant answers with nobody, so the read
  refuses (`manager_session_changed`, 409) and the walk never starts — the hop
  below the closed manager is the target itself. The reach the walk adds is
  therefore across INTERMEDIATE closed hops, which the record resolves, and not
  across the target's own closed manager.

  The departure record names the manager above and is deliberately NOT what
  answers this. A record is a statement about a session that no longer exists:
  no live occupant can confirm or contradict it, so an authorization resting on
  it rests on an input that can only be trusted and never checked, which is what
  the chokepoint exists to refuse. Widening the read is the one direction this
  design does not go.

  **The refusal is a delay, not a dead end, and the fix belongs to the WRITE.** An
  edge still pointing at a closed session is a revert that has not happened, and
  the refusal's own message says what clears it: hand the session over again. One
  republished edge and the closed hop is off that session's line. Driving that
  revert from the walk is not available here — the compensation needs the
  previously published manager and its occupancy, which live only in the log, and
  the walk is synchronous and runs inside the close's no-suspension window — and
  it would make any refused caller, including a stranger, the trigger for a
  durable rewrite of another session's edge.

  **An entry answers for ONE occupant of a key and for no other.** A slot key is
  reusable, so "what did this key report to" has as many answers as the key has had
  occupants, and a lookup taking the key alone can hand back a previous occupant's
  answer — to a walk that would then give that occupant's manager reach over the
  sessions the current one opened. So an entry is keyed on the pair (slot key,
  occupancy generation), where the generation is `_ChatSlot._occupancy_generation`,
  handed out at construction and held in memory only.

  **A read resolves the generation from the key's CURRENT occupancy and matches the
  pair exactly.** The occupancy is recorded where a key is TAKEN — in the slot
  constructor, which every path that puts a session at a key goes through: a tab
  open, a resume, a transfer import, a restore at startup. An entry filed under any
  other generation is therefore not something the walk can reach for: not as a
  fallback, not when the exact entry is missing, not under any ordering of writes.
  That property does not depend on every writer having been fenced correctly, which
  is why it is the one the design rests on.

  **An edge naming a key is ambiguous when a DIFFERENT SESSION holds that key, and the
  walk refuses rather than guessing.** An edge carries a slot key and nothing else, so
  if the session behind the key is not the one the edge was written for, the edge no
  longer says what it meant: resolving it to the session there now hands an earlier
  occupant's sessions to a manager that never had them, and resolving it to the earlier
  one does the reverse.

  The test is on SESSION IDENTITY and not on the key having been taken twice. Those come
  apart in the case a person meets most often — a tab closed and reopened is one session
  taking its own name back, and its runtime loads the same native session, so every edge
  naming that key still means it. Counting takes of the name refuses that, which refuses
  every handoff to a session anybody ever reopened. So the entry for a new occupancy is
  born carrying WHOM the key was taken from, as that session's id, and the walk compares
  it against the id of the session holding the key now: equal resolves, different
  refuses, and an id that cannot be determined on either side refuses too, because an
  ambiguous read must not resolve to a definitive answer. The mark survives that
  occupant recording its own departure — who it took the key from is a fact about the
  take, not about one close.

  **The ASKER is compared on identity too, where the line reaches it.** Matching the
  caller's key is a comparison of names, so the last hop — the asker itself — would
  otherwise be the one place the line is read off a string: a new occupant of a closed
  lead's key, handed that lead's old descendants again, presents a chain that agrees
  with itself at every other hop. A queued delivery's stamped identity is the asker
  there, re-asked at the drain. An asker that offers no identity is read key-only, as
  the one-level fence reads one: a stamp written before that field existed carries none,
  and refusing it would delete text whose sender was authorized to send it and still is.
  A creation edge is not compared, because no record carries an identity for one and the
  one-level answer does not compare one either — a re-check may refuse what the
  admission refused and never what it allowed.

  Replacing the earlier entry rather than deleting it is what makes this visible. The
  deletion served a real purpose, stopping a new occupant inheriting an old manager, and
  a tombstone serves it just as well since a tombstone is not a manager. What the
  deletion also did was erase the only evidence the key had changed hands, leaving the
  walk to resolve an ambiguous edge with confidence. The entry count is unchanged: one
  entry per key-generation, under the same bound.

  **While a hop is LIVE that evidence lives on the occupant, not in the bounded table.**
  The departure record is capped, and the cap is reached by ordinary slot churn, so an
  entry can be dropped while its key is still open and still being walked through — and
  an absence there reads as "this key never changed hands", which grants reach instead of
  costing it. A live occupant therefore carries whom it took its key from itself, for
  exactly as long as it holds the key, and the capped record answers only for hops that
  have departed, where the gap it leaves already stops the walk.

  What this costs is reach across a hop whose key is held by a session its edges were not
  written for. That is reach the feature itself added, and losing it moves no boundary.

  **A line not yet knowable is UNREAD, and a line that ended is READ.** These are
  different facts even though neither leaves a manager to follow, and an entry carries
  which one it is. An entry written at the pop is PENDING: the occupant is inside its
  close and what it reports to is not published yet. An entry written on the committed
  side is settled, and one naming nobody means the line really has ended.

  A pending entry — and an absent one, which leaves the same ignorance, whether the
  entry was never written or the bound dropped it — answers "unread", so BOTH
  polarities refuse: a reader is denied reach, and a writer refuses to add an edge
  below that hop. A settled entry naming nobody answers "read to the end", so a reader
  is denied and a writer may proceed.

  The writer's half is the load-bearing one and it is why the two cannot be collapsed.
  Read as a line that ended, a pending vacancy tells the handoff loop guard that
  nothing sits above the closing manager, so the guard permits an edge that becomes a
  CYCLE when the close commits and the real manager lands: hand T to R while M (R's
  manager, reporting to T) is closing, and the committed departure completes
  T → R → M → T, cutting off every manager above M and persisting the cycle to the
  manager log. With an unfenced caller the guard is the only check in the way, so its
  answer is the whole of the defence.

  The guard REFUSES on any line it could not read, and it names which kind of refusal
  it is. A hop inside its own close clears by itself — that close publishes what it
  reported to when it commits or rolls back — so the caller is told 409
  `manager_line_pending` and to retry. A proven loop, a cycle further up the line, a
  line past the depth bound and an evicted record are not answered differently by
  waiting, so they keep `would_cycle`. The refusal is the same in both cases; only the
  advice differs, and neither is a guess about a line nobody could read.

  Eviction therefore costs reach and never a boundary, and making the cache bigger is
  not a fix for anything — it would only narrow a window.

  **The write refuses too, independently.** A departure whose own generation is not
  the key's occupancy at write time records nothing at all. This is belt and braces
  rather than the mechanism: the read would not reach such an entry either way, and
  refusing the write is what keeps the table free of entries nothing can read.

  Why both, stated plainly: an earlier version of this design fenced only the write,
  comparing generations between the arriving close and whatever entry stood under the
  key. Eviction dropped the entry that carried the comparison, and the fence then had
  nothing to compare against, so a close finishing late for an occupant the key had
  moved on from wrote its stale manager back and the walk handed that manager the
  replacement's sessions. Exact matching on the read removes the question rather than
  answering it: there is no entry under the key to be compared with, only an entry
  under an occupant.

  **The edge is marked at the POP as well as written at the commit.** While the key is
  out of the live table the walk stops reading the manager off a live slot, and the
  departing slot's own manager is not published until its close commits. So each path
  with awaits between the pop and the commit (the single-tab close and the
  idle-archive sweep; the history delete has none) writes a PENDING entry under the
  current occupancy at the pop, and the window refuses on that entry rather than on
  the absence of one. What the window costs is reach, for the length of one close, to
  the session that is closing anyway. Nothing about the closing session's own manager
  is published there, so a close that fails and puts the tab back has still handed
  nothing up.

  The record is in memory and BOUNDED by count, oldest-first, with ONE bound: the
  entries live inside the occupancy table, so at most one generation per key is
  resolvable and a key that is retaken or evicted takes its entry with it. A hop it
  cannot resolve STOPS the walk — the hop is not skipped and nothing above it is
  admitted — so an unresolvable line denies reach, which is the reach the fence
  grants without this knob at all. It is held in memory rather than read back from
  the departed session's persisted metadata because that read is disk I/O and the
  walk is called from the fence, which runs inside the close's no-suspension window.

  Bounded by count rather than by liveness on purpose: dropping an entry that no
  live session's `manager_of` names deletes the hop a CHAINED departure still needs,
  since in a line whose middle and upper hops have both gone, nothing live names the
  upper one.

  **WHAT THE CAP MAY NOT TAKE IS THE WHOLE CHAIN A LIVE SESSION CAN CLIMB, not the
  keys a live slot names itself.** The keep set starts at the edges of open sessions
  and then follows the records that are kept, upward, to the end of every line the
  walk could follow. Stopping at the first level was a grant and not a retention
  limit: A creates B and B creates W, A is deleted and B closes, and B's record is
  kept because the open W names it — while A, named only by B's record, is not.
  Ordinary churn takes A out of the occupancy table and out of the eviction shadow
  after it, and a key absent from both reads as a name nobody has ever held, so the
  next session to open under A's name is marked as having displaced nobody, the
  creation edge above B resolves through it, and it inherits the worker A's session
  was running. The chain the walk follows and the chain the cap protects have to be
  the same chain. The set stays bounded by the walk rather than by the table: a kept
  key is one some open session's line reaches, so at most the open sessions times the
  depth bound, and a cycle stops on the keys already collected.

  A consequence worth stating: a retention denial is therefore not reachable on a line
  an open session can walk. It is reached where the eviction happened while nothing
  named the key and the sessions naming it came back afterwards, which is a restore's
  order.

  **Reach across a departed manager DOES survive a restart, and no new durable record
  was added for it.** A handoff publishes `session/manager_set` only once that entry is
  on disk and the entry outlives the session's close, so what a departed session
  reported to is already recorded. The in-memory table was never the storage — it is a
  cache of a read nobody was doing for keys with no live slot.

  The boot pass that fills it is the manager rebuild's own, one step further: the walk
  only ever meets a departed hop that some LIVE slot's manager edge names, so the keys
  worth reading are reachable from the live table by following those edges upward, which
  is the same input the rebuild already iterates. Bounded by the walk's depth per chain,
  by the key cap overall, and read one key at a time off the loop. A key whose log holds
  more than one unit has run under several session ids, which is reuse, so it comes back
  TOMBSTONED — a restart must not forget that a key is ambiguous and start resolving its
  edges confidently again.

  **An absent record and an unreadable one are different answers.** A decided read of
  "no manager entry for this slot" is a fact and is written down; an UNDECIDED read — a
  unit that would not open, a store that could not be listed, a line naming no session,
  a recorded manager that may not be published — writes NO entry and the hop stays
  refused, because the next read may see what this one missed. A gateway that has never
  written a crew log is the first case and not the second: absent is not broken, and
  refusing every slot on a fresh install would be a different bug wearing the same
  clothes.

  Saying "records nothing" was not enough on its own, and the gap it left was the
  writer's. Once the pass has finished, a key with NO OCCUPANCY AT ALL is what the walk
  reads as a line that ended — so a hop the pass could not decide read exactly like a
  hop it read and found nothing above. A reader loses nothing either way; the handoff's
  loop guard, told that nothing sits above the hop, permits an edge that the hop's real
  manager — held by the log and unreadable to this process — turns into a persisted
  cycle. So an undecided hop CLAIMS the key's occupancy and files no entry, which is the
  one combination the walk already answers correctly: an occupancy with no record under
  it is "no usable record of that line", unread for both polarities, not a wait, and
  answered one level by the queue drain so no accepted message is destroyed over it.

  The same combination is what makes the switch itself safe to turn on. The departure
  writers are gated, so a session that closed while the line was off filed nothing —
  and the one write that is NOT gated is the key's occupancy, so that hop reads as "no
  usable record" rather than as a line that ended, for a writer as well as a reader.

  Until the rebuild settles the widened verbs refuse with the base PR's own
  self-clearing reason, and the permission check itself still reads memory only:
  durability is the write and the boot read, never the check, which runs inside the
  close path's no-suspension window.

  Written ONLY at departure, and only from the COMMITTED manager — `_manager` then
  `created_by`, never a staged handoff value — so the record cannot name a manager
  the persisted line disagrees with.

Ordinary (non-member) callers are untouched: they still require the switch.
The exceptions are `session_end_wait`, whose own creator fence (below, "Ending
a wait early") binds every caller class, owner sessions included, and
`session_reload`, whose creator fence binds every caller class the same way.

#### The strict-internal surface admits a crew member in either spelling, not every scoped caller

The five routes sit behind `_require_internal`, which first refuses anything
without a valid `X-Internal-Secret`. On the authenticated branch,
`_private_caller_refusal` captures the caller's canonical execution scope once
off-loop through `internal_memory_scope`, without opening learned memory:

- an **owner / Global-V1 caller** (no store scope) falls through to the handler,
  exactly as the surface behaved before member dispatch existed;
- a **crew member** is ADMITTED in either spelling — a `member-*` DM slot, or an
  ordinary chat slot bound to that member's private V2 store
  (`member_admitted_to_scoped_surface`) — while the surface is reachable for it — `agent.member_dispatch` OR the global
  `agent.session_control` switch — so its request reaches `session_control.py`
  where the creator-ownership fence above does the real gating;
- **every other scoped caller**, including a member while BOTH switches are
  off, keeps the `member_scope_denied` 403;
- an **unavailable or mismatched execution identity** returns
  `member_identity_unavailable` 409.

The gate reads the SAME two switches the switch gate does — `member_dispatch` is
a bypass ON TOP of `session_control`, not a replacement, so a member with
`member_dispatch` off falls back UNDER the global switch rather than out of a
surface the operator left open to everyone. All reads fail closed on an
unreadable config, so the surface can never open wider than the two switches
behind it. The member DM admission does not widen other scoped callers' access
or replace the creator-ownership checks in `session_control.py`.

#### A created worker receives one execution identity

`create_session` captures the caller's canonical execution before asynchronous
project or template resolution. An omitted member inherits the caller; an explicit
member uses its existing stable member/store identity under the ordinary
delegation rules. Template and project choices do not select memory. An explicit
TEMPLATE named by a caller that carries an execution keeps that caller's store and
member identity and takes the template's selection namespace — `selection_kind`
`"template"`, the requested name as `selection_name`, the resolved provider
template as `template_id` — through `ExecutionContext.with_template`, the same
rewrite the subagent admission gate makes for `spawn_run(agent=…)`. The record
therefore says whose memory the child runs on and, separately, what was picked to
run it: a member's child that selected a template is that member's delegate. The
member operating protocol and briefing are withheld from every child regardless,
because no created worker is the member's DM thread — `ContextBuilder` delivers
that desk only where the caller's `member=` argument names it, and the template
namespace is the second, independent reason
(see [memory-skills-hooks](memory-skills-hooks.md)). A caller whose member has
no persisted `member_id` keeps its identity and rules through a different record
shape: its member is named by the selection alone, so the arm keeps that selection
and changes only the template with its own `replace` — `with_template` would leave
the child attributed to no member, which is the shape the spawn gate mints for that
caller. The child's
execution record is published before slot metadata, broadcast or provider startup.
Publication failure retracts an idle empty child and reports the actual failure.

Member scope is not itself the thing that bounds cross-member delegation — the
authorization below is. The existing session-control switches, creator ownership
fence, application scope and approval policy remain independent. Memory identity
failure is explicit and never substitutes Global.
Incognito and temporary children inherit the stricter retention mode.

#### Who may bind a child to a private store

Selecting an agent is a caller-supplied string, so it cannot itself authorize the
private V2 store that agent names. `create_session` therefore authorizes the
child's private binding ONCE, at the single point where every branch of agent
resolution has produced its final route, and before the slot is allocated. Placing
it per branch is what left the surface open: the explicit member selection, the
inherited caller execution and the caller-agent fallback all reach a member store,
so a check on one of them leaves the others.

A private member store is reachable on two authorities and no others:

- the store is the caller's OWN, which needs TWO sources to AGREE: this process's
  own vouched identity for that session, and the session's durable execution
  record. Neither alone is admissible. The record is metadata on the caller's own
  transcript, so by itself it answers a question about the caller with the
  caller's own claim. The vouched identity is written only by
  `bind_session_execution`, which no session can reach, and only when the store it
  publishes was established independently of that record — a publication that
  carries the owner over FROM the record, as a provider template switch or a
  dashboard fork does, does not vouch for it, because vouching a value the session
  chose would make the two sources one. Claiming it is OPT-IN: the default is not to
  vouch, so a binder that says nothing about provenance publishes the record and
  claims no authority, and a caller that should have claimed it fails loudly at a
  refused dispatch rather than quietly widening access. An owner's pick of a member
  for a dashboard tab or member DM thread (`_pin_private_agent_assignment`) vouches:
  every caller authorizes the owner's request first, and the store it publishes is
  the one config resolves for that member, never one carried from the record. A
  member-LESS identity is never
  vouched even when its caller asks: no member means the Global store, and the admission
  identifies its caller by member, so the entry could never be admitted.
  A record published elsewhere can also leave a vouched entry behind.
  Agreement therefore fails closed against a forged
  record and against a stale vouched entry alike. `slot.agent` and
  `slot.memory_store` remain inadmissible, and not only because a later write can
  change them: both are rehydrated from that same record on restore;
- the caller's delegation lineage is not fenced (`_delegation_lineage_fenced`, the
  live `_created_by` walk below): the owner's own dashboard session, or an agent
  chain rooted in it. This keeps the shipped capability: an owner reopening member
  conversations and dispatching member workers, directly or through a conductor
  started in their own tab.

The vouched half is held in this process, and the gateway also writes a copy of each
vouch to `vouched-executions/` at the data-home root. Every sandbox masks that leaf and
agent file tools refuse it, so only the gateway writes it; it is NOT under `trust/`,
which sandboxes keep writable for the audit log. The copy is removed wherever a vouch is
withdrawn on purpose (a privacy tightening, a selection rollback, an explicit clear) and
when the transcript is deleted, but not on cap eviction or restart. After a restart, the
next gate-verified admission re-vouches a member DM key whose slug the durable record
agrees with, or any other key whose disk copy, durable record, privacy mode and the
member's configured store all agree. A session that rewrites its record to name a peer's
store matches neither source and stays refused.

For an operator, recovery after a gateway restart or a cap eviction is normally
automatic: own-store dispatch self-heals at the next gate-verified admission, when a
member DM key agrees with its record and the config, or the durable
`vouched-executions/` copy agrees (`revouch_at_verified_admission`). Only when neither
source agrees does a member session's worker dispatch keep answering
`memory_delegation_denied`; the fallback is the owner re-selecting that member's agent
on the slot, which binds afresh through the durable path and vouches again.
Re-selecting the agent the slot already names is enough: the owner-facing switch
records the selection with `replace`, so it re-binds rather than short-circuiting on an
unchanged choice. Closing a tab is NOT such a trigger — a non-destructive close and an
idle archive both retain a persistent session's vouch, because the conversation is
recreated from the warm pool on resume and the turn-start rebind publishes nothing when
the selection has not changed, so withdrawing there would charge a restart's refusal to
closing a tab. The refusal
the caller sees names neither store nor member, so it is the server-side cause line that
tells an operator a dropped entry apart from a record that disagrees with what this
process committed; a member cannot diagnose which it hit from the refusal alone.
The cause line also names, on its own, the caller class no re-bind can help: a
caller whose own record is on the SHARED store (a Global-V1 crew member's DM
thread, or an agent-created Global-store session under a carried fence verdict)
is never vouched, because `bind_session_execution` vouches only a truthy
`member_id`, so without that line every such refusal would read "no vouched
identity" and point the operator at a restart or cap churn that is not the
condition. The line says the caller holds no private store of its own for the
selected member's to match and names the two remedies (a template dispatch, or the
owner's own tab), while still naming neither the store nor the member.

The vouched half is also BOUNDED, by one named count cap. The population is not the
set of live sessions: every persistent `bind_session_execution` that establishes its
own store vouches, and several
of its callers mint a key per REQUEST rather than per session — a webhook that sends
no `sessionKey` gets a per-second one, a task runner refine run gets one per run — so
uptime alone would grow the map until the process restarted. Each such producer
releases its own entry at teardown where it has one, and the cap is the backstop for
the producers that do not. Passing the cap evicts the LEAST RECENTLY USED entry — a
successful own-store admission refreshes its entry, so recency follows USE rather than
birth and churn from the teardown-less producers falls on idle keys instead of on the
member session still dispatching through its own. Eviction drops only the in-process entry: the next gate-verified admission re-vouches
from the durable copy as after a restart, the same fail-closed direction when the
sources disagree, and it never touches a durable record.
Overflow is counted and reported, so an evicted entry is distinguishable from one
never vouched — both read as absent. A refusal also records its CAUSE server-side —
authority this process does not hold, against a record that disagrees with what it
committed — so an operator can tell a restart-dropped or evicted entry from the forgery
the agreement exists to refuse. The caller's own refusal still distinguishes neither,
and the log names neither store nor member, so it is not a second disclosure channel
for what the refusal withholds. Two bounds are declared, one per dimension the map
adds: the entry COUNT, and the length of each retained STRING, since 4096 rows of an
unbounded field is unbounded. The string bound is the repository's shared bound for
names and ids, and an execution carrying an oversized field is DROPPED rather than
truncated — a truncated identity would compare equal to the session that owns the
shortened form. The retained key needs no bound of its own: the vouch runs strictly
after the durable write, so the map cannot hold a key the record cannot carry.

A member caller (fenced through the carried member verdict) and every caller whose
delegation lineage `_delegation_lineage_fenced` fences are refused with
`memory_delegation_denied` (403): a cron slot, a member DM slot naming a PEER
member's agent, a channel-linked creator, anything any of them created — the
fenced caller's unfenced deputy — and a chain whose lineage cannot be walked. An app-token caller never reaches the route (`internal_secret_required`)
and an app-scoped one cannot create at all. The refusal names neither the store nor
the member, so it cannot confirm a guessed agent name.

The verdict is the one the HTTP gate settled on the caller's verified scope,
carried in as `caller_fenced` exactly as the other routes carry
`precomputed_ownership_fenced`; absent, it is evaluated inline. It is only ever
read as a REFUSAL, so a config record that stops saying "member" between admission
and this check can turn a refusal into an admission the owner already holds, never
the reverse.

A session with native provider context cannot change members in place. An unused
chat may select a member only with selection revision checks covering prewarming,
resume pointers and late provider completion. Interactive creation and
`session_create` use the same execution selection rules.

### The fence propagates to what a fenced caller creates

`_caller_is_ownership_fenced` covers three populations, not two: a member DM slot,
a cron slot, and **anything either of them created**. The third is the one a key
prefix cannot see, and without it the fence buys nothing. A created child is minted
with a plain `chat-` key and INHERITS its creator's agent, so a fenced caller
running a session-control agent would otherwise get an unfenced deputy for free:
create a child, seed it, and the child — an ordinary caller by key — reads any
same-workspace session and reports back through the transcript its creator is
allowed to read.

The case-(b) member is kept FENCED by the decision the HTTP gate already made,
not by re-reading its member status at the fence. The gate admits a member on the
caller's VERIFIED private scope; the inner fence, left to itself, would re-derive
member-ownership from the MUTABLE config record via `_caller_is_ownership_fenced`
→ `_member_caller` → `_store_is_member_owned` — and that read happens after the
body read and the prewarms have suspended. An operator's own config writer can
change the record in that window in ways that are all legitimate writes, not
corruption: un-assign the member (`owner_member` cleared, `memory_version` still
2), drop `memory_version` (the loader coerces a missing key to `1`), or drop the
entry outright (a malformed entry is silently discarded). After any of them the
record no longer says "member", `_member_caller` goes False, and the fence would
collapse to `bool(_created_by)` — so a chat slot with no `_created_by`, with the
global switch on, would reach a foreign same-workspace session it did not create
(a silent cross-session transcript read). Guarding that by classifying the store
more finely does not hold: whatever property of the record the fence keys on, a
config write can remove it.

So the verified admission travels with the request instead. `_private_caller_refusal`
marks the request when it admits a crew member (either spelling), every route reads
the mark back (`_carried_fence`) and hands it to `stop_target` / `close_target` /
`send_to_target` / `read_messages` as `caller_fenced`, which forward it to
`authorize_target` as `precomputed_ownership_fenced=True`. A member admitted as one
is creator-fenced for the whole request, whatever the record says a beat later. An
owner / Global-V1 caller carries nothing (`None`) and its fence is evaluated inline
exactly as before member dispatch existed — so a plain chat tab bound to a legacy V1
NAMED store keeps the owner reach the gate admitted it with; nothing about a
non-`default` store narrows it to creator-only.

The switch BYPASS is the one thing that still reads the record live, at every gate
(`_member_bypass` → `_member_caller`): a member the operator un-assigns loses its
bypass at once and falls back under the global switch. That direction can only
tighten, so re-reading there is correct where re-reading at the fence was not.
`_store_is_member_owned` itself is a single fail-closed boolean — `True` only for a
record with `memory_version == 2`, a non-empty `owner_member`, and that owner still
an ACTIVE agent bound to exactly this store (a deleted crew leaves its store record
behind with `owner_member` set but no agent, and must not keep the bypass); every
other answer, including an unreadable config or a degraded `memory_stores`
section, is `False`, which withdraws admission and the bypass and can never open
the surface wider than it is.

`_created_by` is the attribution marker, and it needs no lineage walk:
`create_session` is its ONLY writer, so a non-empty value means "an agent made
this session" at any depth. `_caller_is_ownership_fenced` reads this marker alone
for the ownership boundary `authorize_target` evaluates on every verb: ANY
agent-created session is fenced there, so a created session reaches only slots it
created itself and nothing an unfenced creator could reach. A person's own tab and
a fork reach `get_or_create_slot` directly and stay unattributed, so ordinary human
use is unaffected.

The one place an owner-rooted agent chain must be LET THROUGH is the
private-member delegation gate in `create_session`: a conductor the owner started
in their own tab has to be able to mint private-member workers, while a conductor
rooted in a cron, channel link or crew member must not. That decision is made at
the gate alone by `_delegation_lineage_fenced`, read once per create and never by
`authorize_target`, so permitting the owner-rooted dispatch never widens the
per-verb ownership boundary. The gate walk climbs the `_created_by` chain LIVE at
each hop: a creator that is now a crew member, carries a channel link, or is a cron
tab fences the whole chain the moment it does -- there is no frozen verdict to go
stale, so a mid-chain takeover cannot leave an "unfenced" answer behind. The walk
ends unfenced only at an unattributed root (the owner's own tab or a fork); it
fails CLOSED on any gap -- a hop whose creator slot is gone, or a chain past the
depth bound -- so a chain whose middle slot has been closed loses dispatch rather
than widening reach. No verdict is stored on the slot and none is persisted: the
owner-rooted answer is recomputed live from the chain at each delegation, so a
restart changes nothing about the boundary.

The same attribution is the one lineage fact the child's append-only crew log
records: its `session/opened` carries `parent {slot, sid?}` -- `_created_by` as the
slot, and `_created_by_sid`, the creator's ACP session id frozen at mint from the
live caller handle -- but only while the slot carries the in-memory mint witness
(`_lineage_minted`); the `created_by` restored from transcript metadata after a
restart serves this fence and never the crew log (see `crew-log-emitter.md`). The
fence above reads the slot; a fold that builds the tree of sessions reads the crew log.

#### A member-created worker can itself dispatch — the nested-conductor design

Case (b) keys member identity on the STORE, and `create_session` binds a member's
child to that member's own V2 store at birth (the execution record published in
`_persist_birth`, admitted by the own-store authority above). So a worker the member
spawned is ALSO on a member store, which means `_member_caller` case (b) is true for
it too: with
`agent.member_dispatch` on, a member-created worker passes the HTTP gate and
`_member_bypass` and can `session_create` its own children — grandchildren of the
original member — without the operator's global `session_control` switch. This is
INTENDED, not an accidental widening: the conductor model is recursive by design (a
`kirocrew-conductor` dispatches a `kirocrew-conductor` child for a sub-goal that
itself decomposes), and depth is bounded elsewhere — the conductor agent caps
nesting (children may conduct, grandchildren may not), not this fence. Every node
in that tree stays bounded by the SAME ownership fence: a worker reaches only the
grandchildren it created itself, never the user's own sessions and never a sibling
worker's, because `_created_by` is checked at each hop. The population the
store-as-identity predicate admits is therefore "a crew member and its own dispatch
tree", each member-store node fenced to what it created — the recursion carries the
operating model down without carrying reach across it.

There is deliberately NO attendance exemption. `_ChatSlot._human_seen` looks like
the right hatch and is not: it records that a human has EVER driven the slot, is
monotonic and persisted, and says nothing about who authored the turn running now.
Releasing the fence on it would hand the creator its deputy back for the price of
the user glancing at the tab once — cron creates the child, the user types into it,
and from then on every cron-authored turn in that child runs unfenced. The question
the predicate can answer is "whose authority is this session", not "is a person at
the keyboard", so a person working in an agent-created session keeps that session's
reach rather than their own.
The member-facing tool surface is the ordinary `kirocrew-dashboard` `session_*`
tool set, mounted **per session** rather than through the on-disk agent
template: a member DM session's ACP `session/new` **and `session/load`** carry
the dashboard server as a session-level `mcpServers` entry (built by
`members.member_dispatch_session_server`, identity via `KIROCREW_SESSION_KEY`
in the entry's env, plus `KIROCREW_BOUND_PORT` — the entry's env is built from
scratch rather than inherited, and a child left to rediscover the port falls
through to the run-marker check, which needs an `lsof` view the sandbox's user
namespace does not have, so a gateway on any non-default port would be dialled
at the default one) — both establishment paths, because `session/load`
re-initializes the session's MCP servers, so a resume that skipped the
injection would strip a member thread of its tools mid-conversation. On the
KAS backend the wire agent projection additionally grants the server in
`tools` plus the member's approval-free dashboard verbs in `allowedTools`
(ceiling-filtered like every other grant): `_MEMBER_DASHBOARD_GRANTS`, the
conductor's set (`_CONDUCTOR_DASHBOARD_GRANTS`: `chat_folder_tree`,
`chat_folder_create`, `chat_folder_file_self`, `session_create`,
`session_read_message`, `session_status`) plus `session_send`,
`session_broadcast`, `session_stop` and `session_retry` — the write verbs are safe to auto-approve for a member *specifically* because the
`created_by` ownership fence above bounds them to worker sessions the member
itself opened. Member sessions also bypass the provider warm pool
(`bypass_member`): a pooled child was spawned with no session key on the
default backend, so a warm hit would skip both the member backend route and
the mount. The member backend is `agent.member_acp_backend` (default `kas`),
and requires a wire-capable backend (`ACP_BACKENDS_MEMBER_DISPATCH`: the
claude seam, KAS, codex, opencode, goose and deepseek); kiro-cli v2 reads its template from disk and
exposes no per-session channel, so a member session on it runs as plain chat —
the tools are simply not mounted, never mounted-and-refused. Codex qualifies
because `providers/mirrors/codex.py` already gives it a per-session array and
its routing is `SESSION_CONFIG`, a member of `ENFORCED_ROUTINGS`, so
`_apply_session_permission_routing` refuses the session outright when
`mode=read-only` cannot be armed — what was missing was the decision, not a
mechanism. `tool_gate.is_enforced` is true for opencode as well, on
`VERIFIED_SEEDED_SETTINGS`: the value is seeded into the child's environment and READ
BACK from the harness's own config resolution before the first prompt, so a session
that cannot establish the asking posture is refused there too, and
`providers/mirrors/opencode.py` documents `permission_surface_owned` as
accepted-and-ignored for exactly that reason. deepseek's routing, `VERIFIED_GATE_EXTENSION`, is enforced too: Crew's gate plugin answers every tool call with `ask`, and its load marker is read back before the first prompt. deepseek has no mirror, so its member array comes from `AcpClient._mirrorless_member_mounts`: only the member mounts, with `disabled` honoured and a narrowed tool withholding its whole server, since no mirror carries a per-tool deny there. An element that cannot start fails the whole deepseek session, so an entry that does not resolve is never placed.

### The crew panel rides the same vehicle

A crew member's DM session mounts a SECOND session-level server, `kirocrew-panel`,
by which the member publishes its own webview: `panel_publish` sends a JSON object
and names a template, and the Members page Dashboard tab renders it. Same vehicle,
same reason. The server is `opt_in` in `agent._MANAGED_MCP_SERVERS`, so no spec
emits it, and the Capabilities editor's "Add configured MCP connection" list is
built from CONFIGURED connections rather than from host-managed opt-in servers, so
it never appears there either. Without this mount the only remaining grant surface
is a hand-typed custom connection colliding with the managed name, and no crew can
publish a panel at all.

The element is `members.member_panel_session_server`, composed by the one writer
`members._member_session_element` that also builds the dispatch entry, so both
carry `KIROCREW_SESSION_KEY`, `KIROCREW_BOUND_PORT` and the managed `KIROCREW_HOME`
override by construction. On the KAS backend the wire projection grants
`@kirocrew-panel` in `tools` plus `_MEMBER_PANEL_GRANTS` in `allowedTools`,
ceiling-filtered like every other grant.

Both panel verbs are approval-free, and the reason is worth stating because
`mcp_panel`'s own module doc forbids an `autoApprove` key on that server. The two
paths differ in exactly the thing that rule is about: an `autoApprove` key is
resolved inside kiro-cli, emits no permission request, and so skips
`hooks.on_tool_call` and the governance ceiling with it, while a grant in
`allowedTools` is filtered by `kas_agents._ceiling_permitted` through
`may_skip_gate_now`, which fails closed. `panel_templates` is a read.
`panel_publish` is a write, and it passes the invariant the dashboard grant sets
are judged by -- a granted verb may create or read, never mutate something that
already exists and is not the agent's own -- because the panel it writes is the
calling crew's own: the server takes no crew or session argument, resolves the
publishing crew strictly from the calling session, and refuses a subagent rather
than walking `/proc` ancestors to its parent's panel.

Two operator switches, asked per server rather than shared. `agent.crew_panel`
(bool, default **true**) is the ceiling, read through `members.crew_panel_enabled`,
which fails closed on a raising read AND on a config that loaded having discarded
the `agent` section -- `load()` coerces a malformed section away and falls back to
the permissive default, so trusting that default would be a fail-open. A
whole-server `disabled` on `kirocrew-panel` withholds the mount too, for the reason
it withholds the dashboard one. Neither switch is inherited from the other server's
answer: an operator who withdrew session control keeps the drawer, and one who
switched the panel off loses only the panel. The grant follows the mount in the
same call (`AcpRuntime._mount_member_panel` returns both), so a switched-off server
is never both named in `tools` and pre-approved on the session that is not mounting
it.


Which code appends the entry depends on who composes the array.
`AcpClient._append_member_dispatch_server` serves the backends whose array the
CLIENT builds — claude's, opencode's and goose's (none is in
`ACP_BACKENDS_ACP_RUNTIME`) — and honours the permission-surface precondition
there for an UNENFORCED routing only: claude's is `SEEDED_SETTINGS`, declared and
not enforced, so `_permission_surface_governed` (this client authored
`settings.local.json`, or a sibling's file passed share validation) stands in for
the read-back this core does not have,
while a harness whose routing is enforced must not be held to a file it never writes.
A runtime-served harness never reaches that helper: codex's array comes from
`AcpRuntime._mirrored_session_mcp`, and `create_session` / `load_session` append the
member entry themselves keyed on a non-empty `member_session_key`, which
`AcpProvider._member_session_key` returns only for a member key on a backend in the
set. The agent spec cannot supply the server instead:
`mirrors.identity.identity_bound_crew_servers` withholds the spec-described spelling
of it, because such an element carries no session identity and would answer
`identity_unattested` to every verb.

Two operator switch-offs bind the mount, and both are asked wherever the array is
composed. Switching the dashboard server off WHOLE (`disabled`) withholds it with no
backend condition: the form has no per-call spelling, so no harness can refuse a call
to a server it was handed, and the `tools` allowlist that keeps a disabled server out
of the spec-described half of the array does not reach an element a composer appends
itself. `AcpClient` reads the projection's `disabled_servers`; `AcpRuntime` asks
`session_mcp.session_mcp_server_is_disabled` on its create and resume paths, through
that reader rather than a projection field because KAS has no mirror to carry one, and
from the spec scope its host actually resolves the agent from
(`overlay_project_scope`). On KAS the member GRANT follows the same answer, since the
widening is approval-free. The resume half matters on its own: `session/load`
re-initializes the session's servers and would otherwise re-mount what `session/new`
withheld. Switching off one TOOL of that server is narrower and is weighed against the
backend: where withholding the server is the whole of its per-tool deny channel
(`mirrors.registry.PerToolDeny.WHOLE_SERVER`, goose today) the mount is withheld
too, while codex refuses the call at permission time and claude's and opencode's
deny rules refuse it inside the harness, so they keep their mounts. A
server the projection could not keep narrowed on THIS session
(`SessionProjection.unhonoured_servers`) is withheld from the mount on every backend. Because the mount is
session-scoped, no other session on the same agent template gains the tools,
preserving the two-part grant for ordinary agents (the switch AND the
per-agent server assignment).

### Cron callers: unattended admission, bounded by the same fence

A cron job's own slot (`cron-<job_id>`, minted at run start by
`ensure_cron_slot` so identity exists while the turn runs — with
`inject_cron_result_to_dashboard` as the idempotent delivery-time fallback
creator, #8336) is admitted to the surface even though nobody
is watching it, so a scheduled run can enumerate work and dispatch a session per
item. Three refusals had to move for that, and one deliberately did not:

- **The unattended caller refusal now covers `workflow-*` only.** What must not
  happen is a scheduled job reaching the user's OWN conversations, which is a
  question about scope, not attendance — an unattended job already starts a turn
  in the session that owns it every time it delivers with
  `send_message(session="origin")`. A cron can be held to that scope; a workflow
  result slot cannot, because it is minted only once its originating tab is gone
  and so has no owner to fence it to. Membership of `UNATTENDED_SLOT_PREFIXES` is
  the fail direction for any prefix added later: a new unattended surface is
  refused as a source until it is given a fence of its own.
- **A `cron:<job_id>` link is exempt from the caller-side channel-link
  refusals.** Those exist for channel links, where a read lands in front of a
  Slack or Telegram audience. A cron tab's link names the job's own run
  transcript and republishes to nobody. The TARGET-side refusal is not exempted.
- **The `created_by` fence binds a cron caller exactly as it binds a member**
  (`_caller_is_ownership_fenced` is the single predicate both admissions and the
  fence read, so they cannot drift). A cron reaches the sessions it created and
  nothing else, fail-closed on an unowned slot. `unattended_target` still stands,
  so a cron cannot reach another job's tab. A slot a script cron opens through
  the chat routes carries `_created_by` as well, set to its `cron:<job id>` key,
  so that slot reads as the job's and never as the user's own tab, and the chat
  routes apply the same fence: a `cron:` caller that names another creator's
  slot, live or persisted, is refused with `not_creator`.
- **The global switch still gates a cron.** Unlike a member, a cron gets no
  bypass: the switch is the user's statement that agents may open and drive
  sessions at all, and a job running while they are asleep is the last caller
  that should be exempt from it. A script cron that presents its `cron:` key
  meets the same refusal on the two chat routes it writes to. The
  `agent.session_control` entry under Configuration states that rule and its
  scope.

**An APP-owned cron is refused, and ownership is read from the job.** This is the
one place admitting a cron would otherwise open something. `_app` is how every
other isolation decision recognises an app, but `inject_cron_result_to_dashboard`
mints the cron tab WITHOUT `app=`, so an app's own scheduled job arrives with
`_app == ""` and would pass the check beside it. An app could then create a
persistent, sidebar-visible session that is not app-scoped, which is exactly the
confinement escape the `_app` refusal exists to prevent, reached through the app's
cron instead of its session. `_app_owned_cron_refusal` therefore reads ownership off the job, which has **two
spellings** because two writers record it differently: the app cron SDK tags
`created_by = "app:{app_name}"`, while `mcp_cron`'s own `cron_add` records the
calling session in `session_key` and never writes `created_by` at all — so an
app-scoped session's job carries its authority only in the second. Both are
checked, and the second delegates to `_app` on the owning slot (resolved through
`caller_slot_key`, not a naive `removeprefix`) rather than re-deriving app-ness, so
there is one definition of "is this an app". **A new job field that can name a
principal is a hole until it is added to that function.** The refusal code is
`app_owned_cron_caller`, distinct from `app_scoped_caller` because callers render
that one with app-session wording that would misdescribe a cron. A job the registry
cannot produce, or a registry that cannot answer, refuses with
`cron_owner_unverifiable`: "could not verify the owner" must not read as "has no
owner", and nothing legitimate is refused by it because a cron whose job is gone is
not running.

A job whose `session_key` names a session that is no longer open also refuses with
`cron_owner_unverifiable` (#8335). Its `_app` lived only on that slot, because
`cron_add` records the caller in `session_key` and never writes `created_by`, so
allowing it would let an app create a cron, close its session, and have the job
dispatch a persistent non-app session this gate can no longer recognise. The cost is
accepted, not closed: a genuinely user-created dispatching job whose authoring tab
has since been closed is refused too. The operator recovers one with
`kirocrew cron adopt <id> --release` (or `--session-of` an open non-app tab); a
released job carries no `session_key` and passes this gate. Restoring dispatch for
such jobs without reopening the escape needs an ownership record the caller cannot
influence, written at create time, and is tracked in #8449 together with #13583.

Applied at both
caller-side sites so the two halves stay mirrors, and scoped to cron callers so no
other caller pays for the lookup.

In `authorize_target` this refusal sits **before** `_resolve_slot`, unlike the other
caller-side refusals. A caller refused for its own identity must learn nothing from
the attempt, and resolving first makes the refusal an existence oracle: a guessed
target answers `target_not_found` (404) when it does not exist and the refusal (403)
when it does, so a caller allowed to touch nothing could enumerate the user's
session keys and titles by the shape of the error. The unattended prefix gate is
already on that side of the resolution for the same reason. The pre-existing
caller-side block below the resolution (`app_scoped_caller`, `ephemeral_caller`,
`linked_session_caller`, `mirrored_caller`) has the same shape and is left as it is
here: moving those changes refusal precedence for callers that exist today.

A session a cron creates is tagged `SlotOrigin.CRON`, not `USER`. A cron's own
slot carries that tag so its output stays outside the `slots:user` WS scope, and
a USER-labelled child would hand it that exposure by the route of creating a
session and writing there instead. The tag follows the caller's AUTHORITY rather
than its key prefix, for the same reason the fence does: a child inherits its
creator's agent, so a cron's child can itself call `create_session`, and a
prefix-only test mints THAT grandchild `USER` because its caller key is a plain
`chat-`. `create_session` therefore reads the caller slot's own `_origin` as well,
which carries the tag transitively to any depth. Only app tokens are filtered by
origin (`_serialize_for_client` returns the unfiltered payload to a dashboard
user), so a CRON-origin descendant stays in the sidebar exactly as a cron tab does.

Capability remains bounded per agent, which the slot-key prefix could not see:
`@kirocrew-dashboard` is an opt-in per-agent server, absent from the default
agent's spec, so a job whose agent does not mount it never has the verbs at all.
A cron whose fan-out must run without an approval prompt needs the write verbs in
its own agent's `allowedTools`; `_CONDUCTOR_DASHBOARD_GRANTS` deliberately
withholds them, because a conductor agent also runs in dashboard sessions where
no ownership fence applies.

Two notes on scope:

- **Only sessions the dashboard currently holds are addressable.** A closed tab
  is out of reach on purpose — waking one would resurrect a conversation the
  user put away. This is narrower than `list_sessions`, which also lists history.
- **Every target-taking tool is on `CHANNEL_AGENT_BLOCKED_TOOLS`, including the
  read.** A channel agent is contained to channel posts, and session control
  crosses that boundary in both directions: a stop or close reaches the user
  through one of their dashboard transcripts, and `session_read_message` pulls a
  private dashboard conversation into a channel other humans can see. Containment
  is about what crosses the boundary, not about who writes, so the read is
  blocked alongside the rest. `session_create` earns its place for a different
  reason: it writes nothing into an existing conversation, but it puts a
  persistent, sidebar-visible session outside that containment.

All these tools additionally require a **signed** caller identity
(`_resolve_session_key_strict`), not the lenient `/proc` ancestor walk. A
subagent spawned by `spawn_run` lives under its parent slot's process tree, so
the walk resolves it to the parent — and since authorization here is entirely
"what may this session reach", that would let a subagent read or stop the
parent's sibling sessions. A caller the gateway issued no key to is refused with
an explanation rather than silently borrowing one.

The routes are **strict-internal** (`_STRICT_INTERNAL_API_PATHS`): loopback plus
`X-Internal-Secret`, with no cookie fall-through. No browser calls them, and they
are the entry point to opening, stopping, and reading another live conversation —
a cookie path there would be a new authorization surface rather than a
convenience. The MCP process holds the secret; an agent's own sandbox does not
(the gateway mints the secret in-process and never puts it in an agent's env),
which is why these are tools rather than something an agent can curl.

Each handler **re-asserts** `request["internal_auth"] is True` rather than
trusting the path classification. Strict is not self-enforcing at the handler:
with the header absent the middleware falls through to cookie auth, and a
`local_only=False` deployment reclassifies strict paths as mixed. Because these
routes authorize on the `X-Session-Key` the caller supplies, a same-origin page
holding only a dashboard cookie could otherwise act **as** any of the user's
sessions. `internal_auth` is set only after a constant-time secret match, so one
check closes the cookie path, the app-token path, and the non-loopback
reclassification together. The same reasoning is why
`/api/computer-use/frame` re-asserts it.

The config read fails **closed**: `KiroCrewConfig.load()` raising resolves to
disabled even though the field's declared default is enabled, so neither a
malformed unrelated section nor an unreadable setting can produce cross-session
reach.

### Owner-DM channel callers: the audience predicate

The caller-side channel refusals (`linked_session_caller`, `mirrored_caller`) and
the work ledger's `channel_session` refusal exist for one threat: a channel
session acts on words from a thread other people are in, and what it reads lands
in front of them. `session_read_message` would hand a private dashboard transcript
to a Slack or Discord audience that was never party to it; a work brief would
publish a private dispatch's acceptance bar there. That reasoning assumes an
audience distinct from the operator. A **1:1 DM whose only human is the configured
owner** has none — the "audience" the containment protects is the operator
themself — and refusing it made every Discord and Telegram conversation a session
that could dispatch nothing. Two shapes reach that audience: the channel-born DM
session itself, and a **dashboard-born session whose outbound mirror is that DM**
— a conductor tab the operator linked to their own phone, whose words come from the
dashboard and whose readers are exactly the mirror's. Refusing the second while
admitting the first (#15288, the question #14084 left open) cut a dashboard
conductor off from every worker it had dispatched the moment it was mirrored:
`session_send`, `session_read_message`, `session_create` and every `work_ledger_*`
call refused, with the fleet's workers still running and nothing able to seed,
read or accept them.

Three gates decide this, and three different facts are available to them — the
live `linked_session_key`, the key prefix (`is_channel_session_key`), the mirror
store. A key prefix can never be cleared while a link can, so gates keying on
different facts would disagree about one slot. They therefore consult **one
predicate**, `session_control.judge_owner_dm(state, slot)`, whose answer is an
`OwnerDmVerdict`: `refusal` (the first fact that FAILED, `""` when none did, which
is the admission -- `owner_dm_refusal(state, slot)` is this field alone, for the
re-checks that need only the reason), `mirrored` (the fail-closed reading of the
same row, for the `mirrored_caller` refusal) and `admission` (the containment of
that row, the object the gates record). The ledger's entry gate judges the slot
`caller_slot_key` resolves -- the same resolution every session-control verb uses
-- so "the ledger gate and session control agree on the same slot" holds by
construction (`session_owner_dm_refusal(state, session_key)` is the reason-only
form its post-await re-checks use). Every field of the verdict comes from ONE read
of the slot's binding (`_read_mirror_binding`: the mirror row and the Slack
binding, the two accessors the delivery legs read); no gate probes the store a
second time to decide what the first read already saw, and the audience it records
is that read -- the reason a judged row and a recorded row can never differ. There
is no separate boolean face, because the only consumers are the three gates and
each of them needs the reason, not a verdict. Every gate renders that reason into
its refusal — the mirrored refusals too, since the mirrored shape can now fail a
clause — so the three tell a caller the same thing about the same slot and none
of them can name a clause the predicate did not actually evaluate. The refusal
CODES are unchanged (`linked_session_caller`, `mirrored_caller`,
`channel_session`). Both shapes are admitted on the same positive facts — a
verified surface, a live roster naming exactly one owner who is the
conversation's peer, no wider room bound beside it — and differ only in where the
conversation and its peer are read from. The predicate is a conjunction of those
facts, and any it cannot establish answers **false**.

**The channel-born shape** — the slot's `linked_session_key` is a channel key (a
`cron:<job_id>` link is not):

1. The key parses under the canonical grammar (`messaging.link.parse_session_key`)
   as a **direct** conversation with exactly one peer, on a surface in
   `OWNER_DM_CONDUCTOR_SURFACES`. A `group`/`forum` key names a wider audience; a
   `unified` bucket names no peer; the legacy two-segment Slack shape does not
   parse; a surface outside the set fails closed by construction.
2. The channel's **live** transport names exactly one owner and it is that peer:
   `messaging.transport.sole_direct_target(transport.configured_targets())`, the
   same one-identity rule `/sessions` and the proactive owner DM apply, shared with
   `_owner_dm_target` so the two surfaces name the same human. An allow-list is a
   list of people permitted to talk to the agent, not a claim that any of them is
   the operator, so two entries name nobody. Read off the transport (the roster in
   force now, reloaded live, an in-memory read that stays callable from
   `close_target`'s no-suspension re-check) rather than the config record. An
   absent transport means the channel is not running and refuses. This clause is
   one helper (`_sole_owner`) both shapes call, so they cannot drift on who the
   owner is.
3. The outbound mirror, if any, **is** the conversation the session lives in. The
   dispatcher binds the DM as its own mirror on every turn, and that is the same
   audience — but the dashboard can retarget a mirror at any thread or channel,
   and a retargeted DM republishes what it reads to people who are not the owner.
   So the mirror must equal the **origin** conversation the dispatcher recorded
   (`SessionManager.get_origin_link`, written on every inbound turn beside the
   mirror bind — Discord always did; Telegram now records it too). It is compared
   as a whole `ChannelLink` because Discord's DM channel id is not the peer id,
   so no derivation from the key could stand in for the recorded truth. An
   unknown origin, an unreadable store, or a mirror aimed anywhere else refuses.
   A **paused** mirror (the dashboard's Disconnect row) keeps its binding and
   reads exactly like a live one — the audience did not change. An **unlinked**
   DM reads `None` and is admitted: the dispatcher's first turn stamps the
   conversation's namespaced bucket into the legacy `slack_channel_id` field with
   no thread, `clear_mirror_link` pops only the `mirror` row, so `!unlink` /
   `/unlink` (which also persists the opt-out that keeps the next turn from
   rebinding) and the dashboard's mirror-unlink both leave that threadless row —
   and `SessionMap.get_mirror_link` filters it **at the source**: a Slack row that
   names no thread is bookkeeping nobody can deliver through (an empty `thread_ts`
   never enters Slack's thread index) and is never synthesized into a mirror. One
   filter in the store rather than a copy of the rule in each reader, because the
   readers cannot all be enumerated — this clause and `bind_origin_mirror` each
   carried one, and the `!sessions` resume, the link projection and the
   containment probe read the same method without one. A Slack mirror
   that names a thread is a real second audience and refuses — and it is read
   **through `get_slack_link`, not only through the mirror**: `get_mirror_link`
   returns the explicit `mirror` row whenever one exists and never looks at the
   Slack fields beside it, while the dashboard's slack-link writes its thread onto
   the slot's *effective* key (`DashboardState.link_slack`), which for a
   channel-born slot is this very session, and the turn path posts every
   dashboard-driven reply into that thread straight off `get_slack_link`. So a DM
   whose mirror row still equals its origin can carry a Slack thread the mirror
   read cannot see; a non-empty `thread_ts` refuses on its own clause ("the
   session also mirrors to a Slack thread"), and the threadless bucket stays no
   mirror.

   **The origin does not survive a gateway restart, and the exemption goes with
   it.** `set_origin_link` holds the record in memory by design (`session.py`),
   while the slot and its mirror are persisted and the slot is re-surfaced at boot.
   So between a restart and the owner's next channel message this clause is the one
   that fails while the other three hold: a monitor-loop cycle or a turn taken in
   the surfaced tab is refused, and the conductor loop resumes on the next inbound
   DM message, which records the origin again. This is a deliberate fail-closed
   cost, not an edge case — a mirror with no recorded origin cannot be told apart
   from one the dashboard retargeted, and admitting it on the strength of a
   persisted binding alone is exactly the retarget the clause exists to catch. It
   is also the one refusal a correctly configured owner DM can still meet, so all
   three gates name it (`ORIGIN_NOT_ON_RECORD`) and say what clears it, rather than
   reporting a channel link the caller cannot do anything about. Making the
   exemption survive a restart means giving the origin (or an equivalent durable
   record of the conversation a session was born in) persistence in `session.py` —
   out of scope here, since that store serves the auto-compact notice, whose own
   reason for being in memory is that a restart takes the live session with it.

**The dashboard-born shape** — the slot has no channel link, and the mirror row
`SessionMap.get_mirror_link` holds for its effective key is the subject:

1. The slot is an **attended chat tab** — a person's own, or one an agent created
   — and not an unattended slot (`UNATTENDED_SLOT_PREFIXES`). A cron tab's `cron:`
   link is not a channel link, so it reaches this shape with a plain key, and a
   cron slot is otherwise an admitted (fenced) session-control source; but the
   audience argument rests on the owner typing the words the mirror republishes,
   which no scheduled run's tab satisfies, so such a tab mirrored to the owner's
   DM keeps today's mirror refusal (a workflow result tab is refused as a source
   before the predicate runs).
2. The slot **has** an outbound mirror. An unmirrored dashboard tab is not the
   owner's DM — it is nobody's channel conversation — and no gate consults the
   predicate for it, since the gates ask only about a linked or mirrored caller
   (`_has_channel_mirror` answering false admits it as the ordinary caller it is).
3. The mirror is on a surface in `OWNER_DM_CONDUCTOR_SURFACES` and **names no
   thread**. The Slack link the store synthesizes for a threaded Slack row fails
   the first test; a Telegram forum topic fails the second. A Discord thread
   carries its snowflake as the channel id with no thread id, so it falls to the
   peer clause instead — a guild room is never on record as anyone's DM.
4. The channel's **live** transport names exactly one owner (clause 2 of the
   channel-born shape, the same `_sole_owner` call) **and places the mirror's
   conversation as a DM with exactly that person**:
   `transport.direct_peer_of(mirror.channel_id)`, the contract hook described in
   [messaging](messaging.md). Nothing is derived from the conversation id itself,
   because whether it equals the peer is a per-platform fact only the transport
   knows: a Discord DM link persists the channel id `create_dm_channel` returned,
   unrelated to the user snowflake, while a Telegram private `chat_id` IS the user
   id — attested only when the allow-listed id is a positive integer, since a
   group or supergroup `chat_id` is negative and the allow-list is operator-edited
   text. The record the transport answers from is the one it writes when it opens
   the DM — the dashboard's connect row resolves a `user:<id>` target through
   `resolve_configured_target`, which is that call — or when an authorized message
   arrives from it, and on Discord that record lives in the process
   (`cached_dm_recipient`, the pairing the mid-send re-check also decides on). The
   hook answers a peer or nothing, so one refusal covers the two things "nothing"
   means and does not pretend to tell them apart: a conversation the transport
   would not place as a DM at all (a guild room or thread, a Telegram group or
   forum, a chat outside the roster the transport opens DMs for), and a Discord
   DM whose pairing a gateway restart dropped, until the DM is re-opened or its
   peer writes into it. That is `MIRROR_PEER_NOT_ON_RECORD`, the counterpart of
   `ORIGIN_NOT_ON_RECORD`: it names both causes and states the remedy for the DM
   it may be and for nothing else. Every other outcome is the roster clause's own
   refusal: a DM the transport DOES place, with a peer the roster does not name as
   its sole owner — a stranger's DM, or the owner's DM on a roster that names two
   people — refuses with the same words the channel-born shape uses.
4. No Slack thread is bound beside the mirror row, read through `get_slack_link`
   on the slot's own effective key, for the reason clause 3 of the channel-born
   shape gives: `get_mirror_link` answers the `mirror` row alone when one exists.

There is no origin clause for this shape because there is no origin: the slot was
born in the dashboard, and the dashboard is where its words come from. The retarget
the channel-born origin clause catches is caught here by the peer clause — a mirror
moved to a room or to a stranger's DM stops naming the sole owner — and a retarget
while a session-control call is in flight is judged at the **next** call, since
every gate consults the current row and never an earlier admission, and the ledger
re-checks after each read it awaits across (the same post-read re-check the
channel-born shape gets). The one window a next-call judgement cannot close is the
reply itself: a transcript read under the owner-DM mirror becomes part of the
reader's reply, which its turn publishes to the mirror later, resolved live — so a
retarget landing between the read and that publication would put the transcript in
front of the new room. The caller-side admission gates therefore record the
caller's containment (`record_audience_admission`) -- the containment of the row
the predicate judged, carried on its verdict, never a second read -- and the
publisher withholds the cross-surface legs when it has changed (the audience-fence
record above: `publication_withheld` on the one row the channel-neutral leg
delivers to, `cross_surface_withheld` for the Slack legs). The drain-time
re-validation of queued prompts (`containment_snapshot`) is untouched: it judges
the TARGET's containment, and a dashboard conductor's own mirror was never one of
its constraints. A paused mirror to the owner's DM is admitted exactly as a live
one is, for the reason the paused origin mirror is: the predicate reads the
binding and never `mirror_paused`, and the audience did not change.

**What is relaxed.** An admitted owner DM — either shape — may `session_create`,
and may `send`, `read`, `stop` and `close` the sessions its reach covers, and may
hold a work ledger — the whole conductor loop. Holding a ledger needs one more
thing than the gate: the ledger is a projection of the crew log and appends every
write to the acting session's log, refusing (`crew_log_unrecorded`) when there is
nowhere to append, so the Discord and Telegram dispatchers open their own sessions'
crew logs ahead of each turn exactly as the dashboard runner does
(`messaging.dispatch.open_turn_crew_log`, see [messaging](messaging.md)); a
dashboard-born conductor's log is opened by the dashboard runner as it always was.
**What is kept.** The channel-born DM is creator-fenced:
`_caller_is_ownership_fenced` treats every non-cron channel link as fenced (the
only linked caller that gets past the refusals is an owner DM), so it inherits a
crew member's reach, not the owner's own tab's. A wrong audience inference
therefore costs the sessions the DM created and never the person's other
conversations — `session_read_message` on an arbitrary private tab, the disclosure
the containment was written for, is still refused (`not_creator`, worded "an
owner-DM channel session can only control sessions it created itself"). The
dashboard-born shape changes the fence in **neither** direction: the exemption
waives the two channel refusals and nothing else, so a person's own `chat-*` tab
keeps the reach it had before the link (its authority is the owner's dashboard
session — the words that drive it are typed into the dashboard by the authenticated
owner, and the mirror the predicate admitted changes who reads them, the owner
alone, not who authored them; fencing it on the mirror would take the owner's own
reach away for the price of linking a tab to their phone, including the
`memory_delegation_denied` refusal a fenced caller meets when it dispatches a crew
member's worker), while a tab an agent created stays fenced by its `_created_by`
mark whether or not it mirrors. Group and thread sessions on every channel stay
refused by all three gates; every channel outside the set stays refused; every
other mirror — a guild channel or thread, a Slack thread, a group, a DM with anyone
but the sole owner, a DM the transport cannot place — stays refused with today's
codes; `channel.CHANNEL_AGENT_BLOCKED_TOOLS` (the multi-agent Channel feature's
permission-request block) is untouched; the target-side refusals are untouched, so
a channel session, and a mirrored session, is still never a `session_send` target.
The reach is not new to the trust model: the same DM already resumes any dashboard
session into itself through `!sessions` / `/sessions` under the same single-owner
rule, and the same dashboard tab already had it before the link.

**Which channels got which path.** Discord and Telegram: the predicate, because
the three facts membership asserts were verified against their transports — the
DM key is `{surface}:{agent}:direct:{peer}`, `configured_targets()` advertises that
peer as `user:{peer}` from configured state alone (Weixin and WeCom fold learned
identities in, which is why `constants.CHANNEL_OWNER_DM_NAMESPACES` excludes them
and this set is a subset of it), and the transport attests a DM conversation's peer
from its own state (`direct_peer_of`: Discord from the pairing its client records,
Telegram from the identity a positive-integer private `chat_id` carries). Slack, Webex, Teams,
WhatsApp, iMessage, Feishu, WeCom, Weixin and `unified`-scope DMs: no relaxation —
they read as contained exactly as before, in both shapes, and a mirror onto any of
them fails the surface clause before a transport is consulted. A channel graduates
by verifying the three facts for it, overriding `direct_peer_of` (the base class
answers `""`, which admits nothing), and adding its name to
`OWNER_DM_CONDUCTOR_SURFACES`; nothing else changes. No "detach from channel"
dashboard action was built: the dashboard's Disconnect row pauses outbound
delivery and retains the binding by design (see [session](session.md)), and with
the predicate in place an owner needs neither it nor an unlink to conduct — from
the DM or from the tab mirrored to it.

Related fold: the ledger's bind ownership check compares `_created_by` (the
creator's **slot** key, as `session_create` stamps it) against the conductor
**session** key's resolved slot (`caller_slot_key`), because a channel-born
conductor is keyed `discord:…:genN` while its slot is that key folded to the
filename charset — a string comparison refused every worker such a conductor
created as `worker_not_owned`. A dashboard conductor compares as before.

Pinned by `test/test_session_control_owner_dm.py`, against a real `SessionMap`
whose rows carry the dispatcher's first-turn `set_channel` bucket: a Discord
thread and a Telegram forum topic refused by all three gates; an owner DM on
Discord and on Telegram conducting end to end; the fence; the paused mirror; every
fail-closed edge (two identities, a stranger's DM, an absent or unavailable
transport, a retargeted mirror, an unknown origin, an unreadable store,
unparseable and non-direct keys); gate agreement over one slot; the post-read
re-check; the bind fold; that an owner DM which `!unlink`s its own mirror is still
admitted by all three gates while a threaded Slack mirror refuses; and that
mirror-unlink clears only the mirror. For the dashboard-born shape: a `chat-*`
tab mirrored to the owner's DM conducting end to end on Discord and on Telegram;
the same tab mirrored to a guild thread, a stranger's DM, a two-identity roster, a
forum topic, an unverified surface and a Slack thread refused with today's codes;
a Slack thread bound beside the owner-DM mirror; a retarget judged at the next
call and an unlink restoring the ordinary tab; the pairing not on record; a cron
tab and a workflow tab mirrored to the owner's DM refused as unattended; every
fail-closed edge of the shape (no transport, an unreadable roster, an unavailable
owner, a peer lookup that raises, an unreadable store); that the admitted tab keeps
its reach while an agent-created child mirrored to the same DM stays fenced; gate
agreement over a mirrored slot; that every surface in the set overrides
`direct_peer_of`; and the two real transports' answers. The audience record: a
read under the owner-DM mirror is withheld from a retargeted or widened mirror at
publication and published to an unchanged one; the same record for a plain tab
that gains a mirror; every admission gate writing it, not only the transcript
read; polling recording one entry per audience, not one per call; and that a
retarget landing between the gate's validation and its record -- walked across
every read the gate could make, on the session-control gate and on the ledger's
-- is never the recorded audience: the record is the judged row, a reply under the
retarget is withheld, and the admission read the row exactly once; and that a
channel-born owner DM conducting from its own channel turn records no audience
(so a later dashboard reply on that slot is not withheld after a mirror change)
while the same slot driven from the dashboard records and still fences, the
marker being the runner's own; and that a turn whose only call is `create_session`
records its audience at the creator gate; and that a Telegram allow-list entry that
is not a positive integer (a group id, zero, text) is never attested as a DM peer,
so a tab mirrored to such a conversation stays refused with the real transport. The crew-log
opener is pinned in `test/test_discord.py` and `test/test_telegram.py`: a DM turn
followed by a `work_ledger_record` write against the real writer lands, and the
resumed-session path opens nothing.

## The wait → read poll loop

`session_read_message` is the observation half, and polling is the supported
shape:

1. `session_read_message(target)` — record `next_since`.
2. `wait(seconds=…)`.
3. `session_read_message(target, since=<previous next_since>)` — returns only what
   arrived since, so a loop does not re-read the same messages.

`total` is an **absolute position** in the session, not the length of the live
window. A slot retains only its most recent messages in memory and credits each
trimmed row to a frozen-prefix counter, so a length-derived cursor would freeze
at the retention cap — and a poller on a long session would silently stop seeing
replies, on exactly the sessions that need it most. Positions are based on the
**durable-only** frozen-prefix counter (`_disk_older_durable_count`), which
counts only trimmed rows a durable read returns — never the all-rows
`_disk_older_count`, which also counts transient rows and would shift every
position as soon as one was trimmed. A trimmed session therefore keeps an exact
cursor: `next_since` is returned as usual. The one trim-related refusal left is
a `since` **below** the trimmed prefix (409 `cursor_unavailable`): those rows
exist only on disk now, and starting the read at the window instead would
silently skip everything in between. The caller falls back to a tail read.

`running` is what makes the loop terminable: `running: false` with an empty
window means the target finished and went idle, which is different from "nothing
new yet". `queue_depth` reports how much the target still owes.

A turn parked on a tool approval keeps `running` true for the whole approval
window, so `running` alone cannot tell "working" from "waiting on a person".
While the target has an open approval, the read adds `pending_approval: true`
and, when a title is known, `pending_approval_tool` (redacted, at most
`MAX_PENDING_APPROVAL_TOOL_CHARS`). The values are the slot projection's own
`pending_approval` and `pending_approval_info["tool"]` (`slot.to_dict()`), the
fields the dashboard card reads, so the two cannot disagree; that covers both
the slot's `_approval_futures` (a native tool prompt) and
`state.pending_coordinator_approvals(slot.key)` (a sub-agent spawn gate or a
tool inside a running sub-agent). `to_dict()` walks the transcript, so it runs
only once one of those registries has an open entry. The fields are absent when
nothing is waiting. `session_status` rows
carry the same two fields from the same helper, `_pending_approval_fields`; the
row's `status` stays `working`.

The cursor deliberately stops **before the streaming tail**. `chat_runner`
appends a `chunk` row per token burst and `_flush_segment` then deletes that
trailing run, replacing it with one durable assistant message — so chunk rows are
always a suffix, never interleaved. Counting them would inflate `total`, the
flush would shrink the list back under it, and the next `since=next_since` read would
skip the finished reply permanently. A read taken mid-reply therefore reports
`streaming: true`, so an empty window while the target is composing is
distinguishable from an empty window because nothing is happening.

A stale cursor is refused, not clamped. A compacted or rewound transcript shrinks,
so a `since` past the end answers 409 `cursor_unavailable` and the caller falls
back to a tail read. Clamping it to the end would look friendlier and lose data:
the rows below the clamp are what replaced the old tail, a cursor never moves
backwards, so they would be skipped permanently while the response read as
"nothing new". A cursor exactly AT the end is not stale and still returns an empty
window.

## Ending a wait early

`session_end_wait(target)` is the other half of the poll loop: a caller that has
already seen the condition its worker is sleeping on can wake that worker instead
of letting the `wait` run out. `end_wait_target` runs `authorize_target` like the
other verbs and then applies a creator fence of its own: a target whose
`_created_by` is not the caller is refused `not_creator` (403) even for an owner
session, which the shared gate does not fence. Waking a sleep moves another
session's turn forward on the caller's schedule, and the caller that armed the
worker's wait is the one that knows when that is safe.

It reuses the End-wait button's mechanism rather than adding one. It reads the
`wait_id` currently tracked in `_wait_state` at request time (an MCP caller has no
countdown to name a stale id from), parks it in `_end_wait_request`, and records
the caller in `_end_wait_by`. The sleeping tool collects it from its next
keepalive reply, which then carries `end_wait_by`, and returns a normal result
naming the session that ended it. The turn is not cancelled and nothing is
discarded.

A target with no tracked sleep, or with `_wait_contested` set (two sleeps share
one session key, so neither can be aimed at), gets `ok: true, ended: false` with
an `info` string and nothing is parked. The SEL audit detail records
`requested`, `not_waiting` or `contested`. Channel agents are blocked from the
verb, and it is withheld from the conductor and member auto-approve grants; see
the comment on `_CONDUCTOR_DASHBOARD_GRANTS` in `agent.py`.

## Stopping is safe to re-send

The Stop button escalates: a second press while the first cancel is still pending
hard-kills the turn, and the hard-kill path clears the slot's queue and its pending
steers. That is right for a button, where the second press means a person watched
the cooperative stop fail to take. It is wrong for an RPC, where a client that got
no response inside its 30s request timeout re-sends the same request — so on the
button's semantics a timeout retry would silently get the destructive variant of a
verb the caller asked for once, and the queued work would be gone with nothing
saying a retry rather than a decision caused it (issue #5074).

`session_stop` therefore withholds the escalation for a call it cannot tell apart
from a retry. `stop_retry.allow_escalation` records the first stop a caller makes
against a target and answers `False` for any repeat inside `WINDOW_SECS` (120s);
`stop_slot_turn` takes that as `escalate=False` and lets the repeat fall through to
its existing "stop already in progress" no-op.

Three properties are worth stating because each one is a way this could have gone
wrong:

- **Only the escalation is withheld, never the stop.** A repeat that finds the
  target running again soft-stops it exactly as a first call would. The window
  suppresses a kill, not a cancel.
- **The window is anchored at the first stop and is not extended by the repeats it
  absorbs.** So escalation is suppressed for at most one window: a client that
  retries forever is absorbed, and after 120s a stop that STILL finds the target
  winding down escalates — which is the case where escalating is the right answer.
  A sliding window would put a hard kill out of reach of any caller polling faster
  than the window.
- **The key is (caller, target), not the target alone.** A retry comes from the
  caller that made the original request; two different callers stopping one target
  are two independent decisions, and keying on the target would suppress the second
  caller's FIRST call — removing escalation from the RPC rather than making a retry
  safe.

The window is sized against what it has to outlast rather than picked: below the
30s request timeout it would expire before the retry it exists to absorb. Nothing
durable backs it, for `create_rate_limit`'s reason — a restart buys a caller one
window, not a capability.

The caller is told which of the two no-op facts it hit. `already_stopping`
separates "was never running" from "its cancel is still in flight", because a
de-duplicated retry reaches that reply routinely and rendering both as "nothing to
stop" would tell the second caller the opposite of what happened.

## Closing archives, and re-checks at the point of no return

`session_close` is the tool-side equivalent of the tab ✕. It is **non-destructive**:
the conversation is saved to history (`closed=True`) and can be reopened later, so
closing dismisses the LIVE tab, it does not delete the transcript. It is a
strictly heavier act than `session_stop` — an in-flight turn is cancelled first
and its work discarded — so the tool description tells the caller to read the
session before closing it. Close waives the self-target refusal
(`allow_self=True`, as release does) on both its admission gate and its pre-pop
re-check, so a session that has finished its one job can archive its own tab; the
self-close cancels the caller's own running turn, which is the turn that asked
for it, and every other caller and target rule still applies. It reuses the dashboard's own close path
(`close_slot`), the same sequence the ✕ button runs: a synchronous tombstone,
auto-nudge-loop retirement BEFORE the awaits so no nudge resurrects the tab, the
owning app's close hook with rollback, persist-as-closed, and per-tab session
teardown. Its four failure modes surface as their own codes at HTTP 500
(`history_write_running`, `nudge_retire_failed`, `app_close_hook_failed`,
`history_save_failed`), which is why the routes now forward a 500 rather than
degrading it to 400.

**Authorization is re-asserted at the point of no return.** `authorize_target`
runs before `close_slot`, but `close_slot` then awaits — auto-nudge retirement
takes the AutoNudge lock, and the app hook awaits external work — and a target
that was unmirrored and unlinked at admission can gain a channel mirror or link
in that window. Archiving a now-channel-backed session it was never allowed to
reach is exactly the boundary the `mirrored_target` / `linked_session_target`
guards hold, so `close_target` passes a SYNCHRONOUS `pre_pop_check` that runs
immediately before the slot is popped, after every await (the nudge retirements
and the app hook). It re-runs `authorize_target` with `skip_enabled_check=True` —
omitting the one part of that gate that can read config on the loop, since the
feature was already confirmed enabled at admission and disabling it mid-close is
not a containment boundary — and compares the re-resolved slot to the one being
closed **by identity**: a concurrent close-and-reopen can re-mint the same key
onto a different session, and popping that would tear down the replacement while
saving the stale slot (409 `target_replaced`).

There is a SECOND config read on that gate to close, not only the switch: the
ownership fence (`_caller_is_ownership_fenced` → `_member_caller` →
`_store_is_member_owned`) loads config on a cache miss to classify the caller's
store, and running that inside the no-suspension window is the same blocking-IO
hazard. `close_target` therefore never computes the fence inside the window: a
verdict the HTTP gate carried (a caller admitted as a crew member, see "Member
callers") is honoured as-is, and for any other caller it is resolved ONCE up front
— while the cache is still warm from `prewarm_enabled_check` and before
`close_slot`'s awaits — and passed to BOTH the initial gate and the re-check as
`precomputed_ownership_fenced`, so the callback consults a carried boolean instead
of re-deriving member-ownership from config on the loop. Being synchronous is the whole point —
there is no suspension between the last retirement, this re-check, and the pop, so
nothing (a channel mirror/link landing, a re-mint, or a racing `monitor_start`
arming a loop) can change between the final authorization and the archival; an
awaited re-check, by contrast, reopens exactly those windows. Any refusal aborts
the close, rolls back the retired nudge loop, and surfaces as the guard's own
status. This is the same "re-gate adjacent to the mutation, comparing identity not
presence" discipline `create_session` uses for its slot allocation, and the same
theme as the queued-drain re-check (#5911). The human ✕ path passes no check — the
person owns the tab and closes it unconditionally.

## Reviving is the mirror of closing, authorized from the metadata line

`session_revive` is the tool-side equivalent of clicking an archived session in
the History tab. It reuses the dashboard's own resume core
(`chat_handlers.resume_slot_from_history`, the request-free half of
`POST /api/chat/slots/{slot}/resume`), so a controlled revive and a human click
share one materialisation, one set of member-pin and delete/recreate barriers and
one set of refusal codes. The revived slot is idle — nothing runs until a
`session_send` — and the reply carries its live key so every other verb can
address it. A `folder_id` files it as part of the same call, after the revive
has landed and only after the folder's existence was confirmed BEFORE anything
was revived, so a refused filing leaves history untouched.

**There is no live slot to authorize against, so the target-side checks read
the persisted metadata line instead** — the same fields `authorize_target` reads
off a live slot (`workspace`, `app`, `linked_session_key` / `channel_origin`,
`memory_mode`, `created_by`), in the same order, raising the same codes. The
caller-side checks are literally shared: `authorize_target` and `revive_session`
both call `refuse_caller_identity` (key-only refusals, before the target is
resolved, so a refused caller learns nothing) and `refuse_caller_surface` (the
live-slot refusals, whose store-free half is `_check_caller_slot_fields` so the
revive's synchronous last-word check can reuse it), and the fence wording comes
from one `_not_creator_reason`.
The ownership fence therefore has the same reach on an archived session as on a
live one: an ownership-fenced caller (crew member, scheduled run, agent-created
session) may revive only a session whose `created_by` is itself, corroborated
against the crew-log session-tree lineage through `_slot_tree_parent` (the record
`create_session` writes from the in-process `_lineage_minted` witness) because the
metadata line is agent-editable: an unreadable lineage (crew log off, unseeded,
incomplete) or a different parent refuses `ownership_unverified`, fail-closed. The
`created_by` the metadata carries is restored onto the revived slot as attribution
only (`_lineage_minted` stays False) — reviving never transfers ownership to the
reviver. For the per-caller slot cap the revived slot is charged to the reviver
through an in-memory `_revived_by` field (the registry's `creator_slot_count`
charges `_created_by` or `_revived_by`, so create and revive share one
accounting), tested before the resume and re-tested, with the global cap, the
mirror, the store-recorded channel link and the four live-target fields, in the
resume core's pre-publish `containment` hook: the core hands the built slot to
the hook after hydration and before publish, holding it retracted from the slot
table and under construction while the hook awaits (a concurrent resume of the
key meanwhile is answered `resume_in_progress`), with the row broadcast held back
and the reopen write (clearing `closed`) deferred until the hook has passed, so a
refusal discards the built slot with nothing durable to undo and a clear that
cannot land refuses `reopen_failed` instead of publishing a tab that would not
restore. The existence and `created_at` identity barrier that guards the
hook-less resume is re-run after the hook's last await, again on the
verification read after the deferred clear, and a final time OFF the loop
(`asyncio.to_thread(log.get_metadata_status, ...)`, so its bounded retries pause
and outlast a Windows sharing-violation hold on a just-rewritten file) before the
second hook pass, so a session deleted or delete-and-recreated inside the hook
window is refused `resume_session_deleted` rather than published over the
replacement (an unreadable answer on that last read refuses `resume_conflict`
rather than falling through); the marker rollback compares `created_at` too, so it never archives
a replacement. Because the deferred clear, its verification read and that final
identity read are awaits after the hook's store-backed probes, the hook runs a
second time after them as the last awaiting act, so a channel binding recorded
in the store during those awaits is still refused. After that last await no file
is read: two lock-free witnesses stand in for it, the key's history-cache
invalidation generation (snapshotted before the off-loop read; every completed
delete bumps it) and the in-flight delete marker, and either one refuses
`resume_conflict` (409) immediately before the publish. The store-free answers (slot fields, caps) are re-asserted once more in a
synchronous `final_check` after the core's last await, immediately before the
publish; a refusal there restores the marker while the construction mark still
reserves the key, and the restore is confirmed by a re-read (one retry on a
raise): a marker that cannot be confirmed back answers `reopen_rollback_failed`
(503) in place of the refusal that triggered it, so the caller hears that the
durable session may reopen at the next start rather than a refusal that implies
it was left as found. A revive that finds the session already live (a live match
before the history scan, one that went live during it, or a human click that won
the race with the resume) answers through one builder that authorizes the live
slot as any live target before naming it: a protected slot answers with that
refusal and reveals neither its existence nor its key, an unprotected one answers
`target_already_live` with its key. Under a hook the built slot's `_app`,
`linked_session_key` and `channel_origin` are restored from the fresh metadata
re-read (the resume core hydrates neither link field itself), so the hook's app
and link checks read the line as it is, not a constant. The History tab's own
resume passes neither hook and is unchanged.

Resolution mirrors `_resolve_slot`'s doctrine for archived sessions: a target
may be a slot key, the `dashboard:<slot>` session key, the `dashboard_<slot>`
transcript stem, or an exact case-insensitive title, every form is resolved
before anything is returned, and two different sessions matching across forms
is `ambiguous_target`. A target that is LIVE is refused with `target_already_live`
and the message names the live key, because the caller asked for an archived
session and should learn that this one is not — the resume core would have
deduplicated the slot anyway, but silently answering "done" would hide that the
caller's model of the sidebar is stale. Member DM threads
(`member-*`) are refused outright (`member_thread_target`): they are opened only
through the roster route that re-checks the member binding.

## Configuration

`agent.session_control` (bool, default **true**). The grant that decides who may
reach a peer session is the **agent config**, not this switch: the five tools come
from the `kirocrew-dashboard` MCP server, so an agent whose spec does not mount it
never has them — the same rule as every other MCP server. A second default-off
gate on top of that only made the capability unreachable for an agent that had
already been given it deliberately, and `_install_conductor_agent()` shipping that
mount is what an explicit grant looks like.

What the switch is still for is a single withdrawal: an operator who wants the
capability gone from every agent at once, without editing each spec. So the
direction that must keep working is an explicit `false`, and `_safe_bool` is what
keeps a quoted `"false"` from loading as enabled — `bool("false")` is `True`, so a
plain coercion would give a user who wrote it in an editor that quotes values the
opposite of what they read.

A config read that RAISES still resolves to disabled rather than to the default.
That is deliberately not symmetric with the absent case: an unreadable config is a
transient fault the operator can diagnose from the log line, and refusing during it
costs a retry, while assuming the default during it would let unrelated corruption
decide an authorization question.

One consequence worth stating, because it is what the default-off gate was
protecting: the same server carries the `chat_folder_*` tools, so an agent assigned
it for folder organization has the session verbs too. Whether they prompt depends on
that agent's `allowedTools` — naming individual tools leaves the session verbs to
`hooks.on_tool_call`, while naming the whole server auto-approves them, because
`_mcp_pattern` maps a bare `@server` entry to a one-level glob and
`is_tool_in_allowlist` checks `@server` before `@server/<tool>`. The shipped
conductor is in the second class for the verbs in `_CONDUCTOR_DASHBOARD_GRANTS`
(`agent.py`: the folder tools, `session_create`, `session_read_message` and
`session_status`), which is its stated operating model: its patrol
loop runs with nobody at the keyboard and must not block on an approval no one is
there to give. An operator who wants folder tools without session control names the
folder tools individually.

The chat routes check the switch for a script cron as well. While it is off, the
gateway refuses a caller that presents a `cron:` session key on
`POST /api/chat/slots`, `POST /api/chat` and `POST /api/chat/mode` with
`session_control_disabled`. These are the three routes
`ScriptContext.open_session`, `ScriptContext.send_to_session` and
`ScriptContext.set_session_mode` call. The check is `_cron_session_control_refusal`
in `private_chat_route_refusal`, the gate every internal chat-route call passes
after the internal secret validates, and it answers with the same 403 body the
session-control routes send. The same three routes apply this module's creator
fence to a `cron:` caller: `cron_creator_admission` in the chat handlers refuses a
key whose slot was not created by that `cron:` key with 403 `not_creator`, the
code `authorize_target` answers. A live slot is judged on its `_created_by`
through `_created_by_other`. A key with no live slot is judged on the
`created_by` its persisted metadata line records, so a cron cannot mint a
closed session's key as its own, and a key with neither a slot nor a transcript
is left to mint as the cron's own on the two routes that mint; the mode route
mints nothing and runs the fence on the live slot it resolved. The mode route
adds one rule of its own, `cron_mode_refusal`: a `cron:` caller sets `trust` or
`trust_reads` on the one slot it names, and every other mode, `yolo` and
`normal` included, answers 403 `mode_not_allowed` before governance or the
safety override is consulted, while an unnamed slot answers 400 `slot_required`
rather than the owner's all-slots grant. `_cron_session_control_refusal` and
`cron_creator_admission` are mirrors of this module's cron gate, the switch gate
and the `_created_by_other` fence, not a second rule: a change to how this
module gates a cron must change those helpers with it. All three checks key on
the key the caller presents, so they are a courtesy for `ScriptContext` callers
and do not stop a holder of the internal secret. Each refusal is audited through
one shared write, `_audit_cron_chat_denial`, as a `chat.control` denial. Owner
and member callers are unaffected and keep their own gates. The folder routes
are not gated, because folders are not session control.

A slot a `cron:` caller opens on either route is labelled cron-created:
`cron_slot_creator` reads the attested key off the scope the gate resolved, and
the slot is minted with `origin` `CRON` and `_created_by` set to that
`cron:<job id>` key. It is not counted as a user-created session, the
`slots:user` scope does not expose it, and the owner sees it in the sidebar as
any cron tab. The seeded first message is still queued as a user-role turn, a
trade-off the PR that added these methods records. A slot that already exists
under the name a cron sends is never re-labelled.

`agent.member_dispatch` (bool, default **true**). The operator ceiling on the
member switch bypass described under "Member callers". At its default a member DM
caller bypasses `agent.session_control` — the zero-configuration contract, and
today's behaviour, so installing this key changes nothing until it is set. Set it
`false` and a member caller stops bypassing: it falls back under
`agent.session_control` like any ordinary caller, which lets an operator who
withdrew session control keep member DM threads chat-only without disabling the
member. Read at the switch gate via `member_dispatch_enabled()`, and **fails
closed** everywhere it can go wrong. (1) An unreadable config withdraws the member
bypass (`member_dispatch_enabled()` returns false on a raising read). (2) A config
that loads but *discarded the `agent` section* (or the whole file) also withdraws
it: `load()` does not raise on a malformed section — it drops the section and
falls back to the permissive `member_dispatch=True` default, recording the loss in
`degraded_sections`, so `member_dispatch_enabled()` returns false when
`degraded_sections` names `agent` or the whole-config marker `*`, the same
"could not read it" vs "was never set" distinction `tailnet_identity_unknown` and
the publish gate draw. (3) At load time a *missing* key defaults to true (today's
behaviour) while any *present but malformed* value — including a quoted `"false"`,
a routine operator quoting mistake — coerces to false BEFORE schema validation, so
a botched opt-out withdraws the bypass rather than silently leaving it on. This is
a config-level ceiling, not an enterprise `SCOPE_CATALOG` scope: `session_control`
itself is a plain `agent.*` bool with no catalog entry, and a scope would need a
new governance enforcement seam rather than a data-only append, so it is left to a
follow-up.

## What is deliberately not here

- **No delivery to a target outside the addressable set.** `session_send` writes
  into another session's conversation, but only one the same `authorize_target`
  guard admits: a channel-linked, channel-mirrored, incognito,
  app-scoped, unattended or cross-workspace target is refused, so the verb cannot
  reach a conversation other people are party to. That holds for a steer too: the
  mode changes when the message runs, never who may send it.
- **No cross-workspace or cross-machine reach.** The boundary is one gateway's
  live sessions in one workspace.
- **No waking closed sessions.** See above.
- **No writes on the read path.** `session_read_message` never changes the
  target's state, so a poll loop cannot perturb what it is measuring.

### Ordinary Codex dashboard grants

An ordinary agent that explicitly grants `@kirocrew-dashboard` receives the
managed dashboard mount on both create and resume through the Codex mirror.
The grant, disabled-server and per-tool restrictions remain authoritative; a
spec-supplied command is replaced before session identity is attached. Broker
mounts preserve verified per-call identity and are required for host API access
inside enforced sandboxes. Session ownership, workspace and private-session
rules remain enforced by the existing session-control handlers.
