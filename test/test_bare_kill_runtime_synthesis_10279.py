"""A bare kill's substitution body must not assemble the protected name at runtime.

The bare-kill window searched each substitution body raw, with parameter
defaults resolved, and per word of the tokenized view -- all static.  A body
whose ``pgrep``/``pidof`` target is built at runtime from constant-output
substitutions (``"$(printf kiro)$(printf crew)"``) or brace expansion
(``kiro{,}crew``, ``kiro{c..c}rew``) slipped every member.  The window now
folds constant ``printf``/``echo`` output and expands braces per word before
searching.

Posture: these forms are DENIED because the floor can compute them exactly.
An unresolvable operand does NOT fail closed: a pattern variable bound outside
the line and a lookup naming a different product stay allowed and are pinned
here as deliberate residuals.

An over-cap brace PRODUCT does not ride the fan-out cap to safety: a product
whose groups can concatenate to the protected name is denied by a reachability
scan ACROSS the groups that never builds the product (``pidof`` takes many
names at once, so a member that spells the name is a live lookup). A product
that CANNOT spell the name stays the residual, decided the same cheap way.

The protected verb and name are assembled at runtime so an agent shell can
read and grep this file without tripping the very rule under test.
"""

from __future__ import annotations

import pytest

from kiro_crew.security import argv_floor as _argv_floor
from kiro_crew.security import is_denied

_K = "k" + "ill"
_PG = "pg" + "rep"
_PG_KILL = "pk" + "ill"
_NAME = "kiro" + "crew"
_HEAD = _NAME[:4]
_TAIL = _NAME[4:]
_ESAC_PREFIX = "case x in x) :;; esac; "
# Nine single-literal printf substitutions nested innermost-first around the head.
_NEST9 = "$(printf " * 9 + _HEAD + ")" * 9


def _floor(cmd: str) -> bool:
    """The argv floor's own verdict, independent of the regex tier."""
    return _argv_floor._is_self_kill(cmd.lower())


_SYNTHESIZED_BODIES = [
    # nested constant-output substitutions
    f'{_PG} -f "$(printf {_HEAD})$(printf {_TAIL})"',
    f'{_PG} -f "$(echo -n {_HEAD})$(echo -n {_TAIL})"',
    f"{_PG} -f `printf {_HEAD}`{_TAIL}",
    f'{_PG} -f "$(printf "$(printf {_HEAD})"){_TAIL}"',
    # nine nested single-literal substitutions fold to a fixpoint, not one short
    f'{_PG} -f "{_NEST9}{_TAIL}"',
    # a computed value carried through a same-line assignment
    f'x=$(printf {_HEAD}); {_PG} -f "${{x}}{_TAIL}"',
    # brace expansion: alternation and a one-character sequence
    f"{_PG} -f {_HEAD}{{,}}{_TAIL}",
    f"{_PG} -f {_HEAD}{{{_TAIL[0]}..{_TAIL[0]}}}{_TAIL[1:]}",
    f"pidof {_HEAD}{{,}}{_TAIL}",
    # an argument shaped like name=value is NOT an assignment, so its braces
    # expand (bash runs `pgrep -f x=.*|kirocrew`, whose regex names the gateway)
    f"{_PG} -f x='.*|'{_HEAD}{{{_TAIL[0]}..{_TAIL[0]}}}{_TAIL[1:]}",
    # a leading assignment does not stop a later command argument from expanding
    f"TMPDIR=/t {_PG} -f {_HEAD}{{,}}{_TAIL}",
    # a parameter word nested in a brace group does not stop the group
    # expanding (bash brace-expands before parameter expansion)
    f"pidof -x {_HEAD}{{${{x:-}},}}{_TAIL}",
    # identical brace alternatives collapse rather than overflow the fan-out cap
    f"pidof -x {_HEAD}{{{_TAIL[0]}..{_TAIL[0]}}}{_TAIL[1:]}" + "{,}" * 9,
    # an UNPROVEN backtick in a parameter word must not swallow the group's close
    f"pidof -x {_HEAD}{{${{x:-'`'}},{_TAIL[0]}}}{_TAIL[1:]}",
    # same-line variable concatenation (denied through the tokenized view)
    f'a={_HEAD}; b={_TAIL}; {_PG} -f "$a$b"',
    # a quote inside a comment is not syntax and must not mask the next line
    f'# it\'s\n{_PG} -f "$(printf {_HEAD})$(printf {_TAIL})"',
    # a quoted, escaped or unterminated ``${`` does not open a parameter expansion
    f"pidof '${{' {_HEAD}{{,}}{_TAIL}",
    f"pidof \\${{ {_HEAD}{{,}}{_TAIL}",
    f": '${{'; {_PG} -f {_HEAD}{{{_TAIL[0]}..{_TAIL[0]}}}{_TAIL[1:]}",
    # an ANSI-C escaped quote does not close its string
    f": $'\\''; {_PG} -f {_HEAD}{{{_TAIL[0]}..{_TAIL[0]}}}{_TAIL[1:]}",
    # a command substitution inside a parameter default runs, braces and all
    f'echo "${{x:-$({_PG} -f {_HEAD}{{{_TAIL[0]}..{_TAIL[0]}}}{_TAIL[1:]})}}"',
    # a double-quoted close-paren and an escaped brace are data, not delimiters
    f': "$( : ")")"; {_PG} -f {_HEAD}{{{_TAIL[0]}..{_TAIL[0]}}}{_TAIL[1:]}',
    f": ${{x:-\\{{}}; {_PG} -f {_HEAD}{{{_TAIL[0]}..{_TAIL[0]}}}{_TAIL[1:]}; : \\}}",
    # only a substitution's real closer ends its scope: not a subshell's, not a
    # case pattern's, and a ``$$`` (the PID parameter) opens nothing
    f'echo "$( ( : ); {_PG} -f {_HEAD}{{{_TAIL[0]}..{_TAIL[0]}}}{_TAIL[1:]})"',
    f'echo "$(case a in a) printf \'"\';; esac)"; {_PG} -f {_HEAD}{{,}}{_TAIL}',
    f'echo "$$( \'"\' )\'")" ; {_PG} -f {_NAME}',
]


@pytest.mark.parametrize("prefix", ["", _ESAC_PREFIX], ids=["plain", "esac-tail"])
@pytest.mark.parametrize("body", _SYNTHESIZED_BODIES)
def test_runtime_synthesized_target_is_denied(prefix: str, body: str) -> None:
    cmd = f"{_K} $({prefix}{body})"
    assert _floor(cmd), cmd
    assert is_denied(cmd), cmd


@pytest.mark.parametrize(
    "cmd",
    [
        # an unresolvable pattern variable is NOT failed closed
        f'{_K} $({_PG} -f "$pattern")',
        f'{_K} $({_PG} -f "${{pattern}}")',
        # a lookup naming a different product
        f"{_K} $({_PG} -f otherproduct)",
        f"{_K} $({_PG} -f other{{,}}product)",
        f'{_K} $({_PG} -f "$(printf other)$(printf product)")',
        # a constant substitution unrelated to the target
        f'{_K} $({_PG} -f "$(printf worker)")',
        # output words bash keeps apart are not joined
        f'{_K} $({_PG} -f "$(echo {_HEAD} {_TAIL}-worker)")',
        # a format conversion is not modelled -- residual, left unfolded
        f'{_K} $({_PG} -f "$(printf %s%s {_HEAD} {_TAIL})")',
        # a printf escape in the operand is a residual (not decoded, no crash)
        f"{_K} $({_PG} -f \"$(printf '{_HEAD}\\n{_TAIL}')\")",
        f"{_K} $({_PG} -f \"$(printf '\\8')\")",
        # an unknown printf escape is kept, not decoded or crashed on
        f"{_K} $({_PG} -f \"$(printf '\\8')\")",
        # output carrying syntax is left unfolded rather than altered
        f"{_K} $({_PG} -f \"$(printf '{_HEAD};{_TAIL}')\")",
        # a numeric range past the cap cannot add letters
        f"{_K} $(lsof -ti tcp:{{3000..3999}})",
        f"{_K} $({_PG} -f {_HEAD}-cli-server-worker-{{1..300}})",
        # a 300-way alternation of long replica names, no name in any of them
        f"{_K} $({_PG} -f svc-{{" + ",".join(f"worker-alpha-{i:03d}x" for i in range(300)) + "})",
        f"{_K} $({_PG} -f shard-{{a..z}}{{0..9}})",
        # a nested group whose own product overflows the cap, no name in it
        f"{_K} $({_PG} -f 'shard-{{{{a..j}}{{a..z}},main}}')",
        f"{_K} $({_PG} -f shard-{{{{a..z}}{{a..z}},main,canary}})",
        # a trailing (non-leading) echo operand is not an option -> residual
        f'{_K} $({_PG} -f "$(echo {_HEAD} -n)crew")',
        # quoting makes braces and substitutions LITERAL -- bash does not expand
        # them, so neither does the floor (these name no live process)
        f"{_K} $({_PG} -f '{_HEAD}{{c..c}}{_TAIL}')",
        f"{_K} $({_PG} -f '$(printf {_HEAD}){_TAIL}')",
        f"{_K} $({_PG} -f \"$(printf '{_HEAD};')crew\")",
        # a substitution nested inside another's single-quoted arg is inert
        f"{_K} $({_PG} -f \"worker|$(printf '$(printf {_HEAD})'){_TAIL}\")",
        f"{_K} $({_PG} -f \"worker|`printf '$(printf {_HEAD})'`{_TAIL}\")",
        # an escaped substitution and a parameter default's braces are literal
        f'{_K} $({_PG} -f "worker|\\$(printf {_HEAD}){_TAIL}")',
        # quotes that are DATA inside the operand are kept, so no name is invented
        f'{_K} $({_PG} -f "$(printf "\'{_HEAD}\'"){_TAIL}")',
        # an assignment stores its braces; a later reference is never brace-expanded
        f'{_K} $(pattern={_HEAD}{{3,}}{_TAIL}; {_PG} -f "$pattern")',
        f'{_K} $({_PG} -f "worker|{_HEAD}${{x:-{{{_TAIL[0]}..{_TAIL[0]}}}}}{_TAIL[1:]}")',
        f"{_K} $({_PG} -f {_HEAD}-cli-server-worker-{{a..z}}{{a..z}})",
        # the name built in a DIFFERENT command than the kill
        f"{_K} 4242; echo {_HEAD}{{,}}{_TAIL}",
    ],
)
def test_false_positive_controls_stay_allowed(cmd: str) -> None:
    assert not _floor(cmd), cmd


def test_quoted_substitution_close_paren_does_not_end_the_kill_window() -> None:
    # ``"$( : ")")"`` is one word: its inner substitution has its own quoting,
    # so the quoted ``)`` closes nothing and the lookup after it is in the window.
    cmd = f'{_K} $(: "$( : ")")"; {_PG} -f {_NAME})'
    assert _floor(cmd), cmd
    assert is_denied(cmd), cmd


def test_doubling_assignment_chain_fails_closed_before_building_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each step doubles the value; the bound is checked on lengths, so the
    # resolver that would build the strings is never reached.
    from kiro_crew.security import shell_normalizer as sn

    def refuse(_text: str) -> "list[str]":
        raise AssertionError("the resolver must not run past the bound")

    monkeypatch.setattr(sn, "_self_tokens", refuse)
    chain = "; ".join(f"x{i}=${{x{i - 1}}}${{x{i - 1}}}" for i in range(1, 41))
    body = f'x0=$(printf a); {chain}; {_PG} -f "$x40"'
    assert _argv_floor._kill_body_synthesizes_self(body)


def test_continuation_split_chain_fails_closed_before_building_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The doubling refs are split by a line continuation, so sizing the raw text
    # would see each step as separate linear-growth words and let the eager
    # resolver build (and OOM to an empty, allowing result). Sizing folds the
    # continuation the way the resolver does, so the chain is bounded first.
    from kiro_crew.security import shell_normalizer as sn

    def refuse(_text: str) -> "list[str]":
        raise AssertionError("the resolver must not run past the bound")

    monkeypatch.setattr(sn, "_self_tokens", refuse)
    chain = "; ".join(f"x{i}=${{x{i - 1}}}\\\n${{x{i - 1}}}" for i in range(1, 41))
    body = f'x0=$(printf a); {chain}; {_PG} -f "$x40"'
    assert _argv_floor._kill_body_synthesizes_self(body)


def test_quoted_space_chain_fails_closed_before_building_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The doubling refs are concatenated across a QUOTED space, so a naive
    # whitespace split would size each step as separate linear words and let the
    # eager resolver build (and OOM). Sizing splits words quote-awarely, so the
    # quoted value is one assignment and the chain is bounded first.
    from kiro_crew.security import shell_normalizer as sn

    def refuse(_text: str) -> "list[str]":
        raise AssertionError("the resolver must not run past the bound")

    monkeypatch.setattr(sn, "_self_tokens", refuse)
    chain = "; ".join(f'x{i}="${{x{i - 1}}} ${{x{i - 1}}}"' for i in range(1, 41))
    body = f'x0=$(printf a); {chain}; {_PG} -f "$x40"'
    assert _argv_floor._kill_body_synthesizes_self(body)


def test_wide_comma_group_is_capped_before_building() -> None:
    # A comma alternation of DISTINCT alternatives past the cap is signalled as
    # overflow and never materialised, so a group of thousands of them carrying a
    # long seed is an over-cap residual decided without allocating its product.
    # (Identical alternatives collapse before the cap, so the overflow has to be
    # driven by distinct ones -- see the dedup in ``_brace_alternatives``.)
    big = ",".join(f"a{i}" for i in range(20_000))
    alts = _argv_floor._brace_alternatives(big)
    assert alts is not None and len(alts) > _argv_floor._BRACE_EXPANSION_CAP
    body = "z{" + big + "}" + "y" * 100_000
    assert _argv_floor._brace_expansions(body, budget=[_argv_floor._BRACE_CHAR_CAP]) is None


def test_many_unterminated_parameter_openers_are_scanned_once() -> None:
    # Work is counted, not timed: each character is read a bounded number of times.
    from kiro_crew.security import shell_normalizer as sn

    reads = [0]

    class Counting(str):
        def __getitem__(self, key: "int | slice") -> str:  # type: ignore[override]
            reads[0] += 1
            return str.__getitem__(self, key)

    text = Counting("${ " * 2000)
    sn._mask_quoted_braces(text)
    assert reads[0] <= 20 * len(text), reads[0]


def test_connection_operand_brace_check_has_no_character_budget() -> None:
    # The body's work budget is the bare-kill caller's; the connection-target
    # check keeps its word cap alone, so a long path with a modest fan-out is
    # judged on its alternatives rather than refused for its length.
    source = "".join(f"/segment{i}" for i in range(80)) + "/archive{1..220}"
    assert len(_argv_floor._brace_expansions(source) or ()) == 220
    assert not _argv_floor._operand_targets_self(source)


def test_unrelated_expansion_overflow_stays_allowed() -> None:
    cmd = f"{_K} $({_PG} -f x{{1..9999}})"
    assert not _floor(cmd), cmd


@pytest.mark.parametrize(
    "cmd,denied",
    [
        # a word whose fan-out overflows the cap: an over-cap residual (its
        # expansions name no live process), decided without an enumeration
        (f"{_K} $({_PG} -f worker{{{{a..z}}{{a..z}}{{a..f}}" + "{a..a}" * 1200 + ",main}})", False),
        # many single-alternative groups multiplied by local-variable references:
        # ``$x`` is a ``pgrep`` argument, so ``x`` is in the protected window; its
        # resolved value passes the character budget, so the window name's overflow
        # fails the command closed.
        (
            f"{_K} $(x=worker" + "{a..a}" * 1000 + f"; {_PG} -f " + " ".join(["$x" * 40] * 3) + ")",
            True,
        ),
    ],
    ids=["wide-word", "variable-multiplied"],
)
def test_pathological_brace_work_is_bounded(
    cmd: str, denied: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The expansion work is bounded by one cumulative character budget per body,
    # so the number of group rewrites is a small constant here, not one per group
    # per word (over a hundred thousand for the second command without it).
    rewrites = [0]
    real = _argv_floor._brace_alternatives

    def counting(body: str, cap: int = _argv_floor._BRACE_EXPANSION_CAP) -> "list[str] | None":
        rewrites[0] += 1
        return real(body, cap)

    monkeypatch.setattr(_argv_floor, "_brace_alternatives", counting)
    assert _floor(cmd) is denied, cmd
    assert rewrites[0] <= 2000, rewrites[0]


def test_single_literal_resolver_declines_formats_and_multi_operand() -> None:
    from kiro_crew.security import shell_normalizer as sn

    def r(body: str) -> str:
        return sn._single_literal_output(body, _argv_floor._static_substitution_output)

    assert r(f"printf {_NAME}") == _NAME
    assert r(f"echo -n {_NAME}") == _NAME
    assert r("printf '%s@%s' a b") == "\x00"
    assert r(f"echo {_HEAD} {_TAIL}") == "\x00"
    assert r(f"printf '{_HEAD}\\n{_TAIL}'") == "\x00"


# A product of many small brace groups overflows the enumeration cap, so the
# target has to be found by reachability ACROSS the groups rather than by
# building the product.  pidof takes many names at once, so a product member
# that spells the name is a live lookup -- the fan-out does not make it benign.
_OVERFLOW_SPELLING_BODIES = [
    # one two-way group per name character, past the 256-word cap at 2**9
    "pidof -x " + "".join("{" + ch + ",x}" for ch in _NAME) + "{,x}",
    # the same across pgrep -f, with the empty-alternative tail
    f"{_PG} -f " + "".join("{" + ch + ",x}" for ch in _NAME),
    # halves built from ranges that overflow only in product
    f"pidof -x {{{_HEAD[0]}..{_HEAD[0]}}}{{{_HEAD[1]}..{_HEAD[1]}}}"
    + "".join("{" + ch + ",x}" for ch in _NAME[2:]),
    # a duplicate-heavy group that is really one word: deduped before the cap,
    # so it is enumerated and the name is found rather than left a residual
    "pidof -x {" + ",".join([_NAME] * 600) + "}",
    # an over-cap product that spells the name with a trailing digit group
    f"{_PG} -f " + "".join("{" + ch + ",x}" for ch in _NAME) + "{1,2}",
    # four letter-ranges after the head: 26**4 past the cap, yet c,r,e,w spells it
    f"{_PG} -f {_HEAD}" + "{a..z}" * 4,
]


@pytest.mark.parametrize("body", _OVERFLOW_SPELLING_BODIES)
def test_overflowing_product_that_spells_the_name_is_denied(body: str) -> None:
    cmd = f"{_K} $({body})"
    assert _floor(cmd), cmd
    assert is_denied(cmd), cmd


@pytest.mark.parametrize(
    "body",
    [
        # an over-cap product with NO name reachable across the groups stays a
        # residual, decided without building the product
        "{a,b}" * 30,
        f"{_PG} -f " + "{a,b}" * 30,
        "pidof -x other{" + ",".join(f"name{i}" for i in range(600)) + "}",
    ],
)
def test_overflowing_product_with_no_name_stays_allowed(body: str) -> None:
    assert not _floor(f"{_K} $({body})"), body


def test_overflow_reachability_is_decided_without_enumerating_the_product() -> None:
    # 2**40 is far past any machine's memory; the scan is linear in the groups,
    # so a 40-group product resolves in a blink rather than being materialised.
    from kiro_crew.security import shell_normalizer as sn

    word = "".join("{" + ch + ",x}" for ch in _NAME) + "{,x}" * 32
    assert sn._braced_word_spells_self(word)
    assert not sn._braced_word_spells_self("{a,b}" * 40)


def test_quoted_doubling_assignment_chain_fails_closed_before_building_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each assignment word is WHOLLY quoted (``"x1=${x0}${x0}"``), which a split
    # that keeps the quotes read as a command rather than an assignment, so the
    # size bound missed it and the eager resolver's doubling OOMed to an empty,
    # allowing result.  Sizing on the dequoted, operator-split stream the
    # resolver consumes bounds the chain on lengths first.
    from kiro_crew.security import shell_normalizer as sn

    def refuse(_text: str) -> "list[str]":
        raise AssertionError("the resolver must not run past the bound")

    monkeypatch.setattr(sn, "_self_tokens", refuse)
    chain = "; ".join(f'"x{i}=${{x{i - 1}}}${{x{i - 1}}}"' for i in range(1, 41))
    body = f'x0=$(printf a); {chain}; {_PG} -f "$x40"'
    assert _argv_floor._kill_body_synthesizes_self(body)


def test_quoted_doubling_chain_in_a_bare_kill_is_denied() -> None:
    # End to end: the quoted doubling chain preceding a braced name lookup is a
    # self-kill, so is_denied blocks it rather than allowing the OOM-emptied path.
    chain = "; ".join(f'"x{i}=${{x{i - 1}}}${{x{i - 1}}}"' for i in range(1, 41))
    cmd = f"{_K} $(x0=$(printf a); {chain}; {_PG} -f {_HEAD}{{{_TAIL[0]}..{_TAIL[0]}}}{_TAIL[1:]})"
    assert _floor(cmd), cmd
    assert is_denied(cmd), cmd


# ── Review-round hardening of the over-cap scan and the quoted-brace mask ──────


def test_a_wide_run_of_substitutions_does_not_stall_the_gate() -> None:
    # A 19 KB command of repeated quoted ``"$(printf …)"`` under the scan ceiling
    # once drove the shared close-paren walk quadratic (the extractor called it
    # per substitution and each re-read the whole prefix: ~54M character steps,
    # tens of seconds, past the loop watchdog). The walk now begins at the opener
    # it is handed, so the work is linear. Counted, not timed, so a loaded CI
    # runner cannot flake it: pre-fix was ~2800x the input length, so a 100x
    # ceiling fails loudly on a regression while clearing the linear path.
    from kiro_crew.security import shell_normalizer as sn

    many = ('"$(printf ' + "x" * 6 + ")" * 2) * 1200
    payload = many[:19000]
    cmd = f"{_K} $(echo {payload})"
    steps = [0]
    real = sn._iter_shell_chars

    def counting(text: str, *a: object, **k: object):  # type: ignore[no-untyped-def]
        for step in real(text, *a, **k):
            steps[0] += 1
            yield step

    original = sn._iter_shell_chars
    sn._iter_shell_chars = counting  # type: ignore[assignment]
    try:
        is_denied(cmd)
    finally:
        sn._iter_shell_chars = original
    assert steps[0] < 100 * len(cmd), steps[0]


def test_a_long_assignment_dependency_chain_resolves_linearly() -> None:
    # 1,250 chained ``v{i}=$v{i-1}`` links carrying a kill program to a final use.
    # Last-write-wins resolution walks the tokens once, resolving each use against the
    # live value, so the kill reaches the final use and is denied, in time linear in
    # the chain length. Counted, not timed, so a loaded runner cannot flake it: one
    # expansion per token plus a few, well under a 10x bound, fails loudly on an N**2
    # regression (the old closure's failure mode).
    from kiro_crew.security import shell_normalizer as sn

    n = 1250
    links = "; ".join(f"v{i}=$v{i-1}" for i in range(1, n + 1))
    cmd = f"v0={_PG_KILL}; {links}; $v{n} -f {_NAME}"
    calls = [0]
    real = sn._expand_against_values

    def counting(*a: object, **k: object):  # type: ignore[no-untyped-def]
        calls[0] += 1
        return real(*a, **k)

    sn._expand_against_values = counting  # type: ignore[assignment]
    try:
        assert is_denied(cmd), cmd  # the kill program is carried to the final use
    finally:
        sn._expand_against_values = real
    assert calls[0] < 10 * n, calls[0]


def test_an_over_cap_product_is_not_built_before_the_cap_is_checked() -> None:
    # A product of two 512-choice groups after a long literal must be refused as
    # a residual by the size check BEFORE the list is constructed, not built in
    # full (which crashed permission evaluation with MemoryError under a memory
    # limit). The call returns rather than allocating ~262k long strings.
    from kiro_crew.security import shell_normalizer as sn

    g = "{a..h}{a..h}{a..g},{a..h}{a..h}"
    assert sn._group_choices("z" * 3000 + "{" + g + "}{" + g + "}", 0) is None


def test_a_brace_sequence_step_is_honoured_in_the_over_cap_scan() -> None:
    # ``{b..d..2}`` is ``b d`` in bash, not ``b c d`` — inventing the skipped
    # ``c`` would spell a name bash never builds and deny a benign command.
    excl = f"{_K} $(pidof {_HEAD}{{b..d..2}}{_TAIL}" + "{a,b}" * 40 + ")"
    assert not _floor(excl), excl
    # a stepped sequence whose set DOES include the needed letter still denies
    incl = f"{_K} $(pidof -x {_HEAD}{{a..e..2}}{_TAIL[1:]})"  # a,c,e -> c completes
    assert _floor(incl), incl


@pytest.mark.parametrize(
    "body",
    [
        # an empty substitution between the letters is removed by the resolved
        # view, so the over-cap product still reaches the name
        f"pidof -x {_HEAD}$()c{{r,x}}{{e,x}}{{w,x}}{{a,b}}{{a,b}}{{a,b}}{{a,b}}{{a,b}}{{a,b}}",
        # a parameter default resolving to empty, likewise
        f"pidof -x {_HEAD}${{z:-}}c{{r,x}}{{e,x}}{{w,x}}{{a,b}}{{a,b}}{{a,b}}{{a,b}}{{a,b}}{{a,b}}",
    ],
)
def test_over_cap_product_is_scanned_through_the_resolved_view(body: str) -> None:
    cmd = f"{_K} $({body})"
    assert _floor(cmd), cmd
    assert is_denied(cmd), cmd


@pytest.mark.parametrize(
    "prefix",
    ["if true; then", "for x in a; do", "while :; do", "if :; then"],
)
def test_an_assignment_after_a_reserved_word_keeps_its_braces_literal(prefix: str) -> None:
    # A reserved word keeps command position, so ``pattern=kiro{3,}crew`` after it
    # is an assignment whose value braces stay literal (bash never expands them);
    # treating it as an argument expanded the braces into an invented name and
    # denied a benign command.
    cmd = f'{_K} $({prefix} pattern={_HEAD}{{3,}}{_TAIL}; {_PG} -f "$pattern"; done)'
    cmd = cmd.replace("; done)", "; fi)") if prefix.endswith("then") else cmd
    assert not _floor(cmd), cmd


def test_an_assignment_after_a_separator_and_space_keeps_its_braces_literal() -> None:
    # ``a; pattern=kiro{3,}crew`` (a space between the separator and the word)
    # is still a command-position assignment; a separator-then-space run must not
    # drop command position before the real word.
    from kiro_crew.security import shell_normalizer as sn

    masked = sn._mask_quoted_braces(f"a; pattern={_HEAD}{{3,}}{_TAIL}")
    assert "{" not in masked and "}" not in masked, masked


@pytest.mark.parametrize(
    "cmd",
    [
        # ── GPT's decoy: an argument-shaped ``name=value`` adds a value beside the
        # real binding rather than replacing it, so the real ``kiro`` still reaches
        # the ``${x}crew`` lookup and the self-kill is denied.  Three argument
        # positions: a no-op builtin, an ordinary command, and a temporary env
        # prefix.  The ``pgrep`` makes ``${x}crew`` a window argument.
        f'{_K} $(x=$(printf {_HEAD}); : x=other; {_PG} -f "${{x}}{_TAIL}")',
        f'{_K} $(x=$(printf {_HEAD}); true x=other; {_PG} -f "${{x}}{_TAIL}")',
        f'{_K} $(x=$(printf {_HEAD}); env x=other true; {_PG} -f "${{x}}{_TAIL}")',
        # ── Opus's cases: an assignment binds wherever it stands -- in a subshell,
        # a case arm, a function body, a brace group, or after a reserved word --
        # so the ``$F`` lookup resolves and the bare kill is denied.
        f"( F={_PG_KILL}; $F {_NAME} )",
        f"case $1 in start) F={_PG_KILL}; $F {_NAME} ;; esac",
        f"f(){{ F={_PG_KILL}; $F {_NAME}; }}; f",
        f"{{ F={_PG_KILL}; $F {_NAME}; }}",
        f"if true; then F={_PG_KILL}; $F {_NAME}; fi",
        # A reserved-word prefix still carries an assignment the lookup reads.
        f'{_K} $(if true; then a={_HEAD} b={_TAIL}; {_PG} -f "$a$b"; fi)',
        f'{_K} $(export a={_HEAD} b={_TAIL}; {_PG} -f "$a$b")',
        # ── Each candidate binding is scanned in its OWN argv frame, so a decoy
        # value in command position cannot push the real program into an argument
        # slot: ``P=echo; P={name}; $P token`` and the kill twin are denied.
        f"P=echo; P={_NAME}; $P to" + "ken",
        f"P=printf; P={_NAME}; $P to" + "ken",
        f"P=echo; P={_PG_KILL}; $P {_NAME}",
        f"P=echo; P={_PG_KILL}; $P -f {_NAME}",
        # The value set holds every binding of a window name (deduped) up to the
        # cap, so the real last binding sits beside the decoys and is denied:
        "; ".join(f"P=a{i}" for i in range(16)) + f"; P={_NAME}; $P to" + "ken",
        "; ".join(f"p=v{i}" for i in range(8)) + f"; p={_PG_KILL}; $p -f {_NAME}",
        # A window name bound past the value cap fails the WHOLE command closed.
        "; ".join(f"P=d{i}" for i in range(33)) + f"; P={_PG_KILL}; $P {_NAME}",
        # An assignment whose right-hand side overflows the variant cap, or whose
        # expansion exhausts the pass cap with references unresolved, fails closed
        # so a later use never resolves to nothing.
        "; ".join(
            [f"A=a{i}" for i in range(10)]
            + [f"B=b{i}" for i in range(10)]
            + [f"C=c{i}" for i in range(10)]
        )
        + f"; A={_HEAD}; B={_TAIL[:2]}; C={_TAIL[2:]}; X=$A$B$C; $X to"
        + "ken",
        f"x='.*'; n={_NAME}; {_PG_KILL} -f \"{'$x' * 65}$n\"",
        # A decoy FIRST value must not hide an ``eval`` payload a later frame carries:
        # nested-payload discovery runs over every frame.
        f"C=cat; C=eval; $C '{_PG_KILL} -f {_NAME}'",
        # ── A leading ``VAR=val`` or an env/sudo-style prefix is skipped to find the
        # real program, so a kill behind one still opens the lookup's argument window.
        f'x={_NAME}; env A=1 {_PG_KILL} -f "$x"',
        f'x={_NAME}; A=1 {_PG_KILL} -f "$x"',
        f'x={_NAME}; sudo {_PG_KILL} -f "$x"',
        f'{_K} $(x=$(printf {_HEAD}); env A=1 {_PG} -f "${{x}}{_TAIL}")',
        # A shell/eval carrier runs a payload operand, and a reserved word is followed
        # by the real program; both open the window so a variable kill behind them is
        # resolved and denied.
        f"payload='{_PG_KILL} -f {_NAME}'; bash -c \"$payload\"",
        f"c='{_PG_KILL} -f {_NAME}'; eval \"$c\"",
        f"program={_PG_KILL}; if $program -f {_NAME}; then :; fi",
        # ── A wrapper an allow-list would miss (xargs, watch, flock, time, a shell
        # function) still opens its argument window, so a variable kill behind it is
        # resolved and denied. The literal kill in a function body plus the resolved
        # argument are co-present in the one frame.
        f"n={_NAME}; xargs {_PG_KILL} -f $n",
        f"n={_NAME}; watch {_PG_KILL} -f $n",
        f"n={_NAME}; flock /tmp/l {_PG_KILL} -f $n",
        f"n={_NAME}; time -p {_PG_KILL} -f $n",
        f'f(){{ {_PG_KILL} -f "$@"; }}; p={_HEAD}; p+={_TAIL}; f "$p"',
        # ── A bare kill needs the PROGRAM word AND the target word dangerous at the
        # same time; two reassigned variables, one feeding each, must build the frame
        # that pairs them (`$p=pkill` with `$q=<name>`), which a per-slot variation
        # would never place in one frame. The product is over the variables, so the
        # pairing exists and the kill is denied.
        f"p=true; p={_PG_KILL}; q=unrelated; q={_NAME}; $p -f $q",
        f"a=echo; a={_K}; b=x; b={_NAME}; $a -9 $b",
        # ── ``set --`` loads the positional parameters a later ``"$@"`` runs, and an
        # ``alias`` operand defines a command run later; neither is inert, so a kill or
        # mint built from a variable through them is resolved and denied.
        f"P={_NAME}; set -- $P to" + 'ken; "$@"',
        f'P={_PG_KILL}; set -- $P -f {_NAME}; "$@"',
        f'shopt -s expand_aliases; P={_NAME}; alias k="{_PG_KILL} $P"; k',
        # ── An early kill resolves at ITS use site, so trailing reassignments cannot
        # drop the value it needs; the kill is denied however many follow it.
        f"KK={_PG_KILL}; $KK -f {_NAME}; " + "; ".join(f"KK=x{i}" for i in range(32)),
        # An ``echo``/``printf`` of a variable-held kill or mint piped into a shell is a
        # command: the ``$VAR`` resolves and the co-present name is caught.
        f'X={_NAME}; echo "$X to' + 'ken" | sh',
        "T=to" + f'ken; printf "{_NAME} $T" | bash',
        # ── A binding is read the way bash reads it, so a prefix bash lets precede a
        # leading assignment does not hide the binding: a redirection word, a ``time``
        # keyword and its option, and a word carrying an interior control operator all
        # leave the ``name=value`` after them in command position.
        f"2>/dev/null K={_PG_KILL}; $K {_NAME}",
        f">/tmp/x P={_NAME}; $P to" + "ken",
        f"time -p P={_PG_KILL}; $P -f {_NAME}",
        f"if true;then P={_NAME};fi; $P to" + "ken",
        # ── At a control-flow JOIN the reaching set is the UNION of every value that can
        # arrive, so a protected value assigned BEFORE a branch that may be skipped is
        # still live at a use after it, even when a benign value is assigned inside the
        # branch: an ``if`` without ``else``, a loop body (0 iterations keep the earlier
        # value), a ``case`` arm, and the right side of ``&&``/``||`` each union.
        f"P={_NAME}; if c; then P=vite; fi; {_PG_KILL} $P",
        f"P={_NAME}; while c; do P=vite; done; {_PG_KILL} $P",
        f"P={_NAME}; case $x in a) P=vite ;; esac; {_PG_KILL} $P",
        f"[ -f x ] && P={_NAME}; {_PG_KILL} $P",
        f"[ -f x ] && P={_NAME} || P=vite; {_PG_KILL} $P",
        # The protected value may come from a branch, loop, or function body bash MAY
        # OR MAY NOT run (write only inside the construct, no benign sibling), so the
        # join's union still carries it to a kill after the construct. Each conductor
        # join form is pinned individually: a one-armed ``if``, a ``case`` arm, the
        # right side of ``&&``, a ``for`` body (0 iterations keep nothing, so the body
        # value must union in), a function body invoked before the kill, and a plain
        # ``( ... )`` subshell whose assignment the parent would normally not see.
        f"if c; then P={_NAME}; fi; {_PG_KILL} $P",
        f"case $x in a) P={_NAME};; esac; {_PG_KILL} $P",
        f"true && P={_NAME}; {_PG_KILL} $P",
        f"for i in 1; do P={_NAME}; done; {_PG_KILL} $P",
        f"f(){{ P={_NAME}; }}; f; {_PG_KILL} $P",
        f"( P={_NAME} ); {_PG_KILL} $P",
        # ── Fragment assembly through a parameter expansion: ``A=pk; ${{A}}ill``
        # substitutes to the kill program, which the resolver follows like bash.
        f"A=pk; ${{A}}ill -f {_NAME}",
        # ── A ``fi``/``}`` as DATA (an argument) does not close a construct it never
        # opened, so a benign value assigned in a skipped branch does not evict the
        # protected value the branch leaves live.
        f"P={_NAME}; if false; then echo fi; P=vite; fi; {_PG_KILL} -f $P",
        # ── A write bash does not persist to the parent -- a pipeline element, a
        # backgrounded command, or a prefix assignment -- leaves the earlier value live,
        # so the kill still resolves to the protected name bash would run.
        f"P={_NAME}; echo hi | P=vite; {_PG_KILL} -f $P",
        f"P={_NAME}; P=vite true; {_PG_KILL} -f $P",
        f"P={_NAME}; P=vite & {_PG_KILL} -f $P",
        # A pipeline earlier on the line, and a spaced redirection after an assignment,
        # must not stop a later (or the same) assignment from binding.
        f"cat a | grep b; P={_NAME}; {_PG_KILL} -f $P",
        f"P={_NAME} > /dev/null; {_PG_KILL} -f $P",
        f"P={_NAME} 2> /tmp/e; {_PG_KILL} -f $P",
        # A recompute that passes a tracked value through (`P=$(echo $P)`) keeps the
        # value, so the kill/mint it feeds is still caught.
        f"P={_NAME}; P=$(echo $P); {_PG_KILL} $P",
        "T=to" + f"ken; T=$(echo $T); {_NAME} $T",
        # ── A computed RHS the resolver cannot evaluate -- a ``$(...)``- or printf-built
        # fragment glued to a literal -- makes the name's value UNKNOWN (not benign, not
        # unbound), so a kill or mint WINDOW that reads it fails closed for that window.
        f"X=$(printf 'pk')kill; $X -f {_NAME}",
        f"K=$(echo pk)kill; $K -f {_NAME}",
        f"V=$(printf 'to')ken; {_NAME} $V",
        # Fragment assembly resolved inside a kill window: `${{P}}crew` with `P=kiro`
        # spells the name even though `kiro` alone is benign.
        f"P=kiro; false && P=x; {_PG_KILL} ${{P}}crew",
        # A write inside a `$(...)` command substitution does not persist to the parent,
        # so the earlier protected value is still live at the kill.
        f"P={_NAME}; echo $( P=vite; ); {_PG_KILL} $P",
        # Only a LITERAL command word makes an assignment a prefix, so a substitution,
        # an empty expansion, an array-element assignment or a comment after it leaves
        # the value bound, as bash does.
        f"P={_NAME} $(true); {_PG_KILL} $P",
        f"P={_NAME} $NOPE; {_PG_KILL} $P",
        f"P={_NAME} A[1]=y; {_PG_KILL} $P",
        f"P={_NAME} # c\n$P to" + "ken",
        # A bare redirection's file target, and a leading array-element assignment, keep
        # command position, so the assignment that follows still binds.
        f"> /tmp/x P={_NAME}; {_PG_KILL} $P",
        f"2> /dev/null P={_NAME}; {_PG_KILL} $P",
        f"A[0]=x P={_NAME}; {_PG_KILL} $P",
        "A[0]=x T=to" + f"ken; {_NAME} $T",
        # An overflow that stops the walk before a late assembly must fail closed, not
        # resolve to the raw tokens, when a `$NAME` use survives unresolved.
        "x=aaaaaaaa; " + "x=$x$x; " * 16 + "A=pk; A+=ill; B=kiro; B+=crew; $A $B",
    ],
)
def test_an_assignment_in_any_position_is_tracked_and_denied(cmd: str) -> None:
    assert is_denied(cmd), cmd


def test_two_reassigned_variables_build_the_joint_dangerous_frame() -> None:
    # The product is over VARIABLES, not use slots, so two reassigned variables each
    # feeding one word produce the frame that pairs the kill program with the target.
    # A per-slot variation off a single base would never place both in one frame.
    from kiro_crew.security import shell_normalizer as sn

    cmd = f"p=true; p={_PG_KILL}; q=unrelated; q={_NAME}; $p -f $q"
    frames = sn._resolve_assignment_frames(sn._assignment_input(cmd.lower()))
    joined = [" ".join(f) for f in frames]
    assert any(_PG_KILL in line and _NAME in line for line in joined), joined


def test_repeated_uses_of_one_variable_do_not_fan_out() -> None:
    # One variable used many times does not multiply per use. A straight-line singleton
    # keeps the command that uses it nine times to ONE frame, so the whole line is a
    # handful of per-command frames, never 2**9 -- the ordinary reuse stays bounded.
    from kiro_crew.security import shell_normalizer as sn

    cmd = "D=/tmp/a; D=/tmp/b; $D/run $D $D $D $D $D $D $D $D"
    frames = sn._resolve_assignment_frames(sn._assignment_input(cmd.lower()))
    assert len(frames) < 2**8, len(frames)
    assert not is_denied(cmd), cmd


def test_a_pathological_value_product_is_bounded() -> None:
    # Combination-counting, not use-counting: a straight-line singleton reused many
    # times is ONE value, so a benign line (echo) stays linear in its commands -- never
    # A*B multiplied over 400 uses. A GENUINE product -- nine two-valued names in a kill
    # window -- is 2**9 combinations, past the frame cap, so that command fails closed to
    # the denied frames. Counted, not timed.
    from kiro_crew.security import shell_normalizer as sn

    setup = "; ".join([f"A=a{i}" for i in range(16)] + [f"B=b{i}" for i in range(16)])
    benign = setup + "; " + "; ".join(["echo $A$B"] * 400)
    frames = sn._resolve_assignment_frames(sn._assignment_input(benign.lower()))
    assert len(frames) < 1000, len(frames)  # linear in commands, not a product
    assert not is_denied(benign), benign
    product = "; ".join(f"V{i}=a{i}; [ -f f ] && V{i}=b{i}" for i in range(9))
    killer = product + f"; {_PG_KILL} -f {_NAME}" + "".join(f"$V{i}" for i in range(9))
    killer_frames = sn._resolve_assignment_frames(sn._assignment_input(killer.lower()))
    assert all(f in killer_frames for f in sn._fail_closed_frames()), len(killer_frames)
    assert is_denied(killer), killer


def test_a_char_budget_bomb_is_bounded_to_one_fail_closed_frame() -> None:
    # A 5 KB value reused 60 times in a window use ``pkill -f "<60 $a refs>"``: all
    # occurrences of the one name take the one value together (combination-counting), so
    # the single built string is ~300 KB. The shared character budget charges that string
    # before retaining it and overflows, so the kill command -- a window holding the
    # literal name -- fails closed rather than building an unbounded value. Counted via
    # the frame result, not timed.
    from kiro_crew.security import shell_normalizer as sn

    cmd = "a=" + "a" * 5000 + "; a=" + "b" * 5000 + f'; {_PG_KILL} -f "{_NAME}' + "$a" * 60 + '"'
    frames = sn._resolve_assignment_frames(sn._assignment_input(cmd.lower()))
    assert all(f in frames for f in sn._fail_closed_frames()), frames
    assert is_denied(cmd), cmd


def test_the_fail_closed_frames_deny_under_either_rule_alone() -> None:
    # The overflow resolves to TWO frames -- a mint and a bare kill -- so disabling
    # just one of the mint rule or the self-kill rule still leaves the other to refuse
    # the command, rather than relying on one individually-disableable rule.
    from kiro_crew.security import shell_normalizer as sn

    frames = sn._fail_closed_frames()
    mint = any("token" in f and any(sn._SELF_NAME_RE.search(w) for w in f) for f in frames)
    kill = any(
        sn._program_basename(f[0]) in sn._KILL_BY_NAME_PROGRAMS
        and any(sn._SELF_NAME_RE.search(w) for w in f[1:])
        for f in frames
    )
    assert mint and kill, frames


@pytest.mark.parametrize(
    "cmd",
    [
        # A reassignment whose values are ALL benign names nothing protected, so it
        # stays allowed -- recording every value does not pin a false positive.
        f'{_K} $(x=$(printf other); x=also; {_PG} -f "${{x}}{_TAIL}")',
        # A command-position reassignment is last-write-wins: the live value at the
        # lookup is the LAST one bash runs, so a name set earlier then overwritten with
        # a benign value is not refused for the value bash never uses.
        f'{_K} $(x=$(printf {_HEAD}); x=other; {_PG} -f "${{x}}{_TAIL}")',
        # An argument-shaped assignment whose value is benign, used in a non-killing
        # command, is recorded but spells nothing dangerous.
        f'echo a={_HEAD}; {_PG} -f "$a"',
        # A leading env assignment prefixing a benign command.
        "TMPDIR=/t ls",
        # An argument-shaped assignment to a benign value, never used.
        f"echo F={_PG_KILL}",
        "( echo F=x )",
        # A plain reassignment to benign values, each in its own frame.
        "P=echo; P=printf; $P hello",
        # ── Security Scope golden rows: a name that never reaches the protected
        # window is ignored entirely, so these ordinary scripts are not refused.
        # One name rebound past the value cap, echoed (not a window use).
        "; ".join(f"a={i}" for i in range(1, 10)) + "; echo $a",
        # Two names rebound and used several times outside the window.
        "; ".join([f"a={i}" for i in range(1, 9)] + [f"b={i}" for i in range(1, 9)])
        + "; echo $a $b $a $b",
        # Prepending PATH twice before running the tests.
        "export PATH=$HOME/bin:$PATH; export PATH=/opt/tool/bin:$PATH; python3 -m pytest test -n0 -q",
        # A shell carrier and a reserved word with benign payloads/programs.
        "payload='make test'; bash -c \"$payload\"",
        "if true; then echo hi; fi",
        # ── Security Scope golden rows, this round: a window name whose value is
        # benign, or self-referential, must not fail an ordinary command closed.
        # A two-valued interpreter at program position nine times -- the per-slot
        # frames are ten, not 2**9, so it clears the frame budget.
        "PY=python3; [ -x .venv/bin/python ] && PY=.venv/bin/python; "
        "$PY scripts/docs_lint.py && $PY -m isort --check-only src test && "
        "$PY -m flake8 src test && $PY -m mypy src/ && $PY -m pytest a.py -n0 && "
        "$PY -m pytest b.py -n0 && $PY -m build && $PY -m twine check dist/*",
        # A self-referential value expands to empty (bash's unset), not a loop, so a
        # flags variable appended to its own value is allowed.
        'PY=python3; FLAGS="$FLAGS -q"; $PY -m pytest $FLAGS test/x.py -n0',
        # Self-referential PATH kept across sudo via env; the env/sudo window names
        # PATH, but its value refers only to itself and resolves to nothing dangerous.
        'export PATH="$HOME/.local/bin:$PATH"; sudo env PATH="$PATH" make install',
        "export PATH=$PATH:/usr/sbin; sudo env PATH=$PATH iptables -L -n",
        "export LD_LIBRARY_PATH=/opt/lib:$LD_LIBRARY_PATH; "
        "nohup env LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./server --port 8080 > server.log 2>&1 &",
        # A leading self-referential assignment that no command reads is untracked,
        # so it is not resolved and cannot fail closed.
        'PATH="$PATH:/a" PATH="$PATH:/b" pytest',
        # A two-valued name used across several statements -- the per-slot frames grow
        # linearly, not as a product, so an ordinary reuse is not refused.
        "D=/tmp/a; D=/tmp/b; ls $D; echo $D; $D/run $D $D $D $D $D $D $D $D",
        # A frequently-reassigned benign variable: none of its values could make a
        # frame dangerous, so it is not resolved and the line does not fan out closed.
        "; ".join(f"f=file{i}.txt; cp $f backup/" for i in range(40)),
        "a=1; a=2; a=3; a=4; a=5; a=6; a=7; b=1; b=2; b=3; b=4; b=5; b=6; b=7; "
        "c=1; c=2; c=3; c=4; c=5; c=6; c=7; curl $a $b $c",
        # ── A variable reused across statements resolves to the value bash runs at
        # each use (last-write-wins), so an operator who checks the gateway is up then
        # kills a stale dev server by reusing one name is not refused for a value the
        # name does not hold at the kill.
        f"P={_NAME}; pgrep -fl $P; P=vite; {_PG_KILL} -f $P",
        f"svc=vite; {_PG_KILL} -f $svc; svc={_NAME}; $svc status",
        f"BIN={_NAME}; $BIN status; BIN=npm; $BIN to" + "ken list",
        # ── A branch unions BENIGN values, so a conditional reassignment whose every
        # reaching value is harmless is not refused: the union adds values, it does not
        # invent a dangerous one.
        'mode=fast; if [ -n "$CI" ]; then mode=slow; fi; echo $mode',
        "flags=-q; for f in a b; do flags=-v; done; pytest $flags",
        # ── A redirection word and a ``time`` prefix leave the following benign
        # assignment in command position without turning its value dangerous.
        "2>/dev/null LEVEL=debug; echo $LEVEL",
        "time -p DURATION=short make $DURATION",
        # A pipeline or a spaced redirection leaves a later benign assignment bindable
        # without turning its value dangerous.
        "cat a | grep b; D=/tmp/x; ls $D",
        "LEVEL=info > /dev/null; echo $LEVEL",
        # A straight-line reuse after a spaced subshell build: the subshell's `)` closes
        # even though it follows an argument, so the later reassignment is last-write.
        f"( cd web && npm run build ) ; P={_NAME}; pgrep $P; P=vite; {_PG_KILL} $P",
        # A benign same-line doubling that overflows the budget with no `$NAME` use left
        # unresolved stays allowed (nothing dangerous assembled).
        "x=aaaa; " + "x=$x$x; " * 20 + "echo done",
        # ── An ordinary build script that reuses a variable set in a branch across many
        # commands stays allowed and resolves per-command, not as a whole-line product.
        'OUT=build; if [ "$CI" = true ]; then OUT=dist; fi; '
        + "; ".join(f"cp src/f{i} $OUT/f{i}" for i in range(13)),
        # ── A grep for a signature ending in "(" before a reused-variable restart must
        # not let the quoted "(" open a phantom construct that unions the reassignment.
        f'rg -n \'def f(\' src/x; P={_NAME}; pgrep -fl "$P"; P=vite; {_PG_KILL} -f "$P"',
        "rg -n 'def main(' src/x; SUB=restart; echo skipping; SUB=status; " + f"{_NAME} $SUB",
        # ── A bootstrap toggling seven flags then running make under a timeout, and a
        # pgrep after it: window fan-out of the flags stays under the cap, no overflow.
        'A=-q; [ -n "$V" ] && A=-v; B=-j2; [ -n "$V" ] && B=-j8; C=d; [ -n "$V" ] && C=p; '
        'D=-k; [ -n "$V" ] && D=-S; E=a; [ -n "$V" ] && E=b; F=x; [ -n "$V" ] && F=y; '
        'G=1; [ -n "$V" ] && G=2; timeout 600 make run $A $B $C $D $E $F $G; pgrep -fl vite',
        # ── One single-valued name used 65+ times in a single quoted word is ONE
        # reaching value, not many: last-write-wins gives every ``$B`` the same value,
        # so the cap counts distinct value combinations, never uses. A colourised status
        # line and an ``sh -c`` mkdir of many subdirs under one base stay allowed.
        'B=build; echo -e "' + "$B/a " * 70 + '"',
        "B=build; sh -c 'mkdir -p " + " ".join(f"$B/d{i}" for i in range(70)) + "'",
    ],
)
def test_benign_assignments_stay_allowed(cmd: str) -> None:
    assert not is_denied(cmd), cmd


def test_a_reused_branch_variable_fans_out_per_command_not_per_line() -> None:
    # A variable unioned in a branch (``OUT`` -> {build, dist}) and reused across many
    # commands resolves one frame per command, so the frame count is linear in the
    # commands, never the whole-line cartesian product (2**N) that would hold the
    # synchronous gate for seconds. Counted via the frame result, not timed.
    from kiro_crew.security import shell_normalizer as sn

    cmd = 'OUT=build; if [ "$CI" = true ]; then OUT=dist; fi; ' + "; ".join(
        f"cp src/f{i} $OUT/f{i}" for i in range(13)
    )
    frames = sn._resolve_assignment_frames(sn._assignment_input(cmd.lower()))
    assert len(frames) < 2**13, len(frames)
    assert not is_denied(cmd), cmd


def test_an_if_branch_variable_reused_many_times_does_not_stall_the_gate() -> None:
    # Opus's regression shape: an 11-line build script whose one if-branch variable is
    # used in many commands. A whole-line cartesian product took ~10-20 s and crossed
    # the 25 s loop-stall watchdog; the window-scoped fan-out leaves the benign variable
    # unmultiplied. Counted via the frame result (deterministic), not wall-clock timed.
    from kiro_crew.security import shell_normalizer as sn

    cmd = 'OUT=build\nif [ "$CI" = true ]; then OUT=dist; fi\n' + "\n".join(
        f"cp src/f{i}.o $OUT/f{i}.o" for i in range(10)
    )
    frames = sn._resolve_assignment_frames(sn._assignment_input(cmd.lower()))
    assert len(frames) < 8, len(frames)  # the benign branch var does not fan out
    assert not is_denied(cmd), cmd


def test_many_dangerous_valued_names_outside_a_window_do_not_fan_out() -> None:
    # Opus's second shape: twelve two-valued names each able to hold the protected name,
    # then ``echo`` of all twelve. A value-gated fan-out doubled the frames per name to
    # 2**12 and every floor rescanned each (~44 s, past the loop-stall watchdog). The
    # fan-out is confined to kill/mint WINDOWS, and ``echo`` is not one, so the names are
    # never multiplied: the line stays a handful of frames. Counted (deterministic).
    from kiro_crew.security import shell_normalizer as sn

    cmd = (
        "; ".join(f"V{i}={_NAME}; [ -n x ] && V{i}=vite" for i in range(12))
        + "; echo "
        + " ".join(f"$V{i}" for i in range(12))
    )
    frames = sn._resolve_assignment_frames(sn._assignment_input(cmd.lower()))
    assert len(frames) < 64, len(frames)
    assert not is_denied(cmd), cmd


# ── Differential guard (conductor 03:02Z): head-denied MUST be a superset of
# base-denied. The resolver evaluates base's eager last-write deny path alongside the
# window resolver and ORs them, so the window resolver can only ADD denials, never
# suppress one. Every row below is denied on the base branch (its eager resolver
# resolves the assembled/recomputed/appended/indirect name) and must stay denied on
# head; the window resolver adds the branch-union reach on top. This list collects the
# base-denied rows the AI lanes flagged across rounds 5-17 and would FAIL on the
# pre-OR head (f06ba989bb), where the window resolver suppressed them.
_BASE_DENIED_ROWS = [
    # assembly / append across variables (eager last-write resolves both halves)
    "x=aaaaaaaa; " + "x=$x$x; " * 16 + "A=pk; A+=ill; B=kiro; B+=crew; $A $B",
    "A=pk; A+=ill; B=kiro; B+=crew; $A $B",
    f'a=kiro; b=crew; {_PG_KILL} -f "$a$b"',
    # recompute passing a tracked value through
    f"P={_NAME}; P=$(echo $P); {_PG_KILL} $P",
    f'P={_NAME}; P=$(printf vite) true; {_PG_KILL} -f "$P"',
    # prefix / non-command words after an assignment keep it bound
    f"P={_NAME} $(true); {_PG_KILL} $P",
    f"P={_NAME} $NOPE; {_PG_KILL} $P",
    f"P={_NAME} A[1]=y; {_PG_KILL} $P",
    f"P=vite; P={_NAME} sh -c 'pkill -f \"$P\"'",
    # a spaced redirection / array-element prefix keeps command position
    f"> /tmp/x P={_NAME}; {_PG_KILL} $P",
    f'2> /dev/null P={_NAME}; {_PG_KILL} -f "$P"',
    f"A[0]=x P={_NAME}; {_PG_KILL} $P",
    # the two-variable bare kill and the decoy
    f"p=true; p={_PG_KILL}; q=x; q={_NAME}; $p -f $q",
    f"P=echo; P={_NAME}; $P to" + "ken",
    # fragment assembly and append-with-branch-value
    f"A=pk; ${{A}}ill -f {_NAME}",
    f'q=other; if true; then q=crew; fi; p=kiro; p+=$q; {_PG_KILL} -f "$p"',
]


@pytest.mark.parametrize("cmd", _BASE_DENIED_ROWS)
def test_head_denies_every_base_denied_row(cmd: str) -> None:
    assert is_denied(cmd), cmd


def test_base_eager_frame_is_always_among_the_scanned_frames() -> None:
    # The structural guarantee behind the differential: whatever base's eager resolver
    # produces for a command is one of the frames the floor scans, so head cannot deny
    # a strict subset of base. (It is deduplicated when identical to a window frame.)
    from kiro_crew.security import shell_normalizer as sn

    cmd = "A=pk; A+=ill; B=kiro; B+=crew; $A $B"
    tokens = sn._assignment_input(cmd.lower())
    base_frame = sn._resolve_function_aliases(sn._resolve_local_assignments(tokens))
    scanned = sn._self_token_combinations(cmd.lower())
    assert base_frame in scanned, (base_frame, scanned)
