"""Multi-valued ``$VAR`` resolution helpers for the shell self-protection floor.

A name reaches a use with the values bash dataflow could carry there: a straight-line
reassignment REPLACES the reaching value (last-write-wins), while a write inside a
branch that may be skipped (an ``if``/``case`` arm, a loop body, the right side of
``&&``/``||``) is UNIONED with the values already reaching the name. The resolver in
``shell_normalizer`` walks the tokens once, keeping each name's reaching value SET and
replacing every ``$VAR`` use with each value, so a command resolves to one frame per
surviving combination of its uses. These helpers do the substitution: the reference
finder, the per-text expander (``$VAR`` / ``${VAR}`` / ``${!V}`` / ``${VAR<xform>}``
against a value table, bounded by ``_ASSIGN_VARIANT_CAP`` / ``_ASSIGN_EXPAND_PASSES`` /
``_ASSIGN_CHAR_BUDGET``), a self-reference stripper (a value that names itself expands
to empty, as bash expands an unset variable), and the fail-closed frames an expansion
overflow resolves to -- a mint and a bare kill, so the mint floor or the self-kill
floor refuses a pathological input rather than letting an unresolved ``$VAR`` hide a
kill.

These are pure helpers (regex + list work, no package imports), split out so the
reader module stays under its per-file line ceiling.
"""

from __future__ import annotations

import re

from .vocabulary import _KILL_BY_NAME_PROGRAMS, _SELF_NAME_RE

_VAR_USE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")
# ``${VAR:0}`` / ``${VAR^^}`` / ``${VAR/x/y}`` TRANSFORM a variable's own value. The
# ``:-``/``:+``/``:=``/``:?`` default forms are deliberately NOT matched here -- those
# carry a literal of their own and are resolved by ``_resolve_param_defaults``.
_PARAM_TRANSFORM_RE = re.compile(r"\$\{([A-Za-z_]\w*)(?::(?![-+=?])[^}]*|[#%^,/@][^}]*)\}")
# ``${!VAR}`` expands to the value of the variable NAMED by ``VAR`` -- one hop more
# than ``${VAR}``, through the same table.
_INDIRECT_VAR_USE_RE = re.compile(r"\$\{!([A-Za-z_][A-Za-z0-9_]*)\}")

_ASSIGN_VARIANT_CAP = 1024
_ASSIGN_EXPAND_PASSES = 64
_ASSIGN_CHAR_BUDGET = 262_144
#: The protected program name itself, used as a fail-closed sentinel value.
_SELF_NAME_SENTINEL = "kiro" "crew"
#: Fail-closed argv frames a command whose expansion overflows resolves to. TWO frames
#: -- a mint ``<name> token`` and a bare kill ``pkill <name>`` -- so the overflow is
#: refused by the mint floor AND the self-kill floor; disabling either single rule
#: does not let an overflow through.
_FAIL_CLOSED_FRAMES = [[_SELF_NAME_SENTINEL, "token"], ["pkill", _SELF_NAME_SENTINEL]]


def _fail_closed_frames() -> "list[list[str]]":
    """Fresh copies of :data:`_FAIL_CLOSED_FRAMES`, the frames a window name's
    overflow resolves to -- a mint and a bare kill, so the mint floor OR the
    self-kill floor refuses it however the rules are toggled."""
    return [list(frame) for frame in _FAIL_CLOSED_FRAMES]


def _strip_self_ref(value: str, name: str) -> str:
    """*value* with a reference to *name* ITSELF removed -- resolved to empty, as bash
    expands an as-yet-unset variable -- so ``X="$X -q"`` with no prior tracked value
    stores ``-q``, not a value that names itself and never stops expanding at the use
    site. A reference the resolver already replaced with a tracked value is gone
    before this runs, so only an unresolved self-reference is dropped."""
    return re.sub(
        r"\$\{" + re.escape(name) + r"\}|\$" + re.escape(name) + r"(?![A-Za-z0-9_])",
        "",
        value,
    )


def _referenced_names(text: str) -> "set[str]":
    """The variable names *text* references via ``$VAR`` / ``${VAR}`` / ``${!V}`` /
    ``${VAR<xform>}``."""
    names = {m.group(1) or m.group(2) for m in _VAR_USE_RE.finditer(text)}
    names |= {m.group(1) for m in _INDIRECT_VAR_USE_RE.finditer(text)}
    names |= {m.group(1) for m in _PARAM_TRANSFORM_RE.finditer(text)}
    return names


def _first_tracked_ref(
    text: str, values: "dict[str, list[str]]"
) -> "tuple[int, int, list[str], str | None] | None":
    """The leftmost ``$VAR`` reference in *text* a tracked name resolves, as
    ``(start, end, replacements, name)``; ``None`` when none resolve. ``name`` is the
    plain variable name for a ``$VAR`` / ``${VAR}`` use (so EVERY occurrence of that one
    variable can be substituted together -- one variable carries one value, so N uses are
    one reaching value, not N), or ``None`` for an indirect / transform use, which is
    rewritten one occurrence at a time. ``${!V}`` indirects twice; ``${VAR<xform>}``
    resolves to VAR's value (the transform is not modelled); an unknown name is left."""
    best: "tuple[int, int, list[str], str | None] | None" = None
    for match in _INDIRECT_VAR_USE_RE.finditer(text):
        indirect = [v for nm in values.get(match.group(1), []) for v in values.get(nm, [])]
        if indirect:
            best = (match.start(), match.end(), indirect, None)
            break
    for match in _PARAM_TRANSFORM_RE.finditer(text):
        xform = values.get(match.group(1))
        if xform and (best is None or match.start() < best[0]):
            best = (match.start(), match.end(), xform, None)
            break
    for match in _VAR_USE_RE.finditer(text):
        name = match.group(1) or match.group(2)
        plain = values.get(name)
        if plain and (best is None or match.start() < best[0]):
            best = (match.start(), match.end(), plain, name)
            break
    return best


def _expand_against_values(
    text: str, values: "dict[str, list[str]]", budget: "list[int]"
) -> "list[str] | None":
    """Every literal *text* becomes once each tracked reference takes each of its
    bound values. All occurrences of ONE variable are substituted together, since a
    single variable carries a single value at a use site -- so N uses of a one-valued
    name stay ONE variant (not N), and the variant count is the product of DISTINCT
    names' value counts, never the number of uses. Returns a de-duplicated list, or
    ``None`` (the caller fails closed) on variant-count, shared-character-*budget*, or
    pass-cap-with-refs overflow."""
    if "$" not in text:
        return [text]
    variants = [text]
    for _ in range(_ASSIGN_EXPAND_PASSES):
        changed = False
        nxt: list[str] = []
        for variant in variants:
            ref = _first_tracked_ref(variant, values)
            if ref is None:
                nxt.append(variant)
                continue
            changed = True
            start, end, reps, name = ref
            for rep in reps:
                if name is None:  # indirect / transform: one occurrence at a time
                    built = variant[:start] + rep + variant[end:]
                else:  # a plain $NAME: every occurrence takes this same value at once
                    built = _substitute_var(variant, name, rep)
                budget[0] -= len(built)  # charge before building the product
                if budget[0] < 0 or len(nxt) >= _ASSIGN_VARIANT_CAP:
                    return None
                nxt.append(built)
        variants = list(dict.fromkeys(nxt))
        if not changed:
            return variants
    # Passes exhausted with references still unresolved: fail closed.
    return None if any(_first_tracked_ref(v, values) for v in variants) else variants


def _substitute_var(text: str, name: str, rep: str) -> str:
    """*text* with every ``$name`` / ``${name}`` use of the one variable *name* replaced
    by *rep* -- indirect ``${!name}`` and transform ``${name<xform>}`` forms are left for
    their own pass."""

    def _repl(match: "re.Match[str]") -> str:
        return rep if (match.group(1) or match.group(2)) == name else match.group(0)

    return _VAR_USE_RE.sub(_repl, text)


# Reserved words that OPEN a compound construct whose body may or may not run, and the
# words that CLOSE one; a write inside an open construct unions rather than replaces.
_COMPOUND_OPENERS = frozenset({"if", "while", "until", "for", "select", "case"})
_COMPOUND_CLOSERS = frozenset({"fi", "done", "esac"})
# A leading redirection word (``2>/dev/null``, ``>out``, ``<in``) is a prefix bash reads
# WITHOUT consuming command position, so a ``name=value`` after it still binds.
_REDIRECT_RE = re.compile(r"\A\d*(?:>>|&>|>&|<<<|<<|>|<)")
# ``NAME[index]=value`` / ``NAME[index]+=value`` -- a bash array-element assignment,
# which binds like a leading ``NAME=value`` (``_LOCAL_ASSIGN_RE`` does not match the
# ``[index]``), so it keeps command position and is not read as a command word.
_ARRAY_ELEMENT_ASSIGN_RE = re.compile(r"\A[A-Za-z_][A-Za-z0-9_]*\[[^\]]*\]\+?=")
# The fan-out ceiling: a multi-valued name used in a kill/mint window multiplies the
# frames, so a window with enough such names past this fails the command closed. Set so
# a handful of toggled flags (2**7) in a window still resolve, while a pathological kill
# with nine-plus two-valued names fails closed; window-gating keeps non-window commands
# (``echo $a $b …``) from ever fanning out, so this bound is only reached inside a window.
_ASSIGN_FRAME_CAP = 256
# Programs whose operand is a process target or a kill: when a command invoking one
# overflows the resolver budget, its operand is a lookup/kill WINDOW whose value cannot
# be shown benign, so the command fails closed rather than falling back.
_OVERFLOW_SENSITIVE_PROGRAMS = _KILL_BY_NAME_PROGRAMS | frozenset({"kill", "pgrep", "pidof"})
# Programs whose command is a kill, a process lookup, or a carrier that runs an operand
# as a command -- the kill/mint WINDOWS. A ``$VAR`` program is unknown, so it counts
# too. A multi-valued name is fanned out only inside such a command; elsewhere (``echo``,
# ``cp``, ``make`` …) its value cannot make a frame dangerous, so it never multiplies.
_WINDOW_PROGRAMS = _KILL_BY_NAME_PROGRAMS | frozenset(
    {
        "kill",
        "pgrep",
        "pidof",
        "sh",
        "bash",
        "dash",
        "zsh",
        "ksh",
        "eval",
        "source",
        ".",
        "xargs",
        "watch",
        "flock",
        "time",
        "timeout",
        "env",
        "sudo",
        "nohup",
        "exec",
        "command",
        "builtin",
        "set",
        "alias",
    }
)


def _is_window_word(bare: str) -> bool:
    """True if a command whose program is *bare* could be a kill or a mint -- a kill/
    lookup program, the self program, a carrier that runs an operand, or a ``$VAR``
    program (unknown, so treated as a window). Only inside such a command does a
    multi-valued name's value matter, so only there is it fanned out."""
    from . import shell_normalizer as _sn

    if bare.startswith("$"):
        return True
    base = _sn._program_basename(bare)
    return base in _WINDOW_PROGRAMS or _sn._is_self_program(bare)


def _compound_delta(bare: str, in_cmd_pos: bool) -> int:
    """The change this token makes to the open-construct nesting depth. A subshell or
    substitution OPENER (``(``, ``$(``, ``<(``, ``>(`` -- only those exact tokens, so a
    quoted argument that merely ends in ``(`` like ``rg 'def f('`` opens nothing) and its
    close ``)`` apply in ANY position: a subshell's ``)`` follows the last command's
    arguments, not a command word. The word closers/openers (``if``/``fi``/…) and a brace
    group (``{``/``}``) apply only in COMMAND position, so a literal ``fi``/``}`` argument
    closes nothing. A ``case`` arm pattern (``start)``) matches none of these."""
    delta = 0
    if bare == ")":
        delta -= 1
    if bare == "(" or bare in ("$(", "<(", ">("):
        delta += 1
    if in_cmd_pos:
        if bare in _COMPOUND_OPENERS:
            delta += 1
        elif bare in _COMPOUND_CLOSERS:
            delta -= 1
        if bare.endswith("{"):
            delta += 1
        if bare.startswith("}"):
            delta -= 1
    return delta


def _value_is_dangerous(text: str) -> bool:
    """True if *text* could make a frame a self-kill or a mint -- it holds the protected
    name (or a glob/path spelling), a by-name kill program, or the mint verb. It gates
    the fan-out and decides, on an overflow, whether to fail closed or fall back, so a
    benign reused variable neither multiplies the frames nor is refused."""
    from . import shell_normalizer as _sn

    if _SELF_NAME_RE.search(text):
        return True
    for word in text.split():
        if _sn._program_basename(word) in _KILL_BY_NAME_PROGRAMS:
            return True
        if _sn._is_self_program(word) or _sn._is_mint_verb(word):
            return True
    return False


def _overflow_frames(tokens: "list[str]", values: "list[str]") -> "list[list[str]]":
    """The frames a command whose expansion/fan-out overflowed resolves to: the two
    fail-closed frames when any reaching *value*, any token, a kill/lookup program, or a
    surviving unresolved ``$NAME`` use could be dangerous (the overflow may have stopped
    the walk before a late assembly bound a bad value), else the raw tokens -- a benign
    overflow with nothing dangerous left unresolved is allowed."""
    from . import shell_normalizer as _sn

    dangerous = any(_value_is_dangerous(v) for v in values)
    dangerous = dangerous or any(_value_is_dangerous(tok) for tok in tokens)
    dangerous = dangerous or any(
        _sn._program_basename(tok) in _OVERFLOW_SENSITIVE_PROGRAMS for tok in tokens
    )
    dangerous = dangerous or any(
        "$" in tok and not _sn._SHELL_ASSIGN_RE.match(tok.rstrip(";&|\n")) for tok in tokens
    )
    return _fail_closed_frames() if dangerous else [list(tokens)]


def _assignment_persists(tokens: "list[str]", idx: int) -> bool:
    """True if the leading assignment at *idx* changes the shell's own variable, so a
    later use sees it. False when bash runs it WITHOUT persisting: a prefix assignment
    (``NAME=value cmd`` -- set only in ``cmd``'s environment), a backgrounded command
    (``NAME=value &``), or a pipeline element, each of which runs detached. A run of
    leading assignments with no command word after it, ended by ``;``/newline or
    ``&&``/``||``, persists."""
    from . import shell_normalizer as _sn

    first = tokens[idx]
    # A value that opens a substitution spans into later tokens (``$(…)`` / backticks
    # with spaces), so a token after it is part of the value, not a trailing command
    # word; such an assignment persists rather than reading as a prefix.
    if first.count("(") != first.count(")") or first.count("`") % 2 == 1:
        return True
    j = idx + 1
    while j < len(tokens):
        token = tokens[j]
        if token.strip(";&|\n") == "":  # the command's own terminator
            return ("&&" in token) or ("||" in token) or (";" in token) or ("\n" in token)
        bare = token.rstrip(";&|\n")
        redirect = _REDIRECT_RE.match(bare)
        if _sn._SHELL_ASSIGN_RE.match(bare) or _ARRAY_ELEMENT_ASSIGN_RE.match(bare) or redirect:
            if token != bare:  # a trailing separator ends the command here
                return ("&&" in token) or ("||" in token) or (";" in token) or ("\n" in token)
            j += 1
            # A bare redirect operator (``>``, ``2>``) with no glued filename takes the
            # NEXT token as its target, so skip that too -- ``NAME=value > file`` persists.
            if redirect and bare == redirect.group(0) and j < len(tokens):
                target = tokens[j]
                if target != target.rstrip(";&|\n"):  # the target carries the terminator
                    return (
                        ("&&" in target) or ("||" in target) or (";" in target) or ("\n" in target)
                    )
                j += 1
            continue
        if bare.startswith("$") or bare.startswith("#"):
            # A substitution/expansion/comment is NOT a literal command word -- bash
            # keeps the assignment (``P=val $(true)``, ``P=val $x``, ``P=val # c``).
            return True
        return False  # a literal command word follows: this is a prefix assignment
    return True  # end of input with no command word: a standalone assignment


_FUNC_DEF_RE = re.compile(r"\A(?:function\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\(\)")


def _collect_function_aliases(tokens: "list[str]") -> "dict[str, str]":
    """The function/alias-to-protected-program map defined within *tokens*.

    ``x(){ <name> "$@";}; x <verb>`` never puts the program and the verb in one argv, so
    a function (or ``alias``) whose body invokes a protected program is treated as an
    alias for it, and the argv checks see ``<name> <verb>`` at the call site. Only a
    LITERAL body is inspected; over-approximating is the safe direction."""
    from . import shell_normalizer as _sn

    aliases: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        # ``alias x=<name>`` then ``x <verb>`` is the function-wrapper evasion in another
        # shape: the definition holds the program, the call site holds the verb.
        if tokens[i] == "alias" and i + 1 < len(tokens):
            spec = _sn._LOCAL_ASSIGN_RE.match(tokens[i + 1])
            if spec and spec.group(2):
                target = _sn._normalize_operand(spec.group(2))
                if _sn._is_self_program(target):
                    aliases[spec.group(1)] = "kirocrew"
                elif _sn._program_basename(target) in _KILL_BY_NAME_PROGRAMS:
                    aliases[spec.group(1)] = _sn._program_basename(target)
        m = _FUNC_DEF_RE.match(tokens[i])
        if m:
            for body_token in tokens[i + 1 :]:  # the body, to its closing brace
                if _sn._is_self_program(body_token):
                    aliases[m.group(1)] = "kirocrew"
                    break
                if _sn._program_basename(body_token) in _KILL_BY_NAME_PROGRAMS:
                    aliases[m.group(1)] = _sn._program_basename(body_token)
                    break
                if "}" in body_token:
                    break
        i += 1
    return aliases


def _resolve_function_aliases(
    tokens: "list[str]", aliases: "dict[str, str] | None" = None
) -> "list[str]":
    """*tokens* with each function/alias name replaced by the protected program it
    forwards to. The map is collected from *tokens* unless one is supplied -- a caller
    with several frames collects across ALL of them first, so a definition in one
    per-command frame still resolves a call in another."""
    if aliases is None:
        aliases = _collect_function_aliases(tokens)
    if not aliases:
        return tokens
    return [aliases.get(tk, tk) for tk in tokens]


#: Cap on a single tracked value in the base eager resolver, so a same-line doubling
#: (``x=$x$x`` repeated) cannot grow a value without bound and stall the gate. A name is
#: assembled in a few bytes, so a value this long holds no realistic kill/mint spelling.
_LOCAL_VALUE_CAP = 16384


def _resolve_local_assignments(tokens: "list[str]") -> "list[str]":
    """Base's eager last-write resolver, restored as the deny backstop: substitute each
    ``$VAR`` use with the LITERAL last assigned to it in the same command (expanding a
    value built from an already-tracked variable, and carrying a program/verb a ``$(…)``
    computes). It denies the assembly/recompute/append/indirect self-kills and mints the
    base branch catches, while last-write keeps a reused variable (``P=<name>; …; P=vite;
    pkill $P``) allowed. A per-value cap bounds a doubling bomb the base lacked a guard
    for. The window resolver runs ALONGSIDE this and only ADDS denials (the branch-union
    reach); the floor scans both frames, so head denies a superset of what base denies."""
    from . import shell_normalizer as _sn

    values: dict[str, str] = {}
    out: list[str] = []
    tokens = _sn._split_glued_operators(tokens)

    def _store(name: str, value: str) -> None:
        values[name] = value[:_LOCAL_VALUE_CAP]

    for idx, token in enumerate(tokens):
        assign = _sn._LOCAL_ASSIGN_RE.match(token)
        append = None if assign else _sn._SHELL_ASSIGN_RE.match(token)
        if append and append.group(2):
            name, tail = append.group(1), append.group(3)
            if values and "$" in tail:
                tail = _VAR_USE_RE.sub(
                    lambda m: values.get(m.group(1) or m.group(2), m.group(0)), tail
                )
            if _sn._is_computed_value(tail):
                produced = _sn._protected_name_in_substitution(tokens, idx)
                if produced:
                    _store(name, produced)
                out.append(token)
                continue
            piece = tail.strip("\"'").rstrip(";&|")
            _store(name, values.get(name, "") + piece)
            out.append(token)
            continue
        if assign and values and "$" in (assign.group(2) or ""):
            expanded = _VAR_USE_RE.sub(
                lambda m: values.get(m.group(1) or m.group(2), m.group(0)),
                assign.group(2),
            )
            if expanded != assign.group(2):
                token = f"{assign.group(1)}={expanded}"
                assign = _sn._LOCAL_ASSIGN_RE.match(token)
        if assign and _sn._is_computed_value(assign.group(2)):
            produced = _protected_name_in_substitution(tokens, idx)
            if produced:
                _store(assign.group(1), produced)
            elif _mint_verb_in_substitution(tokens, idx):
                _store(assign.group(1), "token")
            out.append(token)
            continue
        if assign and assign.group(2):
            value = assign.group(2).strip("\"'").rstrip(";&|")
            if value:
                _store(assign.group(1), value)
            out.append(token)
            continue
        if values and "$" in token:
            token = _PARAM_TRANSFORM_RE.sub(lambda m: values.get(m.group(1), m.group(0)), token)
            token = _INDIRECT_VAR_USE_RE.sub(
                lambda m: values.get(values.get(m.group(1), ""), m.group(0)), token
            )
            token = _VAR_USE_RE.sub(
                lambda m: values.get(m.group(1) or m.group(2), m.group(0)), token
            )
        out.append(token)
    return out


def _protected_name_in_substitution(tokens: "list[str]", start: int) -> str:
    """The protected program name a substitution starting at *start* could produce.

    Scans forward until the substitution closes (``shlex`` splits it across tokens
    because it splits on whitespace only) and returns the product name, or a by-name
    kill program, if either appears inside it; returns "" when neither does. The depth
    reads the shared quote-aware walk with the quote state carried ACROSS tokens, so a
    quoted ``)`` inside the substitution does not close it early and hide a later name."""
    from . import shell_normalizer as _sn

    depth = 0
    state = 0
    ansi = False
    for token in tokens[start:]:
        walk = _sn._shell_quote_walk(token, state=state, ansi=ansi)
        depth += walk.paren_delta
        state, ansi = walk.end_state, walk.end_ansi
        m = _SELF_NAME_RE.search(token)
        if m:
            return m.group(0)
        for verb in _KILL_BY_NAME_PROGRAMS:
            if verb in token:
                return verb
        if depth <= 0 and state == 0 and token is not tokens[start]:
            break
    return ""


def _mint_verb_in_substitution(tokens: "list[str]", idx: int) -> bool:
    """True if the substitution starting at *idx* prints the credential-minting verb --
    the VERB twin of :func:`_protected_name_in_substitution` (``T=$(printf <verb>);
    <name> $T``)."""
    from . import shell_normalizer as _sn

    joined = " ".join(tokens[idx:])
    for body in _sn._substitution_bodies(joined):
        if any(_sn._is_mint_verb(word) for word in body.split()):
            return True
    return False
