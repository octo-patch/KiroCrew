"""Argv floor for the two recursive-force ``rm`` catalog rules.

The catalog patterns ``rm -rf /.*`` and ``rm -rf ~.*`` match one contiguous
spelling. A shell runs the same wipe for ``rm -fr /``, ``rm -r -f /``,
``rm --recursive --force /`` and ``rm -rf $HOME``, so this floor reads each ``rm``
command's own words instead: recursive and force in any packing, order or long
form, against the root or the home directory ITSELF.

Scope is deliberately that narrow. A descendant (``rm -fr /tmp/build``,
``rm -rf $HOME/.cache``) is left to the catalog regex exactly as it reads today,
and wrappers or interpreter bodies are left to the regex and the OS sandbox.

Frames descend through unexpanded shell payloads, so ``bash -c`` and ``$(...)``
bodies are read as commands of their own without HOME expansion changing their
length. Each frame is split into words on the shared quote state machine
(``_iter_shell_chars``), keeping which characters were quoted: a
quoted ``'~'`` or ``'$HOME'`` is a literal name, not the home directory, and a
quoted ``'/*'`` is not a glob. Comments, heredoc bodies and redirections are not
``rm`` operands. A body only a data consumer receives (``echo bash -c '...'``), or
a substitution inside single quotes, a comment or a quoted heredoc, prints rather
than runs: a frame counts only when some executed frame yields it.
"""

from __future__ import annotations

import os
import posixpath
import re
from typing import NamedTuple

from . import shell_normalizer as _shell
from .denied_rules import BUILTIN_DENIED_RULES

#: The catalog row each wipe target maps to. The floor is gated on that row's
#: pattern, so an operator who disables the row disables the floor with it.
_RM_FLOOR_RULE_IDS: dict[str, str] = {
    "root": "local-destructive-rm-rf-root",
    "home": "local-destructive-rm-rf-home",
}
_RM_FLOOR_BY_KIND: dict[str, tuple[str, str]] = {
    kind: (rule.id, rule.pattern)
    for kind, rule_id in _RM_FLOOR_RULE_IDS.items()
    for rule in BUILTIN_DENIED_RULES
    if rule.id == rule_id
}
_RM_FLOOR_NOTE = (
    "Matched structurally on the command's argv, not by the pattern text above: "
    "rm with recursive and force flags, in any spelling, targets this directory itself."
)

_ROOT_RE = re.compile(r"/+")
# ``${HOME:-x}`` / ``${HOME:?}`` / ``${HOME:=x}`` are HOME whenever it is set;
# ``${HOME:+x}`` is the alternate text instead, so it is not listed.
_HOME_VAR_RE = re.compile(r"\$(?:HOME(?![A-Za-z0-9_])|\{HOME(?::?[-?=][^}]*)?\})")
_OPERATORS = ";&|\n()"
_SUBSTITUTION_OPENERS = ("$(", "<(", ">(", "`")
# Words that end one pipeline (a ``&`` after ``|`` is ``|&``, still the same pipe),
# and words that group commands so one pipe can read them all.
_PIPELINE_ENDS = frozenset({";", "&&", "||", "&", "\n"})
_GROUP_WORDS = frozenset({"(", ")", "{", "}", "do", "done", "then", "fi", "esac"})

#: One word: its text with quotes removed, and per character whether the shell
#: would expand it (``u`` unquoted, ``d`` inside double quotes, ``s`` literal).
_Word = tuple[str, str]


class _Scan(NamedTuple):
    words: "list[_Word]"
    #: Offsets (in the decoded text) of ``$`` / `` ` `` / ``<`` / ``>`` that can
    #: open a substitution the shell runs.
    live: "set[int]"
    text: str
    #: Quote-open/close offsets within each word, including empty quoting runs.
    quote_breaks: "dict[int, set[int]]"
    #: Indexes of words inside a ``((…))`` arithmetic expression: never a program.
    arith_words: "frozenset[int]" = frozenset()


def _heredoc_end(source: str, start: int, tags: "list[tuple[str, bool, bool]]") -> int:
    """Offset just past the heredoc bodies that begin at *start*."""
    pos = start
    for tag, strip_tabs, _expands in tags:
        while pos < len(source):
            end = source.find("\n", pos)
            line = source[pos : len(source) if end == -1 else end]
            pos = len(source) if end == -1 else end + 1
            if (line.lstrip("\t") if strip_tabs else line) == tag:
                break
    return pos


def _scan(source: str) -> _Scan:
    """Split one command line into words and control operators, keeping quote kinds.

    ANSI-C and locale quoting are resolved first by the shared decoder, which
    re-quotes any value that must stay literal. An unquoted ``#`` at a word start
    drops the rest of the line, and a heredoc body is dropped through its tag line;
    an unquoted-tag heredoc body still expands substitutions, a quoted one does not.
    """
    source = _shell._decode_shell_quoted_literals(_shell._fold_line_continuations(source))
    words: list[_Word] = []
    quote_breaks: dict[int, set[int]] = {}
    live: set[int] = set()
    text: list[str] = []
    kinds: list[str] = []
    started = False
    # (tag, strip tabs, body expands)
    tags: list[tuple[str, bool, bool]] = []
    base = 0
    # One entry per unclosed ``(``, True inside ``((`` / ``$((`` arithmetic, where
    # ``<<`` is a shift: no heredoc tag is read there. A ``$(`` or subshell keeps them.
    parens: list[bool] = []
    # (offset, first word) per unclosed ``(``. Bash reads ``((`` as arithmetic only
    # when its inner ``(`` closes right before the outer one: ``((a))`` is math,
    # ``((a) )`` and ``((a) || (b))`` are subshells.
    opens: list[tuple[int, int]] = []
    arith_words: set[int] = set()

    def _end() -> None:
        nonlocal started
        if started:
            word = ("".join(text), "".join(kinds))
            arith = bool(parens and parens[-1])
            prev = words[-1] if words and not arith else ("", "")
            # Any quote in the tag, even an empty ``E''OF``, makes the body literal.
            breaks = quote_breaks.get(len(words), set())
            if prev[0] in ("<<", "<<-") and prev[1] in ("uu", "uuu"):
                tags.append((word[0], prev[0] == "<<-", set(word[1]) <= {"u"} and not breaks))
            elif word[0].startswith("<<") and word[1][:2] == "uu" and word[0][2:3] != "<":
                marker = _shell._heredoc_marker(word[0])
                if marker and not arith:
                    dash = word[0].startswith("<<-")
                    cut = 3 if dash else 2
                    plain = set(word[1][cut:]) <= {"u"} and all(at < cut for at in breaks)
                    tags.append((marker, dash, plain))
            words.append(word)
        text.clear()
        kinds.clear()
        started = False

    while base <= len(source):
        restart = None
        comment = False
        for step in _shell._iter_shell_chars(source[base:]):
            char = step.char
            at = base + step.offset
            if comment:
                if not (step.active and char == "\n"):
                    continue
                comment = False
            if len(step.text) == 1 and step.state != 1 and char in "$`<>":
                live.add(at)
            if len(step.text) == 2 or step.trailing_escape:
                text.append(char)
                kinds.append("s")
                started = True
            elif char in "'\"" and (step.active or step.state == 0):
                quote_breaks.setdefault(len(words), set()).add(len(text))
                started = True
            elif not step.active:
                text.append(char)
                kinds.append("d" if step.state == 2 else "s")
            elif char == "#" and not started:
                comment = True
                live.discard(at)
            elif char.isspace() and char != "\n":
                _end()
            elif (
                char == "&"
                and source[at + 1 : at + 2] == ">"
                or (char == "&" and text and text[-1] in "<>")
            ):
                text.append(char)
                kinds.append("u")
                started = True
            elif char == "(" and text and text[-1] == "=":
                # ``a=(...)`` is an array assignment, not a subshell.
                text.append(char)
                kinds.append("u")
                parens.append(False)
                opens.append((at, len(words)))
            elif char in _OPERATORS:
                _end()
                if char == "(":
                    parens.append(bool(parens and parens[-1]) or source[at + 1 : at + 2] == "(")
                    opens.append((at, len(words)))
                elif char == ")" and parens:
                    parens.pop()
                    inner, _first = opens.pop()
                    if opens and opens[-1][0] == inner - 1 and source[at + 1 : at + 2] == ")":
                        # Through both closing ``)``: this one and the next, glued.
                        arith_words.update(range(opens[-1][1], len(words) + 2))
                if words and words[-1] == (char, "u") and char in "&|":
                    words[-1] = (char * 2, "uu")
                else:
                    words.append((char, "u"))
                if char == "\n" and tags:
                    # Bodies follow in tag order; only an unquoted tag's body expands.
                    restart = at + 1
                    for tag in tags:
                        end = _heredoc_end(source, restart, [tag])
                        if tag[2]:
                            live |= {i for i in range(restart, end) if source[i] in "$`"}
                        restart = end
                    tags.clear()
                    break
            else:
                text.append(char)
                kinds.append("u")
                started = True
        if restart is None:
            break
        base = restart
    _end()
    return _Scan(words, live, source, quote_breaks, frozenset(arith_words))


def _split_redirect(word: _Word) -> "tuple[_Word | None, bool]":
    """The operand glued before an unquoted redirection, and whether a target follows.

    ``~>/dev/null`` is the operand ``~`` plus a redirection; ``2>x`` is only a
    redirection (a leading fd number is part of the operator).
    """
    text, kinds = word
    cut = next((i for i, (c, k) in enumerate(zip(text, kinds)) if c in "<>" and k == "u"), -1)
    if cut == -1:
        return word, False
    if cut and text[cut - 1] == "&" and kinds[cut - 1] == "u":
        cut -= 1
    head = text[:cut]
    operand = None if not head or head.isdigit() else (head, kinds[:cut])
    # The LAST redirection decides: ``0</dev/null<`` still takes the next word.
    tail = len(text) - len(text.rstrip("<>&"))
    return operand, bool(tail) and "u" * tail == kinds[-tail:] and "&" != text[-tail:]


def _flag_kinds(token: str) -> "set[str]":
    """The ``recursive`` / ``force`` meanings an ``rm`` option word carries."""
    if token.startswith("--"):
        # GNU accepts any unambiguous prefix of a long option; three characters
        # is the shortest prefix that names only one of these two.
        name = token.split("=", 1)[0]
        if len(name) >= 3 and "--recursive".startswith(name):
            return {"recursive"}
        if len(name) >= 3 and "--force".startswith(name):
            return {"force"}
        return set()
    kinds = set()
    if "r" in token[1:] or "R" in token[1:]:
        kinds.add("recursive")
    if "f" in token[1:]:
        kinds.add("force")
    return kinds


def _is_dir_itself(rest: str) -> bool:
    """True when *rest*, after a directory spelling, still names that directory."""
    # ``${HOME}.`` is a sibling named ``<home>.``: only ``/`` starts a path inside it.
    return (not rest or rest[0] == "/") and bool(_ROOT_RE.fullmatch(posixpath.normpath("/" + rest)))


def _folded(path: str) -> str:
    # Only a Windows path ignores case and takes either separator; a POSIX path
    # keeps its case and its literal backslashes.
    if os.name == "nt":
        return posixpath.normpath(path.replace("\\", "/")).casefold()
    return posixpath.normpath(path)


def _target_kind(word: _Word, homes: "frozenset[str]", quote_breaks: "set[int]") -> "str | None":
    """``"root"`` or ``"home"`` when *word* names that directory or the glob over it."""
    text, kinds = word
    if text.endswith("/*") and kinds[-1] == "u":
        text, kinds = text[:-1], kinds[:-1]
    if not text:
        return None
    if text.startswith("/") and _ROOT_RE.fullmatch(posixpath.normpath(text)):
        return "root"
    # Bash expands ``~`` only at the word's start with the prefix up to the first
    # ``/`` all unquoted, so ``~''`` and ``''~`` are a literal ``~``.
    if (
        text.startswith("~")
        and kinds[0] == "u"
        and (len(text) == 1 or text[1] == "/")
        and not any(at <= 1 for at in quote_breaks)
    ):
        return "home" if _is_dir_itself(text[1:]) else None
    match = _HOME_VAR_RE.match(text)
    # The ``$``, brace and name must come from ONE quoting run: ``"$HO""ME"`` expands
    # HO then ME. A quote inside a ``${HOME:-"x"}`` default still expands HOME.
    name_end = 6 if text.startswith("${") else 5
    if (
        match
        and len(set(kinds[:name_end])) == 1
        and kinds[0] != "s"
        and not any(0 < at < name_end for at in quote_breaks)
    ):
        return "home" if _is_dir_itself(text[match.end() :]) else None
    # Literal home paths also count; the Windows word split can read backslashes
    # as escapes, so ``homes`` includes that spelling.
    if _folded(text) in homes:
        return "home"
    return None


def _frame_targets(scan: _Scan, homes: "frozenset[str]") -> "set[str]":
    """Which of root / home a recursive-force ``rm`` in one command line deletes."""
    words = scan.words
    # An empty quoted word (``''``) is an argument bash passes, never an operator.
    # Words inside ``((…))`` are math: an argument like any other, never a program.
    math = scan.arith_words
    ends = [
        bool(text)
        and i not in math
        and (_shell._ends_argv(text) or text == ")")
        and set(kinds) <= {"u"}
        for i, (text, kinds) in enumerate(words)
    ]
    # A quoted word that merely LOOKS like an operator is data; an unquoted reserved
    # word in command position (``then``, ``!``, ``time -p``) passes it to the next word.
    base = _shell._KEEPS_COMMAND_POSITION
    names, at_command, keep = [], True, base
    for i, (text, kinds) in enumerate(words):
        kept = at_command and set(kinds) <= {"u"} and text in keep
        keep = base | {"-p", "--"} if kept and text == "time" else base
        at_command = kept or ends[i]
        plain_word = i not in math and (ends[i] or not _shell._ends_argv(text))
        names.append(";" if kept else text if plain_word else "_")
    programs = _shell._argv_programs(names)
    found: set[str] = set()
    flags: set[str] = set()
    targets: set[str] = set()
    options_done = False
    skip_next = False
    for index, word in enumerate(words):
        is_rm_argument = (
            index > 0
            and programs[index].lower() == "rm"
            and programs[index - 1].lower() == "rm"
            and not ends[index]
        )
        if is_rm_argument and skip_next:
            skip_next = False
        elif is_rm_argument:
            operand, skip_next = _split_redirect(word)
            text = operand[0] if operand else ""
            if operand is None:
                pass
            elif not options_done and text == "--":
                options_done = True
            elif not options_done and text.startswith("-") and len(text) > 1:
                flags |= _flag_kinds(text)
            else:
                target = _target_kind(operand, homes, scan.quote_breaks.get(index, set()))
                if target:
                    targets.add(target)
        if ends[index] or index + 1 == len(words):
            if {"recursive", "force"} <= flags:
                found |= targets
            flags, targets, options_done, skip_next = set(), set(), False, False
    return found


def _live_substitution_bodies(scan: _Scan, source: str) -> "set[str]":
    """Substitution bodies of *source* that run: one with at least one live opener."""
    text = scan.text
    bodies: set[str] = set()
    # Bodies come from the scanned text the live offsets index, so a ``$'…'`` value
    # is its decoded literal. An empty body runs nothing (and finding it would visit
    # every offset); each distinct body is located once, however often it repeats.
    found = set(_shell._substitution_bodies(text))
    for body in sorted(found - {""}):
        spots = []
        start = text.find(body)
        while start != -1:
            spots += [
                start - len(o) for o in _SUBSTITUTION_OPENERS if text.startswith(o, start - len(o))
            ]
            start = text.find(body, start + 1)
        # A body not found as written cannot be proven inert, so it is kept.
        live = [spot for spot in spots if spot in scan.live]
        if spots and not live:
            continue
        # Only ``$((…))`` is math; a `` `(…)` `` or ``<(…)`` body is a subshell.
        if live and _is_math(body) and all(text.startswith("$(", spot) for spot in live):
            bodies |= {_shell._decode_printf_escapes(b) for b in _math_bodies(body)}
        else:
            bodies.add(_shell._decode_printf_escapes(body))
    return bodies


def _is_math(body: str) -> bool:
    """True when ``$(`` + *body* + ``)`` is ``$((…))``: the inner ``(`` closes last."""
    if not body.startswith("("):
        return False
    end, proven = _shell._matching_close_paren(body, 1)
    return proven and end == len(body)


def _math_bodies(body: str) -> "set[str]":
    """The substitutions inside a ``$((…))`` expression, nested math unwrapped."""
    found: set[str] = set()
    for inner in set(_shell._substitution_bodies(body[1:-1])) - {""}:
        found |= _math_bodies(inner) if _is_math(inner) else {inner}
    return found


def _unquoted_ops(word: _Word) -> str:
    """*word* with each quoted ``;`` ``&`` ``|`` masked: in quotes they start no command."""
    return "".join("_" if c in ";&|\n" and k != "u" else c for c, k in zip(*word))


def _pipeline_disqualified(scan: _Scan) -> "list[bool]":
    """Per word: whether its own pipeline refuses the data-consumer exemption.

    ``echo bash -c '…'; echo : | sh`` pipes only the second command into ``sh``.
    A group or compound word (``{ …; } | sh``, ``do … done | sh``) can feed every
    inner command to one pipe, so such a frame is judged whole, as before.
    """
    words = scan.words
    tokens = [text for text, _kinds in words]
    plain = [set(kinds) <= {"u"} for _text, kinds in words]
    if any(p and t in _GROUP_WORDS for t, p in zip(tokens, plain)):
        return [_shell._data_consumer_command_disqualified(tokens)] * len(tokens)
    verdicts: list[bool] = []
    start = 0
    for index in range(len(tokens) + 1):
        cut = index == len(tokens) or (
            plain[index]
            and tokens[index] in _PIPELINE_ENDS
            and not (index and tokens[index - 1] == "|")
        )
        if cut:
            verdict = _shell._data_consumer_command_disqualified(tokens[start:index])
            verdicts += [verdict] * (index - start) + ([True] if index < len(tokens) else [])
            start = index + 1
    return verdicts


def _executed_children(
    scan: _Scan, source: str, tokens: "list[str]", joined: "set[str]"
) -> "set[str]":
    """The bodies this frame RUNS: every payload except one only a data consumer receives."""
    payloads = _shell._nested_shell_payloads(
        tokens, allow_join=source not in joined, joined_out=joined
    )
    programs = _shell._argv_programs(tokens)
    disqualified = _pipeline_disqualified(scan) if payloads else []
    positions: dict[str, list[int]] = {}
    for index, token in enumerate(tokens):
        positions.setdefault(token, []).append(index)
    children: set[str] = set()
    for payload in payloads:
        occurrences = positions.get(payload, [])
        if occurrences and all(
            _shell._data_consumer_exempt(
                i,
                _unquoted_ops(scan.words[i]),
                programs,
                tokens,
                command_disqualified=disqualified[i],
            )
            for i in occurrences
        ):
            continue
        children.add(_shell._decode_printf_escapes(payload))
    return children | _live_substitution_bodies(scan, source)


def _rm_wipe_targets(text: str) -> "set[str]":
    """Which of root / home a recursive-force ``rm`` in *text* deletes.

    *text* is the command as submitted: ``$HOME`` is case-sensitive, so a
    lowercased copy would read an unrelated ``$home`` variable as home.
    """
    home = os.path.expanduser("~")
    homes = frozenset({_folded(home), _folded(home.replace("\\", ""))}) - {".", "~"}
    pending = [text]
    seen = {text}
    # Joining an eval suffix again would build a quadratic chain of frames.
    joined: set[str] = set()
    found: set[str] = set()
    while pending:
        source = pending.pop()
        scan = _scan(source)
        found |= _frame_targets(scan, homes)
        tokens = [word for word, _kinds in scan.words]
        for child in _executed_children(scan, source, tokens, joined):
            # Unexpanded bodies shrink at every step, regardless of HOME's length.
            if len(child) < len(source) and child not in seen:
                seen.add(child)
                pending.append(child)
    return found
