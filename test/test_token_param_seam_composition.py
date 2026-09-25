"""How the companion URL-prefix seam composes with pass 4's coalescing.

The seam (``_token_param_exempt``, ``defer_token_param``) governs a ``token=``
value NO earlier pass touched: under a configured prefix such a value is exempt,
and under the streamed-delta keyword it is deferred. The coalescing governs a
value an earlier pass claimed only PARTLY: it is one tag, never a tag per gap,
so the redactor's output is a fixed point of itself. The two decide disjoint
values, and the seam is judged first on the values it governs, so this file
pins that neither changes the other's answer.
"""

from __future__ import annotations

import dataclasses

import pytest

from kiro_crew.security import redact_credentials

_PREFIX = "https://approve.example.com/auth"
_SECRET = "SECRETVALUE0123456789abcdefghijklmnopqrstuv"
_AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
_OPAQUE = "opaquesecretvalue123"
_TAG = "[REDACTED: credential]"


class _StubCredentialPolicy:
    def __init__(self, prefixes: frozenset[str]) -> None:
        self._prefixes = prefixes

    def token_param_exempt_url_prefixes(self) -> frozenset[str]:
        return self._prefixes


def _install(prefixes: frozenset[str]) -> None:
    from kiro_crew.config import KiroCrewConfig
    from kiro_crew.platform.bootstrap import build_default_context
    from kiro_crew.platform.context import set_context

    base = build_default_context(KiroCrewConfig())
    set_context(dataclasses.replace(base, credentials=_StubCredentialPolicy(prefixes)))


@pytest.fixture(autouse=True)
def _reset_platform_context():
    from kiro_crew.platform.context import reset_context

    yield
    reset_context()


def test_a_whole_value_under_a_configured_prefix_is_exempt_as_on_the_base() -> None:
    # The seam's own answer: a value no pass touched, behind a configured companion
    # prefix, is left alone; the same value behind another host is redacted.
    _install(frozenset({_PREFIX}))
    exempt = f"Open {_PREFIX}?token={_SECRET} now."
    out, warnings = redact_credentials(exempt)
    assert out == exempt, out
    assert not any("token parameter" in w for w in warnings)
    other = f"Open https://other.example.net/cb?token={_SECRET} now."
    out, warnings = redact_credentials(other)
    assert out == f"Open https://other.example.net/cb?token={_TAG} now.", out
    assert any("token parameter" in w for w in warnings)


def test_a_partly_claimed_value_still_coalesces_to_one_tag_under_a_prefix() -> None:
    # The coalescing's own answer, with the seam installed: a value pass 1 claimed
    # partly (an AWS key id heading an opaque tail) is one tag, with or without
    # the companion prefix in front of it, and with or without the deferral
    # keyword; the seam governs only a value no pass touched.
    _install(frozenset({_PREFIX}))
    for text in (
        f"Open {_PREFIX}?token={_AWS_KEY}{_OPAQUE} now.",
        f"Open https://other.example.net/cb?token={_AWS_KEY}{_OPAQUE} now.",
    ):
        for kwargs in ({}, {"defer_token_param": True}):
            out, warnings = redact_credentials(text, **kwargs)
            assert _OPAQUE not in out and _AWS_KEY not in out, (text, kwargs, out)
            assert out.count(_TAG) == 1, (text, kwargs, out)
            assert _TAG + _TAG not in out, (text, kwargs, out)
            assert out == text.replace(f"{_AWS_KEY}{_OPAQUE}", _TAG), (text, kwargs, out)


def test_the_redacted_output_is_a_fixed_point_with_the_seam_installed() -> None:
    # A second run over the redacted text changes nothing and warns nothing:
    # the one-tag output of a partly claimed value and the untouched exempt
    # value are both stable under the seam.
    _install(frozenset({_PREFIX}))
    text = f"A {_PREFIX}?token={_SECRET} and b?token={_AWS_KEY}{_OPAQUE} end."
    once, _ = redact_credentials(text)
    twice, warnings = redact_credentials(once)
    assert twice == once, (once, twice)
    assert not any("token parameter" in w for w in warnings), warnings
