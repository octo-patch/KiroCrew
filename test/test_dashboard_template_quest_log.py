"""The quest-log dashboard stays factual, useful, and preview-only."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from kiro_crew.dashboard_templates import instance
from kiro_crew.dashboard_templates.manifest import ManifestError, load_template

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "src/kiro_crew/dashboard_templates/builtin/quest-log"
SAMPLE = ROOT / "test/fixtures/dashboard_templates/sample_quest-log.json"


def _walk(value: Any, path: str) -> tuple[bool, Any]:
    current = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


@pytest.fixture(scope="module")
def loaded():
    try:
        return load_template(DIRECTORY)
    except ManifestError as exc:  # pragma: no cover
        pytest.fail(f"quest-log: {exc}")


@pytest.fixture(scope="module")
def html(loaded) -> str:
    return loaded[1]


@pytest.fixture(scope="module")
def sample() -> dict[str, Any]:
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


class TestDataContract:
    def test_every_fact_comes_from_the_work_fold(self, loaded) -> None:
        manifest, _ = loaded
        folded = {name: field for name, field in manifest.fields.items() if not field.agentic}
        assert set(folded) == {"goal", "round", "last_entry_at", "items", "omitted", "forgotten"}
        assert {field.fold for field in folded.values()} == {"work"}

    def test_the_only_judgment_is_strictly_shaped_advice(self, loaded) -> None:
        manifest, _ = loaded
        agentic = {name: field for name, field in manifest.fields.items() if field.agentic}
        assert set(agentic) == {"advice"}
        shape = agentic["advice"].shape
        assert shape is not None and shape.type == "array" and shape.items is not None
        assert set(shape.items.properties or {}) == {"text", "why", "task_id"}
        assert shape.items.required == ("text", "why")

    def test_every_fold_path_resolves_against_real_work_shape(self, loaded, sample) -> None:
        manifest, _ = loaded
        work = sample["folds"]["work"]["value"]
        missing = [
            f"{name} -> {field.path}"
            for name, field in manifest.fields.items()
            if field.fold and not _walk(work, field.path or "")[0]
        ]
        assert missing == []


class TestPageOrderAndFallbacks:
    def test_sections_follow_the_persons_task_order(self, html: str) -> None:
        body = html.split("</style>", 1)[1].split("<script>", 1)[0]
        markers = [
            'class="ql-goal"',
            'id="ql-needs-title"',
            'id="ql-active-title"',
            'id="ql-advice-title"',
            'id="ql-upcoming-title"',
            'id="ql-finished"',
        ]
        positions = [body.index(marker) for marker in markers]
        assert positions == sorted(positions)

    @pytest.mark.parametrize(
        "words",
        [
            "No current goal in the work log.",
            "Nothing needs you right now.",
            "No quest is active right now.",
            "No current advice. The task facts above still stand.",
            "Advice is older than the work log, so it is hidden.",
            "No queued quest or dependency is recorded.",
        ],
    )
    def test_each_section_says_what_empty_or_stale_means(self, html: str, words: str) -> None:
        assert words in html

    def test_the_state_line_and_finished_count_are_derived(self, html: str) -> None:
        assert "active: act.length" in html
        assert "needs: need.length" in html
        assert "upcoming: next.length" in html
        assert "finished: finishedCount()" in html
        assert "forgottenCount()" in html
        for invented in ("percent", "eta", "health", "level", "rarity", "score"):
            assert re.search(rf"\b{invented}\b", html, re.IGNORECASE) is None

    def test_finished_tasks_are_counted_but_never_drawn(self, html: str) -> None:
        assert "finishedCount()" in html
        assert "openItems()" in html
        assert "drawTasks('ql-active'" in html
        assert "drawTasks('ql-upcoming'" in html


class TestTaskClassification:
    def test_needs_you_is_a_question_or_a_named_human_action(self, html: str) -> None:
        assert "item.status === 'question'" in html
        assert "item.reason === 'approval' || item.reason === 'needs_human'" in html
        assert "Approve the pending request in chat." in html
        assert "Answer this task in chat." in html
        assert "Clear the blocker named below." in html

    def test_upcoming_is_queued_or_externally_blocked(self, html: str) -> None:
        assert "return !text(item.status) || item.status === 'blocked';" in html
        assert "item.status === 'blocked' ? i18n('dependency') : i18n('queued')" in html

    def test_open_work_not_needing_the_person_or_queued_is_active(self, html: str) -> None:
        assert "text(item.state) === 'open' && !needsYou(item) && !upcoming(item)" in html

    def test_fifty_items_are_bounded_and_the_hidden_count_is_named(self, html: str) -> None:
        assert "var ROW_CAP = 12;" in html
        assert "list.length > ROW_CAP" in html
        assert "list.length - ROW_CAP" in html


class TestAdviceSafety:
    def test_only_three_current_advice_rows_can_render(self, html: str) -> None:
        assert ".slice(0, 3)" in html
        assert "names.indexOf('advice') === -1" in html
        assert "movedMs > wroteMs" in html
        assert "state: 'stale', rows: []" in html

    def test_advice_is_labelled_as_agent_advice(self, html: str) -> None:
        assert "Agent advice" in html
        assert 'data-dashboard-field="advice"' in html


class TestLinksAndUntrustedText:
    def test_every_created_link_has_all_three_safe_attributes(self, html: str) -> None:
        assert "link.href = url;" in html
        assert "link.target = '_blank';" in html
        assert "link.rel = 'noopener noreferrer';" in html
        assert "event.preventDefault();" in html
        assert "kirocrew-dashboard:open" in html

    def test_only_real_github_pull_request_urls_become_links(self, html: str) -> None:
        assert "var PR_URL = /^https:\\/\\/github\\.com" in html
        assert "PR_URL.test(url)" in html
        assert "acceptance.repo" in html

    def test_untrusted_values_only_enter_as_text(self, html: str) -> None:
        assert ".innerHTML" not in html
        assert ".outerHTML" not in html
        assert "textContent" in html
        assert "document.createTextNode" in html

    def test_private_keys_are_not_read_or_drawn(self, html: str) -> None:
        for private in ("slot_key", "session_key", "worker_session_key", ".spender"):
            assert private not in html


class TestResponsiveThemeAndMotion:
    def test_panel_and_desktop_widths_have_explicit_layouts(self, html: str) -> None:
        assert "@media (max-width: 360px)" in html
        assert "@media (min-width: 720px)" in html
        assert "overflow-wrap: anywhere" in html

    def test_dark_and_light_themes_use_host_variables(self, html: str) -> None:
        assert "var(--bg," in html
        assert "var(--card," in html
        assert "var(--text," in html
        assert "@media (prefers-color-scheme: light)" in html

    def test_reduced_motion_disables_every_animation(self, html: str) -> None:
        assert "@media (prefers-reduced-motion: reduce)" in html
        assert "animation: none !important" in html
        assert "transition: none !important" in html

    def test_no_third_party_brand_or_asset_is_present(self, html: str) -> None:
        lowered = html.lower()
        for forbidden in ("blizzard", "warcraft", "battle.net", "<img", "url(http", "fetch(", "xmlhttprequest"):
            assert forbidden not in lowered


class TestPreviewOnly:
    def test_shipping_the_template_does_not_change_the_default(self) -> None:
        assert instance.DEFAULT_TEMPLATE_ID == "project-report"

    def test_preview_and_rollback_paths_remain_the_existing_instance_paths(self) -> None:
        assert callable(instance.stage_preview)
        assert callable(instance.apply_preview)
        assert callable(instance.rollback)
