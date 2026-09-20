"""The website may not show a number that is not in a committed results file.

This is the rule that keeps the site honest: the hero table is the verbatim
output of `caliper compare`, and the metric strip repeats deltas from the
comparison JSON. If either drifts from the committed results, this fails.
"""

import html
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site" / "index.html"
RESULTS = ROOT / "results"


@pytest.fixture(scope="module")
def site() -> str:
    return SITE.read_text(encoding="utf-8")


def test_hero_table_is_verbatim_committed_output(site):
    table = (RESULTS / "reference-v2-vs-v1.txt").read_text(encoding="utf-8").rstrip("\n")
    assert html.escape(table, quote=False) in site


def test_metric_strip_figures_come_from_the_comparison_json(site):
    data = json.loads((RESULTS / "reference-v2-vs-v1.json").read_text(encoding="utf-8"))
    by_label = {m["label"]: m for m in data["metrics"]}

    success = f"{by_label['task success']['delta'] * 100:+.1f}"
    assert f'<span class="fig">{success}</span>' in site

    tool_err = f"{by_label['tool-error rate']['delta'] * 100:+.1f}"
    assert f'<span class="fig">{tool_err}</span>' in site

    cost = by_label["cost / task"]
    rel = (cost["after"] - cost["before"]) / abs(cost["before"])
    assert f'<span class="fig">{rel * 100:+.0f}%</span>' in site


def test_grader_count_on_the_site_matches_the_registry(site):
    from caliper.graders import registered

    assert f'<span class="fig">{len(registered())}</span>' in site


def test_site_has_no_author_byline(site):
    assert "Manav Arya" not in site


def test_site_is_ascii_only(site):
    assert site.isascii()


def test_site_uses_no_emoji_or_box_drawing(site):
    for ch in site:
        assert ord(ch) < 128


def test_site_loads_no_remote_assets(site):
    lowered = site.lower()
    assert "<script" not in lowered
    assert "<img" not in lowered
    # The only stylesheet is the local one.
    assert lowered.count("<link") == 1
    assert 'href="style.css"' in lowered


def test_site_links_the_sibling_projects(site):
    for repo in ("caliper", "strata", "quarry"):
        assert f"https://github.com/Manavarya09/{repo}" in site


def test_stylesheet_has_no_gradients_or_shadows():
    css = (ROOT / "site" / "style.css").read_text(encoding="utf-8").lower()
    for banned in ("gradient", "box-shadow", "text-shadow"):
        assert banned not in css


def test_reference_run_files_are_committed():
    for name in ("reference-v1.json", "reference-v2.json",
                 "reference-v2-vs-v1.txt", "reference-v2-vs-v1.json"):
        assert (RESULTS / name).exists(), f"missing committed result: {name}"


def test_committed_runs_are_the_two_reference_versions():
    v1 = json.loads((RESULTS / "reference-v1.json").read_text(encoding="utf-8"))
    v2 = json.loads((RESULTS / "reference-v2.json").read_text(encoding="utf-8"))
    assert v1["agent_version"] == "v1" and v2["agent_version"] == "v2"
    assert v1["suite"] == v2["suite"] == "reference"
    assert v2["summary"]["task_success"] > v1["summary"]["task_success"]
