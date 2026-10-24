"""
Prompt-mode and summary-merge-mode tests.

Verifies:
- PromptMode.NEUTRAL produces a system prompt without composition warnings.
- PromptMode.CAUTIOUS produces a system prompt with the composition warning.
- SummaryMergeMode.COLLAPSED produces a single bullet list with no
  per-tenant grouping signal.
- SummaryMergeMode.PER_CLIENT_BLOCKS produces multiple per-tenant blocks.
- Both flags propagate through run_probe correctly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mnembound.agent_io import (
    PromptMode,
    build_prompt,
    run_probe,
    stub_agent_always_recall,
)
from mnembound.generator import generate_corpus
from mnembound.retrieval import (
    Rung,
    SummaryMergeMode,
    render_summary_text,
    retrieve,
)
from mnembound.schema import ProbeType
from mnembound.scoring import ValueView


def _make_corpus():
    return generate_corpus(
        seed=42, n_tenants=5, records_per_tenant=20,
        n_supported_probes=5, n_cross_boundary_probes=10,
    )


# ----- PromptMode -----


def test_neutral_prompt_has_no_composition_warnings() -> None:
    """The NEUTRAL prompt must not tell the agent to avoid combining events."""
    corpus = _make_corpus()
    p = corpus.probes[0]
    result = retrieve(Rung.SHARED_VECTOR_INDEX, p, corpus)
    prompt = build_prompt(
        p, result, tenant_visible=False, prompt_mode=PromptMode.NEUTRAL
    )
    assert "Do not combine information from different events" not in prompt.system
    assert "abstain" in prompt.system.lower()  # declining is offered in both prompts
    assert prompt.prompt_mode == PromptMode.NEUTRAL
    print("[OK] NEUTRAL prompt has no composition warnings")


def test_cautious_prompt_has_composition_warning() -> None:
    """The CAUTIOUS prompt explicitly tells the agent to avoid combining events."""
    corpus = _make_corpus()
    p = corpus.probes[0]
    result = retrieve(Rung.SHARED_VECTOR_INDEX, p, corpus)
    prompt = build_prompt(
        p, result, tenant_visible=False, prompt_mode=PromptMode.CAUTIOUS
    )
    assert "Do not combine information from different events" in prompt.system
    assert prompt.prompt_mode == PromptMode.CAUTIOUS
    print("[OK] CAUTIOUS prompt contains composition warning")


def test_run_probe_threads_prompt_mode() -> None:
    """The prompt_mode kwarg is honored by run_probe."""
    corpus = _make_corpus()
    p = corpus.probes[0]
    result = retrieve(Rung.SHARED_VECTOR_INDEX, p, corpus)

    seen_systems: list[str] = []

    def capture_agent(system, user):
        seen_systems.append(system)
        return stub_agent_always_recall(p, result, value_view=ValueView.TENANT_HIDDEN)

    run_probe(
        p, result, capture_agent, tenant_visible=False,
        prompt_mode=PromptMode.NEUTRAL,
    )
    run_probe(
        p, result, capture_agent, tenant_visible=False,
        prompt_mode=PromptMode.CAUTIOUS,
    )

    assert len(seen_systems) == 2
    assert "Do not combine" not in seen_systems[0]  # NEUTRAL
    assert "Do not combine" in seen_systems[1]      # CAUTIOUS
    print("[OK] run_probe threads prompt_mode to the agent")


# ----- SummaryMergeMode -----


def test_collapsed_summary_has_no_per_client_blocks() -> None:
    corpus = _make_corpus()
    traps = [p for p in corpus.probes if p.probe_type == ProbeType.CROSS_BOUNDARY]
    p = traps[0]
    text = render_summary_text(
        p, tenant_visible=False, mode=SummaryMergeMode.COLLAPSED
    )
    # COLLAPSED: a single flat bullet list, no per-tenant headers.
    assert "Summary of recent activity for this client" not in text
    # Should have a bullet for each fragment (5 for a CROSS_BOUNDARY probe).
    n_bullets = sum(1 for line in text.splitlines() if line.startswith("- "))
    assert n_bullets == len(p.fragments), (
        f"COLLAPSED summary has {n_bullets} bullets, expected {len(p.fragments)}"
    )
    print(
        f"[OK] COLLAPSED summary is a flat {n_bullets}-bullet list with no "
        "per-client framing"
    )


def test_per_client_blocks_summary_has_tenant_grouping() -> None:
    corpus = _make_corpus()
    traps = [p for p in corpus.probes if p.probe_type == ProbeType.CROSS_BOUNDARY]
    p = traps[0]
    text = render_summary_text(
        p, tenant_visible=False, mode=SummaryMergeMode.PER_CLIENT_BLOCKS
    )
    # PER_CLIENT_BLOCKS: one "Summary of recent activity..." header per
    # tenant; a trap spans at least two tenants.
    n_headers = text.count("Summary of recent activity for this client")
    assert n_headers >= 2, (
        f"PER_CLIENT_BLOCKS for a CROSS_BOUNDARY trap should have ≥2 "
        f"per-tenant headers; got {n_headers}"
    )
    print(
        f"[OK] PER_CLIENT_BLOCKS summary has {n_headers} per-tenant headers"
    )


def test_run_probe_threads_summary_merge_mode() -> None:
    corpus = _make_corpus()
    traps = [p for p in corpus.probes if p.probe_type == ProbeType.CROSS_BOUNDARY]
    p = traps[0]
    result = retrieve(Rung.SUMMARY_MERGE, p, corpus)

    seen_users: list[str] = []

    def capture_agent(system, user):
        seen_users.append(user)
        return json.dumps({"abstain": True, "reason": "stub"})

    run_probe(
        p, result, capture_agent, tenant_visible=False,
        summary_merge_mode=SummaryMergeMode.COLLAPSED,
    )
    run_probe(
        p, result, capture_agent, tenant_visible=False,
        summary_merge_mode=SummaryMergeMode.PER_CLIENT_BLOCKS,
    )

    assert len(seen_users) == 2
    # Collapsed: no per-tenant headers in the user prompt.
    assert "Summary of recent activity for this client" not in seen_users[0]
    # Per-client-blocks: has per-tenant headers.
    assert "Summary of recent activity for this client" in seen_users[1]
    print("[OK] run_probe threads summary_merge_mode to prompt construction")


# ----- Defaults -----


def test_default_summary_mode_is_collapsed() -> None:
    """
    Without summary_merge_mode, build_prompt renders the COLLAPSED
    summary.
    """
    corpus = _make_corpus()
    traps = [p for p in corpus.probes if p.probe_type == ProbeType.CROSS_BOUNDARY]
    p = traps[0]
    result = retrieve(Rung.SUMMARY_MERGE, p, corpus)
    # Build prompt WITHOUT specifying summary_merge_mode -> uses default.
    prompt = build_prompt(p, result, tenant_visible=False)
    assert "Summary of recent activity for this client" not in prompt.user, (
        "default summary mode should be COLLAPSED (no per-client headers)"
    )
    print("[OK] default summary_merge_mode is COLLAPSED")


def test_default_prompt_mode_is_cautious() -> None:
    """
    The default prompt_mode of build_prompt is CAUTIOUS; run_paper.py and
    run_smoke.py pass the prompt mode explicitly.
    """
    corpus = _make_corpus()
    p = corpus.probes[0]
    result = retrieve(Rung.SHARED_VECTOR_INDEX, p, corpus)
    prompt = build_prompt(p, result, tenant_visible=False)
    assert prompt.prompt_mode == PromptMode.CAUTIOUS
    print("[OK] default prompt_mode is CAUTIOUS")


def main() -> int:
    tests = [
        test_neutral_prompt_has_no_composition_warnings,
        test_cautious_prompt_has_composition_warning,
        test_run_probe_threads_prompt_mode,
        test_collapsed_summary_has_no_per_client_blocks,
        test_per_client_blocks_summary_has_tenant_grouping,
        test_run_probe_threads_summary_merge_mode,
        test_default_summary_mode_is_collapsed,
        test_default_prompt_mode_is_cautious,
    ]
    failures = []
    for t in tests:
        try:
            t()
        except Exception as e:
            failures.append((t.__name__, e))
            print(f"[FAIL] {t.__name__}: {type(e).__name__}: {e}")
    print()
    if failures:
        print(f"=== {len(failures)} test(s) failed ===")
        return 1
    print(f"=== all {len(tests)} prompt-mode tests passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
