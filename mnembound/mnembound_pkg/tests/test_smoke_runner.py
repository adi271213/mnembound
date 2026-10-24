"""
Smoke-runner tests.

Verifies, by running run_smoke.py with the deterministic stub:
- The runner's default prompt mode is NEUTRAL, the prompt of the paper's
  headline conditions, and its default summary mode is COLLAPSED.
- The runner reports cross-boundary abstention separately for the unsafe
  rungs (shared_vector_index, summary_merge) and the safe rungs (isolated,
  boundary_preserving_topk).
- PromptBundle carries the two prompts, the rung, the view and the prompt
  mode.
"""

from __future__ import annotations

import contextlib
import functools
import io
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import run_smoke
from mnembound import agent_io
from mnembound.retrieval import Rung


@functools.lru_cache(maxsize=None)
def _stub_run() -> tuple[int, str, dict]:
    """Exit code, printed output and smoke_config.json of a stub run with the default options."""
    with tempfile.TemporaryDirectory() as tmp:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = run_smoke.main(["--stub", "--out-dir", tmp])
        config = json.loads((Path(tmp) / "smoke_config.json").read_text())
    return code, out.getvalue(), config


def test_runner_default_prompt_mode_is_neutral() -> None:
    """Without --prompt-mode, the runner uses 'neutral', the prompt of the paper's headline conditions."""
    code, _, config = _stub_run()
    assert code == 0, "the stub run should pass every check of the expected rung pattern"
    assert config["prompt_mode"] == "neutral"
    print("[OK] runner default --prompt-mode is 'neutral'")


def test_runner_default_summary_mode_is_collapsed() -> None:
    _, _, config = _stub_run()
    assert config["summary_merge_mode"] == "collapsed"
    print("[OK] runner default --summary-mode is 'collapsed'")


def test_run_smoke_partitions_abstention_by_rung_safety() -> None:
    """
    The runner reports cross-boundary abstention on shared_vector_index and
    summary_merge separately from isolated and boundary_preserving_topk. The
    stub recalls a trap when it sees all five fragments and abstains when a
    rung strips some, so the 50 traps on two rungs give 0 of 100 abstentions
    on the unsafe rungs and 100 of 100 on the safe ones.
    """
    _, output, _ = _stub_run()
    assert "Cross-boundary abstention (unsafe rungs only, shared+summary): 0/100 (0.0%)" in output
    assert "Cross-boundary abstention (safe rungs, isolated+boundary-preserving): 100/100 (100.0%)" in output
    print("[OK] runner reports abstention separately for the unsafe and the safe rungs")


def test_prompt_bundle_fields() -> None:
    """PromptBundle carries the two prompts, the rung, the view and the prompt mode."""
    pb = agent_io.PromptBundle(system="s", user="u", rung=Rung.ISOLATED, tenant_visible=True)
    assert (pb.system, pb.user, pb.rung, pb.tenant_visible) == ("s", "u", Rung.ISOLATED, True)
    assert pb.prompt_mode == agent_io.PromptMode.CAUTIOUS
    print("[OK] PromptBundle carries its fields")


def main() -> int:
    tests = [
        test_runner_default_prompt_mode_is_neutral,
        test_runner_default_summary_mode_is_collapsed,
        test_run_smoke_partitions_abstention_by_rung_safety,
        test_prompt_bundle_fields,
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
    print(f"=== all {len(tests)} smoke-runner tests passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
