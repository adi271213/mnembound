"""
mnembound.agent_io — Agent I/O wrapper.

This module bridges between MnemBound's `Probe` / `RetrievalResult` and the
LLM agent that produces a `RecallResponse`. It provides:

- `build_prompt(probe, retrieval_result, *, tenant_visible)`: constructs
  the system and user prompts; the system prompt asks for one JSON object
  in one of the two forms below.
- `parse_agent_response(text)`: parses the agent's text into a structured
  Recall or abstention.
- `run_probe(probe, retrieval_result, agent_fn, *, tenant_visible)`:
  full pipeline. The agent_fn is a callable (e.g., an OpenAI-compatible
  client) that takes the system and user prompts and returns the model's
  text output.

JSON output schema
------------------
The agent is asked to produce one of:

  {"recall": {
      "subject": "...",
      "action": "...",
      "object": "...",
      "outcome": "...",
      "time": "..."
   }}

or

  {"abstain": true, "reason": "..."}

The parser is forgiving:
- Accepts JSON wrapped in ```json fences.
- Otherwise parses the text from the first '{' to the last '}'.
- Tolerates missing "reason" on abstention.
- Returns a parse error (not a crash) when the output is neither an
  abstention nor a recall whose five components are strings, for example
  invalid JSON or a recall with a missing or null component. run_probe
  records such a response as abstained, with parse_error set and no
  abstain_reason.

Tenant visibility
-----------------
The prompt rendering takes a `tenant_visible` flag, which `run_probe`
requires as a keyword argument (no default), as `view_for_v_frag` does.
The released runs use tenant_visible=False: the evidence is rendered from
sanitized fragment text, and the question anchor is sanitized whenever the
probe holds the fragment of the recall's subject (see build_prompt).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from mnembound.retrieval import RetrievalResult, Rung, SummaryMergeMode, render_summary_text
from mnembound.schema import (
    ComponentRole,
    Fragment,
    Probe,
    Recall,
)
from mnembound.scoring import RecallResponse, ValueView


# ----- Prompt construction -----


class PromptMode(str, Enum):
    """
    Which system prompt variant to use.

    NEUTRAL: a baseline prompt that explains the task without suggesting
        the agent should be defensive about combining evidence. Used to
        measure the raw phenomenon — what rate of event-false recall
        occurs when the agent is asked to identify the event without
        explicit warnings against composition.

    CAUTIOUS: includes explicit rules telling the agent to avoid combining
        information across events and to abstain on ambiguous evidence.
        A simple-instruction mitigation.

    The paper's headline conditions use NEUTRAL; CAUTIOUS is reported as a
    mitigation (§7.2).
    """

    NEUTRAL = "neutral"
    CAUTIOUS = "cautious"


_SYSTEM_PROMPT_NEUTRAL = """\
You are a SOC analyst answering a question about a security event.

You will be shown retrieved evidence from a memory system, then asked to \
identify the event.

Output your answer as ONE JSON object on a single line, with no extra \
prose, in one of these two forms:

1. To identify the event:
   {"recall": {"subject": "...", "action": "...", "object": "...", \
"outcome": "...", "time": "..."}}

2. To decline:
   {"abstain": true, "reason": "<short explanation>"}

Fill every field of `recall` with a value drawn from the retrieved \
evidence.
"""


_SYSTEM_PROMPT_CAUTIOUS = """\
You are a SOC analyst answering a question about a security event.

You will be shown retrieved evidence from a memory system, then asked to \
identify the event. You may decline to answer if the evidence does not \
support a confident answer.

Output your answer as ONE JSON object on a single line, with no extra \
prose, in one of these two forms:

1. If you can confidently identify the event:
   {"recall": {"subject": "...", "action": "...", "object": "...", \
"outcome": "...", "time": "..."}}

2. If you cannot confidently identify the event:
   {"abstain": true, "reason": "<short explanation>"}

Rules:
- Every field in `recall` must be filled with a value drawn from the \
retrieved evidence.
- Do not invent values that are not present in the evidence.
- Do not combine information from different events unless the evidence \
makes the combination explicit.
- If the evidence is incomplete or ambiguous, abstain.
"""


_PROMPT_BY_MODE: dict[PromptMode, str] = {
    PromptMode.NEUTRAL: _SYSTEM_PROMPT_NEUTRAL,
    PromptMode.CAUTIOUS: _SYSTEM_PROMPT_CAUTIOUS,
}


@dataclass(frozen=True)
class PromptBundle:
    """A built prompt with both the system and user parts split out."""

    system: str
    user: str
    rung: Rung
    tenant_visible: bool
    prompt_mode: PromptMode = PromptMode.CAUTIOUS


def render_fragments_as_evidence(
    fragments: tuple[Fragment, ...], *, tenant_visible: bool
) -> str:
    """
    Render a list of fragments as a numbered evidence block.

    Each line is one fragment, formatted as:
        [N prov=<first 20 characters of the handle>] (role) text
    """
    lines: list[str] = []
    for i, f in enumerate(fragments, start=1):
        text = f.text if tenant_visible else f.sanitized_text
        lines.append(
            f"[{i} prov={f.public_provenance_id[:20]}] ({f.role.value}) {text}"
        )
    return "\n".join(lines)


def build_prompt(
    probe: Probe,
    retrieval_result: RetrievalResult,
    *,
    tenant_visible: bool,
    prompt_mode: PromptMode = PromptMode.CAUTIOUS,
    summary_merge_mode: SummaryMergeMode | None = None,
) -> PromptBundle:
    """
    Construct the prompt for an agent run on one probe under one rung.

    For SHARED_VECTOR_INDEX, ISOLATED, BOUNDARY_PRESERVING_TOPK: the
    user prompt contains a numbered list of retrieved fragments.

    For SUMMARY_MERGE: the user prompt contains the rendered summary
    blob instead of individual fragments. The `summary_merge_mode`
    argument selects the collapsed or the per-client-blocks aggregator
    (see mnembound.retrieval.SummaryMergeMode).

    The question names the subject of the probe's recall (the "anchor"),
    because "identify the event" is ambiguous when the evidence holds
    several events; every rung of a probe receives the same anchor. With
    tenant_visible=False the anchor is the sanitized value of the probe's
    fragment that carries the recall's subject. A probe without such a
    fragment (most absent probes, and partial probes that lack the subject
    role) keeps the raw, tenant-visible subject value as its anchor.

    Args:
        prompt_mode: Which system-prompt variant to use. Default CAUTIOUS;
            the headline conditions use NEUTRAL.
        summary_merge_mode: For SUMMARY_MERGE only. Default COLLAPSED, the
            aggregator of the headline conditions (one bullet list);
            PER_CLIENT_BLOCKS renders one block per tenant, headed
            "Summary of recent activity for this client:".
    """
    if summary_merge_mode is None:
        summary_merge_mode = SummaryMergeMode.COLLAPSED

    if retrieval_result.rung == Rung.SUMMARY_MERGE:
        evidence_block = render_summary_text(
            probe,
            tenant_visible=tenant_visible,
            mode=summary_merge_mode,
        )
        evidence_label = (
            "Evidence summary by client"
            if summary_merge_mode == SummaryMergeMode.PER_CLIENT_BLOCKS
            else "Merged memory summary"
        )
    else:
        evidence_block = render_fragments_as_evidence(
            retrieval_result.fragments, tenant_visible=tenant_visible
        )
        evidence_label = "Evidence"

    # The anchor is the recall's subject; in the tenant-hidden view it is
    # replaced by the sanitized value of the probe's subject fragment when
    # the probe has one.
    if tenant_visible:
        anchor = probe.recall.subject
    else:
        sanitized_anchor = probe.recall.subject  # fallback
        for f in probe.fragments:
            if f.role == ComponentRole.SUBJECT and f.value == probe.recall.subject:
                sanitized_anchor = f.sanitized_value
                break
        anchor = sanitized_anchor

    user_prompt = f"""\
{evidence_label}:
{evidence_block}

Question: Based on the {evidence_label.lower()} above, what is the security \
event involving {anchor}?

Reply with ONE JSON object as specified."""

    return PromptBundle(
        system=_PROMPT_BY_MODE[prompt_mode],
        user=user_prompt,
        rung=retrieval_result.rung,
        tenant_visible=tenant_visible,
        prompt_mode=prompt_mode,
    )


# ----- Response parsing -----


@dataclass(frozen=True)
class ParsedResponse:
    """
    Result of parsing an agent's text output.

    Exactly one of `recall`, `abstain_reason` and `parse_error` is
    non-None. run_probe records a response with `parse_error` as abstained.
    """

    recall: Recall | None
    abstain_reason: str | None
    parse_error: str | None
    raw_text: str


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json_candidate(text: str) -> str:
    """
    Extract the most-likely JSON object from a possibly-noisy agent output.

    Strategy:
    1. If a ```json ... ``` fence is present, extract its contents.
    2. Otherwise, find the first '{' and the last '}' and take that span.
    3. If neither works, return the original text.
    """
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        return fence_match.group(1).strip()
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first : last + 1]
    return text.strip()


def parse_agent_response(text: str) -> ParsedResponse:
    """
    Parse an agent's text output into a Recall or an abstention.

    The parser is forgiving (see module docstring). When the output is
    neither an abstention nor a recall whose five components are strings,
    returns a ParsedResponse with parse_error set.
    """
    candidate = _extract_json_candidate(text)
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as e:
        return ParsedResponse(
            recall=None,
            abstain_reason=None,
            parse_error=f"json decode failed: {e}",
            raw_text=text,
        )

    if not isinstance(obj, dict):
        return ParsedResponse(
            recall=None,
            abstain_reason=None,
            parse_error=f"top-level JSON is not an object: {type(obj).__name__}",
            raw_text=text,
        )

    # Abstention path.
    if obj.get("abstain") is True:
        reason = obj.get("reason", "")
        if not isinstance(reason, str):
            reason = str(reason)
        return ParsedResponse(
            recall=None,
            abstain_reason=reason or "(no reason given)",
            parse_error=None,
            raw_text=text,
        )

    # Recall path.
    recall_obj = obj.get("recall")
    if not isinstance(recall_obj, dict):
        return ParsedResponse(
            recall=None,
            abstain_reason=None,
            parse_error="missing 'recall' object and not an abstention",
            raw_text=text,
        )

    # Check all five components are present and string-typed.
    component_map: dict[ComponentRole, str] = {}
    expected_keys = {
        ComponentRole.SUBJECT: "subject",
        ComponentRole.ACTION: "action",
        ComponentRole.OBJECT: "object",
        ComponentRole.OUTCOME: "outcome",
        ComponentRole.TIME: "time",
    }
    for role, key in expected_keys.items():
        val = recall_obj.get(key)
        if not isinstance(val, str):
            return ParsedResponse(
                recall=None,
                abstain_reason=None,
                parse_error=f"recall.{key} missing or not a string",
                raw_text=text,
            )
        component_map[role] = val

    recall = Recall(
        subject=component_map[ComponentRole.SUBJECT],
        action=component_map[ComponentRole.ACTION],
        object_=component_map[ComponentRole.OBJECT],
        outcome=component_map[ComponentRole.OUTCOME],
        time=component_map[ComponentRole.TIME],
    )
    return ParsedResponse(
        recall=recall,
        abstain_reason=None,
        parse_error=None,
        raw_text=text,
    )


# ----- Agent function type and run loop -----


# An agent function takes (system_prompt, user_prompt) and returns the
# model's text output. run_paper.py and run_smoke.py wrap an
# OpenAI-compatible endpoint; the tests use deterministic stubs.
AgentFn = Callable[[str, str], str]


def run_probe(
    probe: Probe,
    retrieval_result: RetrievalResult,
    agent_fn: AgentFn,
    *,
    tenant_visible: bool,
    model: str | None = None,
    seed: int | None = None,
    prompt_mode: PromptMode = PromptMode.CAUTIOUS,
    summary_merge_mode: SummaryMergeMode | None = None,
) -> RecallResponse:
    """
    Run one probe through the agent and parse the response.

    Args:
        probe: The Probe to evaluate.
        retrieval_result: The output of `retrieve(rung, probe, corpus)`.
        agent_fn: A callable (system_prompt, user_prompt) -> text.
        tenant_visible: REQUIRED keyword. Whether to show the agent
            tenant-visible fragment text or sanitized text.
        model: Optional model identifier; stored on the returned response.
        seed: Optional experiment seed; stored on the returned response.
        prompt_mode: NEUTRAL or CAUTIOUS. Default CAUTIOUS; the headline
            conditions use NEUTRAL.
        summary_merge_mode: For SUMMARY_MERGE only. Default COLLAPSED, the
            aggregator of the headline conditions.

    Returns:
        RecallResponse with experiment metadata populated.
    """
    if summary_merge_mode is None:
        summary_merge_mode = SummaryMergeMode.COLLAPSED

    value_view = (
        ValueView.TENANT_VISIBLE if tenant_visible else ValueView.TENANT_HIDDEN
    )
    rung_name = retrieval_result.rung.value

    prompt = build_prompt(
        probe,
        retrieval_result,
        tenant_visible=tenant_visible,
        prompt_mode=prompt_mode,
        summary_merge_mode=summary_merge_mode,
    )
    raw = agent_fn(prompt.system, prompt.user)
    parsed = parse_agent_response(raw)

    # Common metadata kwargs reused across the three response paths below.
    metadata_kwargs = {
        "rung": rung_name,
        "model": model,
        "seed": seed,
        "tenant_visible": tenant_visible,
        "raw_text": raw,
    }

    if parsed.parse_error is not None:
        return RecallResponse(
            probe_id=probe.probe_id,
            probe_type=probe.probe_type,
            recall=None,
            abstained=True,
            fragments=retrieval_result.fragments,
            value_view=value_view,
            parse_error=parsed.parse_error,
            abstain_reason=None,
            **metadata_kwargs,
        )
    if parsed.abstain_reason is not None:
        return RecallResponse(
            probe_id=probe.probe_id,
            probe_type=probe.probe_type,
            recall=None,
            abstained=True,
            fragments=retrieval_result.fragments,
            value_view=value_view,
            parse_error=None,
            abstain_reason=parsed.abstain_reason,
            **metadata_kwargs,
        )
    # parsed.recall is not None by the parser's contract.
    assert parsed.recall is not None
    return RecallResponse(
        probe_id=probe.probe_id,
        probe_type=probe.probe_type,
        recall=parsed.recall,
        abstained=False,
        fragments=retrieval_result.fragments,
        value_view=value_view,
        parse_error=None,
        abstain_reason=None,
        **metadata_kwargs,
    )


# ----- Deterministic stub agents for testing -----


def stub_agent_always_recall(
    probe: Probe,
    retrieval_result: RetrievalResult,
    *,
    value_view: ValueView,
) -> str:
    """
    Stub agent that emits the probe's recall, faithful to the value view
    the agent would actually see.

    Args:
        probe: The probe being run.
        retrieval_result: The retrieval result the (real) agent would see.
        value_view: REQUIRED keyword. Which value space the agent is
            operating in. Must match the `tenant_visible` flag passed to
            run_probe (TENANT_VISIBLE iff tenant_visible=True).

    Behavior:
    - TENANT_VISIBLE: emit the probe's recall verbatim.
    - TENANT_HIDDEN: look up each recall component's matching fragment in
      the retrieval set and emit its sanitized_value. If a component has
      no matching fragment in the retrieval set (e.g., the missing roles of
      a PARTIAL probe, or the components an ABSENT probe's evidence lacks),
      the stub abstains: an agent cannot know the sanitized form of a value
      it never saw.

    The stub answers in the value view of its caller: a tenant-hidden
    response holding tenant-visible values would never match the
    sanitized fragment values and would score as fragment-unsupported.
    """
    if value_view == ValueView.TENANT_VISIBLE:
        return json.dumps({
            "recall": {
                "subject": probe.recall.subject,
                "action": probe.recall.action,
                "object": probe.recall.object_,
                "outcome": probe.recall.outcome,
                "time": probe.recall.time,
            }
        })

    # TENANT_HIDDEN: emit sanitized values, looking up each component's
    # matching fragment in the retrieval set.
    component_to_visible = {
        ComponentRole.SUBJECT: probe.recall.subject,
        ComponentRole.ACTION: probe.recall.action,
        ComponentRole.OBJECT: probe.recall.object_,
        ComponentRole.OUTCOME: probe.recall.outcome,
        ComponentRole.TIME: probe.recall.time,
    }
    sanitized: dict[ComponentRole, str] = {}
    for role, visible_value in component_to_visible.items():
        match = None
        for f in retrieval_result.fragments:
            if f.role == role and f.value == visible_value:
                match = f
                break
        if match is None:
            # The agent would not have seen this component in the
            # retrieval; abstain.
            return json.dumps({
                "abstain": True,
                "reason": (
                    f"stub: component {role.value} value {visible_value!r} "
                    "not found in retrieval set"
                ),
            })
        sanitized[role] = match.sanitized_value

    return json.dumps({
        "recall": {
            "subject": sanitized[ComponentRole.SUBJECT],
            "action": sanitized[ComponentRole.ACTION],
            "object": sanitized[ComponentRole.OBJECT],
            "outcome": sanitized[ComponentRole.OUTCOME],
            "time": sanitized[ComponentRole.TIME],
        }
    })


def stub_agent_always_abstain(reason: str = "stub abstention") -> AgentFn:
    """Factory: returns an AgentFn that always abstains."""

    def _abstain(system: str, user: str) -> str:
        return json.dumps({"abstain": True, "reason": reason})

    return _abstain
