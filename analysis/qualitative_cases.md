# Case studies: fragment-faithful, event-false recalls (§7.3, Appendix E)

One cross-boundary trap per model under the headline condition on the shared-vector rung: the boundary-invalid recall that spans the most boundaries, ties broken by the lowest probe_id.

## Qwen3-30B-A3B-Instruct-2507-FP8 (seed 42, neutral / collapsed): crossboundary-000005

Rung: shared_vector_index. Value view: tenant_hidden. Boundaries spanned by the recall: b-B, b-D, b-E, b-H, b-I (5).

### Retrieved evidence

| # | role | boundary | sanitized value | sanitized text |
|---|---|---|---|---|
| 1 | subject | b-B | `user:u_6cb60e35` | `Egress event initiated by user:u_6cb60e35.` |
| 2 | action | b-I | `elevated_privileges_on` | `Privilege-elevation action observed at 2026-01-20T18:07:33+00:00.` |
| 3 | object | b-E | `host:h_81fa9869` | `Target resource: host:h_81fa9869.` |
| 4 | outcome | b-H | `completed` | `Egress outcome: completed.` |
| 5 | time | b-D | `2026-01-01T07:40:39+00:00` | `Event timestamped 2026-01-01T07:40:39+00:00.` |

### Recall

```json
{
  "recall": {
    "subject": "user:u_6cb60e35",
    "action": "Privilege-elevation action",
    "object": "host:h_81fa9869",
    "outcome": "Egress outcome: completed",
    "time": "2026-01-01T07:40:39+00:00"
  }
}
```

### Supplying fragments per component

| component | value | fragment (boundary) | match |
|---|---|---|---|
| subject | `user:u_6cb60e35` | #1 (b-B) | exact |
| action | `Privilege-elevation action` | #2 (b-I) | recall-in-fragment-text |
| object | `host:h_81fa9869` | #3 (b-E) | exact |
| outcome | `Egress outcome: completed` | #4 (b-H) | fragment-in-recall |
| time | `2026-01-01T07:40:39+00:00` | #5 (b-D) | exact |

Match kinds: exact 3, fragment-in-recall 1, recall-in-fragment-text 1. Components supplied by exactly one fragment: 5 of 5.

## Mistral-Small-3.2-24B-Instruct-2506-FP8 (seed 42, neutral / collapsed): crossboundary-000002

Rung: shared_vector_index. Value view: tenant_hidden. Boundaries spanned by the recall: b-A, b-B, b-D, b-G, b-H (5).

### Retrieved evidence

| # | role | boundary | sanitized value | sanitized text |
|---|---|---|---|---|
| 1 | subject | b-A | `user:u_f1c9dbb6` | `Egress event initiated by user:u_f1c9dbb6.` |
| 2 | action | b-H | `executed_on` | `Executable launched at 2026-03-22T14:52:58+00:00.` |
| 3 | object | b-D | `cve:CVE-2024-10018` | `Endpoint affected: cve:CVE-2024-10018.` |
| 4 | outcome | b-B | `executed` | `EDR response outcome: executed.` |
| 5 | time | b-G | `2026-03-24T14:05:10+00:00` | `Event timestamped 2026-03-24T14:05:10+00:00.` |

### Recall

```json
{
  "recall": {
    "subject": "user:u_f1c9dbb6",
    "action": "Executable launched",
    "object": "cve:CVE-2024-10018",
    "outcome": "executed",
    "time": "2026-03-24T14:05:10+00:00"
  }
}
```

### Supplying fragments per component

| component | value | fragment (boundary) | match |
|---|---|---|---|
| subject | `user:u_f1c9dbb6` | #1 (b-A) | exact |
| action | `Executable launched` | #2 (b-H) | recall-in-fragment-text |
| object | `cve:CVE-2024-10018` | #3 (b-D) | exact |
| outcome | `executed` | #4 (b-B) | exact |
| time | `2026-03-24T14:05:10+00:00` | #5 (b-G) | exact |

Match kinds: exact 4, recall-in-fragment-text 1. Components supplied by exactly one fragment: 5 of 5.

