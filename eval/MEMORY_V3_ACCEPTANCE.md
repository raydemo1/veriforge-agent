# Memory v3 acceptance

Date: 2026-10-24, Hong Kong.

SQLite is now the production authority. Session extraction and current-state consolidation are separate durable phases. Retrieval combines FTS5 and SiliconFlow BAAI/bge-m3 dense vectors; Markdown files are generated topic views. Native memory tools, idle scheduling, session recovery, startup preferences and bounded automatic recall use the same service. Old production store code was removed without importing legacy data.

## Local verification

- Full suite: 774 passed, 3 deselected.
- Relevant tests after the final recovery/session-lifecycle changes: 91 passed.
- Standalone consolidation checks: 17 passed, including transactional failure, stale decisions, conflict holds, explicit confirmation, re-remembering and older-update protection.
- Standalone runtime checks: 5 passed, including native tools, append-only recall, startup preferences and separate background phases.

## Real-provider verification

The six-case run in `results/memory-consolidation-20261023-172821/` passed confirmed update, partial update, successive updates and invalidation without replacement. Paraphrase deduplication and scoped coexistence failed the strict rubric. After fixing source-context preservation and duplicate scope handling, both cases passed a fresh targeted run in `results/memory-consolidation-20261023-173628/`. All six cases therefore have passing evidence across the full run and targeted retest; this is not a fresh single-run 6/6 result. Original failure artifacts remain intact.

Two actual InteractiveSession instances verified native writing and cross-session startup/semantic recall. A subsequent background drain verified separate extraction, consolidation and indexing without duplicating the saved preference. Evidence is in `results/memory-runtime-20261023-173947/`, including `summary.json` and `worker-summary.json`. These native runtime checks used 5 model requests and 3 embedding requests in total.

This completion round used 39 model and 46 embedding requests. Including earlier upgrade checks, the upgrade used 69 model and 87 embedding requests: 156 requests against the fixed 160-request aggregate cap. No further paid requests were made for final documentation or cleanup.

## Limits and cleanup

Retrieval and semantic reconciliation are approximate. Exact-identity incoming changes hold existing facts out of definite-current results before judgement; conflicts with different identities still require successful retrieval and judgement. Provider failure leaves durable work and explicit degraded status rather than claiming complete semantic recall. The synthetic fixtures do not establish broad multilingual quality or universal conflict detection.

The new implementation is wired into production. The user ran `scripts/cleanup_legacy_memory.ps1` and deleted all five legacy directories; the completion manifest and a fresh empty preview confirm cleanup. Session journals and Codex memories remain present. The default memory-v3 directory will be created on production initialization; real-provider acceptance used isolated stores.
