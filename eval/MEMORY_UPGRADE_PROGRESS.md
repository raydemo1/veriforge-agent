# Memory upgrade delivery

## Final delivery status — 2026-10-24

Production integration is complete. The service, native tools, interactive session lifecycle, idle worker, topic navigation and context references now use the SQLite memory system. The old Markdown store implementation has been removed; no legacy import or parallel production path remains. The stage notes below describe earlier checkpoints, not the current activation state.

Final validation: 774 passed and 3 deselected in the full suite; 91 relevant tests passed after the final recovery/session-lifecycle changes. Standalone consolidation and runtime checks passed 17 and 5 checks respectively. Real-provider acceptance and its limitations are recorded in `MEMORY_V3_ACCEPTANCE.md`.

The user completed legacy cleanup with `eval/scripts/cleanup_legacy_memory.ps1`. The saved manifest reports completion, and a fresh preview finds no remaining legacy directories. Original session journals and Codex memories remain present. The default memory-v3 directory is created on production initialization.

## Accepted design

- SQLite is the sole authority; Markdown topic pages are generated views.
- Six kinds: preference, project_fact, convention, decision, lesson, workflow.
- Project namespaces share repository identity across worktrees; user memory is a separate namespace.
- Automatic extraction and cross-session consolidation are independent durable phases.
- Default recall makes no auxiliary LLM requests. Semantic retrieval uses SiliconFlow BAAI/bge-m3 (1024 dimensions), alongside SQLite FTS5; query embeddings incur separate provider requests.
- Explicit saves are immediate when unambiguous; uncertain conflicts require review.
- Idle extraction defaults to six hours. Processing pauses when the Harness exits and resumes on startup.
- No legacy import. Delete all Harness legacy memory stores only after final acceptance; retain session journals and Codex memories.

## Stage 1: SQLite storage foundation

Implemented in `harness_code_agent/memory/database.py`, deliberately not activated in the existing runtime yet:

- Namespace-bound CRUD, stable memory IDs, optimistic revision checks and retained revision history.
- Exact fact identity and source deduplication; colliding values become review-required rather than two active facts.
- Resolution requires new confirmation/verification evidence.
- Transactions cover record changes, history, audit revision, suppression and export outbox.
- Markdown topic exports are retryable projections; external file edits are not imported.
- Forget removes records, revisions, proposals and empty topics and queues projection cleanup. Source/fact suppression hashes contain no bodies.

This foundation does not yet validate evidence against journals, perform semantic reconciliation, or replace the production Markdown store. Those integrations belong to the later stages, not a compatibility path retained in the final system.

Validation on 2026-10-23:

- Storage acceptance tests: 20 passed, including process termination inside an uncommitted transaction, concurrent connections, namespace isolation, correction rollback, export failure/retry, snapshot mutation and forget suppression.
- Full local suite: 777 passed, 3 integration tests deselected.
- Ruff passed for the new storage implementation and tests.
- No paid requests, production runtime switch, legacy deletion, commit or push performed in this stage.

## Stage 2: Durable incremental extraction

Implemented as a separate, not-yet-activated pipeline:

- Typed user-input and structured-verification journal sources, isolated from recalled content and ordinary model/tool messages.
- Six-hour idle coalescing, resumable byte/character checkpoints, lease fencing, cancellation and bounded failures.
- Evidence quotes checked against the original source; secret redaction preserves offsets across chunk boundaries.
- Candidate observations and consolidation jobs committed with the extraction checkpoint. Candidates are not automatically published facts.
- Isolated real-model check on 2026-10-23: two requests, seven independent conventions retained and transient progress excluded. This only measures candidate extraction.

## Stage 3: Current-state consolidation and hybrid retrieval

Implemented on 2026-10-24, still isolated from the production runtime:

- Independent durable consolidation jobs, source validation, bounded retries and concurrency/version fencing.
- Batched add/duplicate/update/review decisions. There is no model-driven physical delete.
- Original candidate payloads and fact revisions remain available; confirmed updates replace the current projection. Revision restoration creates another revision.
- An exact-identity incoming change holds the old fact out of definite-current results before consolidation. Failed processing leaves the hold in place. Semantic-only conflicts with different identities still require successful discovery and judgement; pre-judgement detection is not guaranteed.
- Unresolved conflicts return uncertainty without asserting either value; ordinary Markdown topic exports contain confirmed current values only.
- SQLite FTS5 and sqlite-vec independently produce candidates. Reciprocal-rank fusion merges them, with namespace/applicability checks, bounded result tokens and pagination.
- SiliconFlow BAAI/bge-m3 uses a separate credential, fixed 1024-dimensional dense vectors, no hidden HTTP retries, and request/usage hashes without prompt or key content.
- Fact changes invalidate old vectors transactionally. Durable index work retries separately; stale embedding responses cannot restore overwritten or forgotten facts. Missing semantic results are explicitly labelled, not presented as complete retrieval.
- The large default view is never injected in full. Session/tool wiring and topic navigation remain the next delivery stage.

Validation:

- Existing storage/extraction checks: 60 passed.
- Standalone robustness script: 14 checks passed, including 2,000 synthetic facts. Its fixture vectors validate index mechanics, not semantic quality.
- New consolidation scenarios live under eval/scripts, outside default pytest collection.
- Bounded real-provider protocol: at most 28 LLM requests and 48 embedding requests, zero automatic retries; six fixed synthetic cases. Completed: 28 model and 41 embedding requests in isolated synthetic stores. Scope normalization and lexical stop-word defects were fixed; the frozen strict compactness rubric still passes only 1/6 on final stored fixtures. See eval/MEMORY_CONSOLIDATION_REPORT.md. Do not treat this as final production acceptance.

Current-stage release gate remains open: extraction compactness/topic normalization changes still need complete semantic and compactness validation.

## Planned stages at the earlier checkpoint (now delivered)

4. Runtime/tool/management integration, topic navigation, evidence validity checks and immutable session-context snapshots.
5. Final end-to-end acceptance, runtime switch and authorized legacy cleanup. Remove the superseded production path at this stage.

The final gate above supersedes this earlier checkpoint. Physical legacy cleanup remains a user-run operation.
