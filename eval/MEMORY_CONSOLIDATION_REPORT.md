# Memory consolidation and hybrid retrieval evaluation

Historical checkpoint: the runtime integration and later acceptance results are now recorded in `MEMORY_V3_ACCEPTANCE.md`. The original failures and request accounting below are retained as evidence.

Date: 2026-10-24 (Hong Kong). Isolated synthetic stores only; the production memory path remains unchanged.

## Scope and request accounting

The run used DeepSeek's configured fast Profile and SiliconFlow `BAAI/bge-m3`, with 1024-dimensional dense vectors stored by sqlite-vec. FTS5 independently supplies lexical candidates; reciprocal-rank fusion combines both pools. It does not use bge-m3 sparse or ColBERT outputs, or an extra reranker model.

Aggregate cap: 28 model requests plus 48 embedding requests, including failed requests. No transport retries, no automatic experiment expansion.

Actual aggregate: **28 model requests and 41 embedding requests**. Model usage: **40,069 tokens**. The six-case initial run used 22 model / 24 embedding requests. Repairs used the remaining six model requests; read-only retrieval checks stayed within the embedding cap.

Artifacts: `eval/results/memory-consolidation-20261023-162125/`.

- `protocol.json`, `summary.json`: original frozen cases and original results, preserved.
- `repair-protocol.json`, `repair-summary.json`: repair of the two failed consolidation cases, with original aggregate caps.
- `retrieval-repair.json`: cross-language retrieval after stop-word handling changed.
- `postfix-query-protocol.json`, `postfix-query-summary.json`: six read-only checks of existing fixture databases after fixes. These did not re-extract the other four cases.
- `model-requests.jsonl`, `embedding-requests.jsonl`: aggregate attempt/usage records. Fixtures and model outputs are synthetic; credentials are not included.

## Findings

| Case | Observed behavior | Remaining limitation |
|---|---|---|
| Confirmed engine update | Current engine changed to PostgreSQL; original SQLite revision remained in history | An extra migration-completed fact was stored, and the current body still mentioned the deprecated value |
| Paraphrased preference | Scope-normalization repair merged the two sources into one Chinese-language preference | Original retrieval repair was also needed; the final read-only check passed |
| Scoped coexistence | tools/ PostgreSQL and the default SQLite convention coexisted | An additional non-tools fact and three separate topics made the representation unnecessarily fragmented |
| Partial update | Build changed to BUILD_M33; TEST_M32 remained | The stored current body also narrated BUILD_M31, failing the frozen forbidden-token condition |
| Successive updates | After repair, PORT_M41 -> PORT_M42 -> PORT_M43 completed; current port is PORT_M43 | A separate PORT_M42-is-disabled fact remained, failing the one-record compactness condition |
| Invalidation without replacement | DEPLOY_M51 became explicitly disabled; a new entry was recorded as undetermined | Two facts instead of the frozen one-record expectation |

The original strict rubric passed **0/6**. On the final stored data, after the targeted repairs and read-only retrieval checks, it passed **1/6**. This rubric includes exact fact-count and forbidden-token checks: this is not an answer-accuracy score, and not every failure indicates a contradictory current value. The original failures are not relabelled as successes.

Two actual implementation defects were fixed and exercised with real requests:

1. The extractor described the same global applicability with different prose. Exact-string validation rejected an otherwise correct duplicate decision. Consolidation now returns canonical applicability; target scope must still match, and path-specific observations cannot overwrite a different scope.
2. English stop words such as `is` and `for` gave all archive-label distractors lexical hits. Two-channel fusion then outweighed the relevant semantic-only result. Standard English function words are now excluded from lexical terms. With 40 distractors, the Chinese-language preference ranked first for the English question, with observed cosine similarity approximately 0.509.

The initial port case also exposed a prompt/protocol problem: the model treated an explicitly confirmed symbolic value as an unverifiable placeholder. The updated prompt accepts source-provided identifiers without inventing external validation requirements; first-time uncertain facts can be held for review instead of causing a protocol error.

Extraction instructions now separate current values from change narration, keep completion-only context in episodes, use compact applicability and aggregate topic names. **The other four extraction cases were not rerun after these prompt changes. Their compactness and topic issues cannot be declared fixed.** No new paid requests were made after the aggregate model cap was reached.

## Offline validation

- Full existing suite before live-discovered fixes: 817 passed, 3 integration tests deselected.
- Relevant storage/extraction suite after fixes: 60 passed.
- Explicit robustness script: 14 checks, including transactional failure, bounded retries, revision restoration, concurrent corrections, unresolved conflict warnings, namespace isolation, evidence tampering, vector failure, forget during embedding, and 2,000-fact indexed retrieval/pagination.
- Ruff passed for the changed implementation and evaluation scripts.

Run explicit checks with:

```powershell
D:\miniconda\miniconda\python.exe eval/scripts/check_memory_consolidation.py
```

Fixture embeddings in the offline script only test mechanics. They are not evidence of semantic recall quality. New consolidation checks are outside default pytest collection.

## Release boundary

Do not activate the new production pipeline or delete legacy memories yet. The next review must settle the current-fact versus historical-event boundary, scope/topic normalization, and semantic correctness versus compactness criteria before an end-to-end runtime switch. Automatic pre-consolidation holds currently detect exact fact identities; an unrelated spelling of an identity still needs successful discovery and semantic judgement.

The current read API uses bounded candidate pools (200 per channel), a default 2,000-token item budget, namespace/applicability checks, and pagination. It does not inject the entire current view. Runtime tool wiring, topic expansion, file-evidence expiry, and session-context integration remain the next stage.

## Interface references

- SiliconFlow embedding API: https://docs.siliconflow.cn/cn/api-reference/embeddings/create-embeddings
- BGE-M3 model card: https://huggingface.co/BAAI/bge-m3
- sqlite-vec Python integration: https://alexgarcia.xyz/sqlite-vec/python.html
