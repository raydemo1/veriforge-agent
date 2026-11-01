"""BM25 retrieval for deferred tool discovery.

The index text is the tool name, its description and every parameter name and
description.  Coverage is the share of the query's IDF mass that a document
matches; it is comparable across queries, unlike a raw BM25 score, so the
discovery layer can use it as a confidence tier.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

BM25_K1 = 1.5
BM25_B = 0.75

_STOPWORDS = frozenset(
    [
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
        "how", "i", "in", "into", "is", "it", "its", "me", "my", "of", "on", "or", "the",
        "this", "that", "to", "was", "were", "what", "when", "where", "which", "with",
        "worth", "will", "much", "many", "one", "every", "give", "show",
    ]
)

_ENGLISH_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class SearchDocument:
    key: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchHit:
    key: str
    score: float
    coverage: float
    document: str
    metadata: dict[str, Any] = field(default_factory=dict)


def tokenize(text: str) -> list[str]:
    """English words with stopwords removed.

    Queries are English keywords, so the index is tokenized the same way.  CJK
    descriptions are never matched by an English query; supporting Chinese
    tokens would only dilute the length normalization.
    """
    lowered = str(text or "").lower()
    return [
        token
        for token in _ENGLISH_TOKEN_RE.findall(lowered)
        if len(token) > 1 and token not in _STOPWORDS
    ]


class BM25Index:
    """Dependency-free BM25 over a fixed document set."""

    def __init__(
        self,
        documents: list[SearchDocument],
        *,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(document.text) for document in documents]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.avg_len = sum(self.doc_len) / max(1, len(self.doc_len)) or 1.0
        self.term_freq = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_freq: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            self.doc_freq.update(set(tokens))

    def _idf(self, term: str) -> float:
        doc_freq = self.doc_freq.get(term, 0)
        return math.log(1 + (len(self.documents) - doc_freq + 0.5) / (doc_freq + 0.5))

    def search(self, query: str, limit: int = 5) -> list[SearchHit]:
        terms = sorted(set(tokenize(query)))
        if not terms or limit <= 0:
            return []
        total_idf = sum(self._idf(term) for term in terms) or 1.0
        scored: list[tuple[float, float, int]] = []
        for index, freqs in enumerate(self.term_freq):
            length = self.doc_len[index]
            score = 0.0
            matched_idf = 0.0
            for term in terms:
                freq = freqs.get(term, 0)
                if not freq:
                    continue
                denominator = freq + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                score += self._idf(term) * freq * (self.k1 + 1) / denominator
                matched_idf += self._idf(term)
            # A zero score means the document shares no term with the query; it
            # is not a weak match and must not fill the result list.
            if score > 0.0:
                scored.append((score, matched_idf / total_idf, index))
        scored.sort(key=lambda item: (-item[0], item[2]))
        hits: list[SearchHit] = []
        for score, coverage, index in scored[:limit]:
            document = self.documents[index]
            hits.append(
                SearchHit(
                    key=document.key,
                    score=score,
                    coverage=coverage,
                    document=document.text,
                    metadata=dict(document.metadata),
                )
            )
        return hits


def search_bm25(documents: list[SearchDocument], query: str, *, limit: int = 5) -> list[SearchHit]:
    if not documents or limit <= 0:
        return []
    return BM25Index(documents).search(query, limit=limit)
