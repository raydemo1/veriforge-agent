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
from dataclasses import dataclass

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


@dataclass(frozen=True)
class SearchHit:
    key: str
    coverage: float


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


def search_bm25(documents: list[SearchDocument], query: str, *, limit: int = 5) -> list[SearchHit]:
    terms = sorted(set(tokenize(query)))
    if not documents or not terms or limit <= 0:
        return []

    frequencies = [Counter(tokenize(document.text)) for document in documents]
    lengths = [counts.total() for counts in frequencies]
    average_length = sum(lengths) / len(documents) or 1.0
    document_frequency = Counter(term for counts in frequencies for term in counts)
    weights = {}
    for term in terms:
        frequency = document_frequency[term]
        weights[term] = math.log(1 + (len(documents) - frequency + 0.5) / (frequency + 0.5))
    total_weight = sum(weights.values())

    ranked = []
    for index, counts in enumerate(frequencies):
        score = 0.0
        matched_weight = 0.0
        length_penalty = BM25_K1 * (1 - BM25_B + BM25_B * lengths[index] / average_length)
        for term, weight in weights.items():
            count = counts[term]
            if count:
                score += weight * count * (BM25_K1 + 1) / (count + length_penalty)
                matched_weight += weight
        if score > 0:
            ranked.append((score, SearchHit(documents[index].key, matched_weight / total_weight)))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [hit for _, hit in ranked[:limit]]
