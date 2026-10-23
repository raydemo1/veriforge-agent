import re

_STOP_WORDS = frozenset(["a", "an", "and", "are", "as", "at", "be", "been", "by", "do", "does", "for", "from", "how", "i", "in", "is", "it", "its", "my", "of", "on", "or", "our", "should", "that", "the", "their", "these", "this", "those", "to", "was", "we", "were", "what", "which", "who", "with", "you", "your"])


def search_terms(text: str) -> str:
    words = re.findall(r"[a-z0-9_./:-]+|[\u3400-\u9fff]+", text.casefold())
    tokens = []
    for word in words:
        if re.fullmatch(r"[\u3400-\u9fff]+", word):
            tokens.extend(word[i:i + 2] for i in range(max(1, len(word) - 1)))
        elif word not in _STOP_WORDS:
            tokens.append(word)
            tokens.extend(re.findall(r"[a-z0-9]+", word))
    return " ".join(token for token in dict.fromkeys(tokens) if token not in _STOP_WORDS)

