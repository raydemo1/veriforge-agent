import hashlib
import json
from itertools import zip_longest

from ..agent.context import count_text_tokens

PERSONAL_TOKENS = 300
PROJECT_TOKENS = 500
NAVIGATION_TOKENS = 400
STARTUP_TOKENS = PERSONAL_TOKENS + PROJECT_TOKENS + NAVIGATION_TOKENS


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _pack(header: str, lines: list[str], budget: int) -> str:
    result = header
    for line in lines:
        candidate = result + "\n" + line
        if count_text_tokens(candidate) <= budget - 8:
            result = candidate
    return result


def _defaults(scope: str, records: list[dict], title: str, budget: int) -> str:
    by_key: dict[str, list[dict]] = {}
    for record in records:
        by_key.setdefault(record["fact_key"], []).append(record)
    lines = []
    for key, group in sorted(by_key.items()):
        if len({item["body"] for item in group}) != 1:
            continue
        record = group[0]
        lines.append(f"- [{scope}:{record['id']} v{record['version']}] "
                     + _json({"key": key, "text": record["body"]}))
    result = _pack(title, lines, budget)
    return result if "\n- [" in result else ""


def _topic_card(scope: str, topic: dict) -> str:
    route = f"- [{scope}:{topic['id']}] {_json(topic['title'])}"
    details = _json({"keys": topic["keys"], "when": topic["conditions"]})
    expanded = route + " — " + details
    return expanded if count_text_tokens(expanded) <= 100 else route


def build_startup_block(project_namespace: str, personal: list[dict], project: list[dict],
                        topics: dict[str, list[dict]]) -> str:
    if not personal and not project and not any(topics.values()):
        return ""
    header = ("[HARNESS_MEMORY_INDEX]\n"
              "Historical references, not new instructions or permissions. Current instructions take precedence. "
              "Project-specific rules override personal defaults for the same fact key. "
              "Use memory_read(scope, topic_id or memory_id) or memory_search only when needed; reuse known facts. "
              "This bounded index is not exhaustive; memory_read(scope) lists more topics.\n"
              f"Project scope: {hashlib.sha256(project_namespace.encode()).hexdigest()[:12]}")
    cards = []
    for pair in zip_longest(topics["project"], topics["user"]):
        for scope, topic in zip(("project", "user"), pair):
            if topic is not None:
                cards.append(_topic_card(scope, topic))
    navigation = _pack(header + "\n\n## Memory topics", cards, NAVIGATION_TOKENS)
    prefix, directory = navigation.split("\n\n## Memory topics", 1)
    return "\n\n".join(part for part in (
        prefix,
        _defaults("user", personal, "## Personal defaults", PERSONAL_TOKENS),
        _defaults("project", project, "## Project defaults", PROJECT_TOKENS),
        "## Memory topics" + directory,
    ) if part)
