import re


def dedupe_repeated_lines(text: str) -> str:
    """
    Remove exact repeated lines while preserving order.
    Useful for newsletter HTML blocks copied multiple times.
    """
    if not text:
        return ""

    seen: set[str] = set()
    result: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if line in seen:
            continue
        seen.add(line)
        result.append(line)
    return "\n".join(result)
