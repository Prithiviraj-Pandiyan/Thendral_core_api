from ML.content_processing.cleaners.html_cleaner import html_to_text
from ML.content_processing.transformers.deduper import dedupe_repeated_lines


def _looks_like_html(text: str) -> bool:
    markers = ("<html", "<body", "<table", "<tr", "<td", "<div", "<a ", "<img")
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def process_content(raw_text: str, source_type: str = "generic") -> dict:
    """
    Generic content processing pipeline.
    Returns normalized text + metadata for downstream ML preprocessing.
    """
    if raw_text is None:
        raw_text = ""
    text = str(raw_text)

    is_html = _looks_like_html(text)
    if is_html:
        text = html_to_text(text)

    text = dedupe_repeated_lines(text)

    return {
        "source_type": source_type,
        "detected_html": is_html,
        "normalized_text": text,
    }
