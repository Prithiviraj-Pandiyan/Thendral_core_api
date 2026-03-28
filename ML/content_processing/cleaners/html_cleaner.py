from bs4 import BeautifulSoup


def html_to_text(html_content: str) -> str:
    """
    Convert HTML content into readable text.
    Keeps semantic text while removing layout/script/style noise.
    """
    if not html_content:
        return ""

    try:
        soup = BeautifulSoup(html_content, "lxml")
    except Exception:
        soup = BeautifulSoup(html_content, "html.parser")

    # Remove high-noise elements commonly found in marketing templates.
    for tag in soup(["script", "style", "noscript", "svg", "canvas", "iframe"]):
        tag.decompose()

    # Remove tracking images and most layout-only tables.
    for img in soup.find_all("img"):
        alt_text = (img.get("alt") or "").strip()
        if alt_text:
            img.replace_with(f" {alt_text} ")
        else:
            img.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)
