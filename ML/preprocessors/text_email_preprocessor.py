import html
import re
import unicodedata

from ML.preprocessors.base import BasePreprocessor


URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTISPACE_RE = re.compile(r"\s+")
NON_TEXT_RE = re.compile(r"[^a-z0-9\s_]")
OTP_RE = re.compile(r"\b\d{4,8}\b")

BOILERPLATE_LINE_RE = re.compile(
    r"(unsubscribe|manage preferences|privacy policy|terms|view in browser|help centre|you are receiving this email)",
    re.IGNORECASE,
)


class TextEmailPreprocessor(BasePreprocessor):
    """Email/text focused preprocessor for NLP classification tasks."""

    def _strip_boilerplate_lines(self, text: str) -> str:
        kept_lines = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if BOILERPLATE_LINE_RE.search(line):
                continue
            kept_lines.append(line)
        return "\n".join(kept_lines)

    def clean_text(self, text: str) -> str:
        if text is None:
            return ""

        text = str(text)
        text = html.unescape(text)
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = self._strip_boilerplate_lines(text)

        text = HTML_TAG_RE.sub(" ", text)
        text = text.lower()

        text = URL_RE.sub(" url_token ", text)
        text = EMAIL_RE.sub(" email_token ", text)
        text = OTP_RE.sub(" otp_token ", text)

        text = NON_TEXT_RE.sub(" ", text)
        text = MULTISPACE_RE.sub(" ", text).strip()
        return text
