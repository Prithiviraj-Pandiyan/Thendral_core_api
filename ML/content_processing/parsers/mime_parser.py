from email import policy
from email.parser import BytesParser


def parse_eml_bytes(raw_bytes: bytes) -> dict:
    """
    Parse .eml bytes and extract key fields for downstream cleaning.
    """
    message = BytesParser(policy=policy.default).parsebytes(raw_bytes)
    subject = message.get("subject", "")
    sender = message.get("from", "")

    plain_parts: list[str] = []
    html_parts: list[str] = []

    if message.is_multipart():
        for part in message.walk():
            content_type = part.get_content_type()
            if part.get_content_disposition() == "attachment":
                continue
            if content_type == "text/plain":
                plain_parts.append(part.get_content() or "")
            elif content_type == "text/html":
                html_parts.append(part.get_content() or "")
    else:
        content_type = message.get_content_type()
        if content_type == "text/plain":
            plain_parts.append(message.get_content() or "")
        elif content_type == "text/html":
            html_parts.append(message.get_content() or "")

    return {
        "subject": subject,
        "from": sender,
        "plain_text": "\n".join(plain_parts).strip(),
        "html": "\n".join(html_parts).strip(),
    }
