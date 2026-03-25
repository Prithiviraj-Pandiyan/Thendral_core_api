import hmac
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from settings import settings


admin_api_key_header = APIKeyHeader(
    name="X-Thendral-Admin-Key",
    auto_error=False,
    description="Admin key required for protected dataset/training endpoints",
)


def require_admin_key(api_key: str | None = Security(admin_api_key_header)) -> None:
    valid_keys = settings.get_admin_keys()
    if not valid_keys:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Admin API key not configured on server",
        )

    if not api_key:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    if not any(hmac.compare_digest(api_key, key) for key in valid_keys):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
