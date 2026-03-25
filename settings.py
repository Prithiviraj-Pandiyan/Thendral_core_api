from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    thendral_admin_api_key: str | None = None
    thendral_admin_api_keys: str | None = None
    hf_token: str | None = None
    

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def get_admin_keys(self) -> set[str]:
        keys: set[str] = set()
        if self.thendral_admin_api_key:
            keys.add(self.thendral_admin_api_key.strip())
        if self.thendral_admin_api_keys:
            for item in self.thendral_admin_api_keys.split(","):
                value = item.strip()
                if value:
                    keys.add(value)
        return keys


settings = Settings()
