from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    thendral_admin_api_key: str | None = None
    thendral_admin_api_keys: str | None = None
    hf_token: str | None = None
    

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Single source of truth: load config from .env only.
        return (dotenv_settings,)

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
