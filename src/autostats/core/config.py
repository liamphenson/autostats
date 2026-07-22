from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AUTOSTATS_", extra="ignore")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = "gpt-5"

    data_dir: Path = Path("./data")
    fred_api_key: str = ""
    census_api_key: str = ""

    @property
    def sessions_dir(self) -> Path:
        d = self.data_dir / "sessions"
        d.mkdir(parents=True, exist_ok=True)
        return d


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
