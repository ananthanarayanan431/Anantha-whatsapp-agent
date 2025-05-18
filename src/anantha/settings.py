
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_file_encoding="utf-8")

    GROQ_API_KEY: str
    TOGETHER_API_KEY: str
    
    QDRANT_API_KEY: str | None
    QDRANT_URL: str | None
    QDRANT_PORT: str = "6333"
    QDRANT_HOST: str | None = None


    TEXT_MODEL_NAME: str = "llama-3.3-70b-versatile"
    ITT_MODEL_NAME: str = "llama-3.2-90b-vision-preview"


settings = Settings()