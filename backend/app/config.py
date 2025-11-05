from functools import lru_cache
from pydantic import BaseModel
import os


class Settings(BaseModel):
    approval_threshold: float = float(os.getenv("APPROVAL_THRESHOLD", "0.7"))
    environment: str = os.getenv("ENV", "development")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


