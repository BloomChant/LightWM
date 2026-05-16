"""
Centralized runtime configuration for LightWM.

Import `get_settings` from `config` instead of calling `os.getenv` directly.

Supported environment variables:
  DASH_API_KEY      — API key for DashScope-compatible endpoints (Alibaba Cloud)
  DASH_BASE_URL     — Optional base URL override (defaults to DashScope)
  OPENAI_API_KEY    — Fallback API key for standard OpenAI-compatible endpoints
  OPENAI_BASE_URL   — Fallback base URL for OpenAI-compatible endpoints
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class ApiSettings:
    dash_api_key: str
    dash_base_url: str


@dataclass(frozen=True)
class Settings:
    api: ApiSettings


@lru_cache()
def get_settings() -> Settings:
    dash_api_key = os.environ.get("DASH_API_KEY", "")
    dash_base_url = os.environ.get(
        "DASH_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    return Settings(
        api=ApiSettings(
            dash_api_key=dash_api_key,
            dash_base_url=dash_base_url,
        ),
    )
