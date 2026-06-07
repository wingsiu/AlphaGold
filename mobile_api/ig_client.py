"""Shared IG session helper for mobile API (consistent .env + argument order)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if TYPE_CHECKING:
    from ig_scripts.ig_data_api import IGService


def create_ig_service() -> IGService:
    from dotenv import load_dotenv
    from ig_scripts.ig_data_api import API_CONFIG, IGService

    load_dotenv(PROJECT_ROOT / ".env")
    return IGService(
        api_key=API_CONFIG["api_key"],
        username=API_CONFIG["username"],
        password=API_CONFIG["password"],
        base_url=API_CONFIG["base_url"],
    )
