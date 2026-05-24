"""Shim — canonical v13 config lives in V13/config/v13_config.py."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parent.parent / "V13" / "config" / "v13_config.py"
_spec = importlib.util.spec_from_file_location("_v13_config_impl", _path)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

for _name, _val in vars(_mod).items():
    if not _name.startswith("_"):
        globals()[_name] = _val
